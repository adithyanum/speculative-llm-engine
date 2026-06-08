from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import Optional
import torch
import logging

from models.loader import ModelLoader
from engine.speculative import speculative_decode
from engine.draft import generate_response
from metrics.logger import MetricsLogger

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Model config ───────────────────────────────────────────────────────────────
DRAFT_MODEL_NAME  = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# ── Global state ───────────────────────────────────────────────────────────────
# Models are loaded once at startup and reused across all requests.
# Never reload per-request — way too slow for 7B models.
state = {
    "draft_model":   None,
    "target_model":  None,
    "tokenizer":     None,   # shared tokenizer (both Qwen models use same tokenizer)
    "device":        None,
    "models_loaded": False,
}

metrics_logger = MetricsLogger()


# ── Lifespan: load models on startup, cleanup on shutdown ─────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not state["models_loaded"]:
        logger.info("Loading models via lifespan...")
        try:
            draft_loader = ModelLoader(DRAFT_MODEL_NAME)
            state["draft_model"], state["tokenizer"] = draft_loader.load()
            target_loader = ModelLoader(TARGET_MODEL_NAME)
            state["target_model"], _ = target_loader.load()
            state["device"] = draft_loader.get_device()
            state["models_loaded"] = True
            logger.info(f"Models loaded on {state['device']}")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
    else:
        logger.info("✅ Models pre-loaded — skipping lifespan load")
    yield
    logger.info("Shutting down...")
    if state["draft_model"] is not None:
        del state["draft_model"]
    if state["target_model"] is not None:
        del state["target_model"]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── App init ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Speculative Decoding Inference Engine",
    description="FastAPI backend for speculative decoding with Qwen2.5 models",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this for production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ─────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input prompt for the model")
    max_new_tokens: Optional[int] = Field(default=200, ge=1, le=1000)
    mode: Optional[str] = Field(default="speculative", pattern="^(speculative|baseline)$")

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Explain binary search in simple terms",
                "max_new_tokens": 200,
                "mode": "speculative"
            }
        }


class GenerateResponse(BaseModel):
    response: str
    mode: str
    tokens_generated: int
    tokens_per_sec: float
    latency: float
    acceptance_rate: Optional[float] = None   # only for speculative mode
    cycles: Optional[int] = None              # only for speculative mode
    model: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    device: Optional[str]
    draft_model: str
    target_model: str


class StatsResponse(BaseModel):
    total_runs: int
    speculative_runs: int
    baseline_runs: int
    avg_tokens_per_sec: Optional[float]
    avg_acceptance_rate: Optional[float]       # speculative runs only
    best_tokens_per_sec: Optional[float]
    logs_available: int


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """
    Check if the server is alive and models are loaded.
    Use this before sending /generate requests.
    """
    return HealthResponse(
        status="ok" if state["models_loaded"] else "degraded",
        models_loaded=state["models_loaded"],
        device=state["device"],
        draft_model=DRAFT_MODEL_NAME,
        target_model=TARGET_MODEL_NAME,
    )


@app.post("/generate", response_model=GenerateResponse, tags=["Inference"])
def generate(req: GenerateRequest):
    """
    Run inference using speculative decoding or baseline (target-only) mode.

    - **speculative**: Draft model proposes k=4 tokens, target verifies in one pass. Faster.
    - **baseline**: Target model generates autoregressively. Slower but simpler.
    """
    if not state["models_loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded yet. Check /health and try again."
        )

    try:
        if req.mode == "speculative":
            result = speculative_decode(
                draft_model=state["draft_model"],
                target_model=state["target_model"],
                tokenizer=state["tokenizer"],
                prompt=req.prompt,
                max_new_tokens=req.max_new_tokens,
            )
            model_label = "Qwen2.5-0.5B+7B"

        else:  # baseline — target model only
            result = generate_response(
                model=state["target_model"],
                tokenizer=state["tokenizer"],
                prompt=req.prompt,
                max_new_tokens=req.max_new_tokens,
            )
            result["mode"] = "baseline"
            result["acceptance_rate"] = None
            result["cycles"] = None
            model_label = "Qwen2.5-7B"

        # Log to metrics/logs.json
        log_entry = {**result, "prompt": req.prompt, "model": model_label}
        metrics_logger.log(log_entry)

        return GenerateResponse(
            response=result["response"],
            mode=result["mode"],
            tokens_generated=result["tokens_generated"],
            tokens_per_sec=result["tokens_per_sec"],
            latency=result["latency"],
            acceptance_rate=result.get("acceptance_rate"),
            cycles=result.get("cycles"),
            model=model_label,
        )

    except Exception as e:
        logger.error(f"/generate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse, tags=["Metrics"])
def stats():
    """
    Aggregate statistics computed from metrics/logs.json.
    Includes avg tokens/sec, avg acceptance rate, and run counts by mode.
    """
    try:
        logs = metrics_logger.read_logs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read logs: {e}")

    if not logs:
        return StatsResponse(
            total_runs=0,
            speculative_runs=0,
            baseline_runs=0,
            avg_tokens_per_sec=None,
            avg_acceptance_rate=None,
            best_tokens_per_sec=None,
            logs_available=0,
        )

    speculative = [l for l in logs if l.get("mode") == "speculative"]
    baseline    = [l for l in logs if l.get("mode") == "baseline"]

    all_tps = [l["tokens_per_sec"] for l in logs if "tokens_per_sec" in l]
    spec_ar = [l["acceptance_rate"] for l in speculative if l.get("acceptance_rate") is not None]

    return StatsResponse(
        total_runs=len(logs),
        speculative_runs=len(speculative),
        baseline_runs=len(baseline),
        avg_tokens_per_sec=round(sum(all_tps) / len(all_tps), 3) if all_tps else None,
        avg_acceptance_rate=round(sum(spec_ar) / len(spec_ar), 3) if spec_ar else None,
        best_tokens_per_sec=round(max(all_tps), 3) if all_tps else None,
        logs_available=len(logs),
    )
