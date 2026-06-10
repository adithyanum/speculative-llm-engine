from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import Optional
import torch, logging, os

from engine.speculative import speculative_decode
from engine.draft import generate_response
from metrics.logger import MetricsLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

state = {
    "draft_model":   None,
    "target_model":  None,
    "tokenizer":     None,
    "device":        None,
    "models_loaded": False,
}

metrics_logger = MetricsLogger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Models are pre-loaded externally — lifespan does nothing
    if state["models_loaded"]:
        logger.info("✅ Models pre-loaded — skipping lifespan load")
    else:
        logger.error("❌ Models not loaded! Inject before starting server.")
    yield
    logger.info("Shutting down...")
    torch.cuda.empty_cache()

app = FastAPI(title="Speculative Decoding Engine", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: Optional[int] = Field(default=200, ge=1, le=1000)
    mode: Optional[str] = Field(default="speculative", pattern="^(speculative|baseline)$")
    log: Optional[bool] = Field(default=False)
    
class GenerateResponse(BaseModel):
    response: str
    mode: str
    tokens_generated: int
    tokens_per_sec: float
    latency: float
    acceptance_rate: Optional[float] = None
    cycles: Optional[int] = None
    model: str

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    device: Optional[str]

class StatsResponse(BaseModel):
    total_runs: int
    speculative_runs: int
    baseline_runs: int
    avg_tokens_per_sec: Optional[float]
    avg_acceptance_rate: Optional[float]
    best_tokens_per_sec: Optional[float]
    logs_available: int

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if state["models_loaded"] else "degraded",
        models_loaded=state["models_loaded"],
        device=state["device"],
    )

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if not state["models_loaded"]:
        raise HTTPException(status_code=503, detail="Models not loaded.")
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
        else:
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

        if req.log:
            metrics_logger.log({**result, "prompt": req.prompt, "model": model_label})

            
        return GenerateResponse(
            response=result["response"], mode=result["mode"],
            tokens_generated=result["tokens_generated"],
            tokens_per_sec=result["tokens_per_sec"],
            latency=result["latency"],
            acceptance_rate=result.get("acceptance_rate"),
            cycles=result.get("cycles"), model=model_label,
        )
    except Exception as e:
        logger.error(f"/generate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=StatsResponse)
def stats():
    try:
        logs = metrics_logger.read_logs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if not logs:
        return StatsResponse(total_runs=0, speculative_runs=0, baseline_runs=0,
            avg_tokens_per_sec=None, avg_acceptance_rate=None,
            best_tokens_per_sec=None, logs_available=0)
    speculative = [l for l in logs if l.get("mode") == "speculative"]
    baseline    = [l for l in logs if l.get("mode") == "baseline"]
    all_tps = [l["tokens_per_sec"] for l in logs if "tokens_per_sec" in l]
    spec_ar = [l["acceptance_rate"] for l in speculative if l.get("acceptance_rate") is not None]
    return StatsResponse(
        total_runs=len(logs), speculative_runs=len(speculative), baseline_runs=len(baseline),
        avg_tokens_per_sec=round(sum(all_tps)/len(all_tps), 3) if all_tps else None,
        avg_acceptance_rate=round(sum(spec_ar)/len(spec_ar), 3) if spec_ar else None,
        best_tokens_per_sec=round(max(all_tps), 3) if all_tps else None,
        logs_available=len(logs),
    )