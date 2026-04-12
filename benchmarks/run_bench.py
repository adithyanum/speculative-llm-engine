from models.loader import ModelLoader
from engine.draft import generate_response
from metrics.logger import MetricsLogger

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

prompts = [
    "Who discovered penicillin?",
    "Explain binary search in simple terms",
    "Explain stack vs queue in Malayalam"
]

def run_baseline():
    loader = ModelLoader(MODEL_NAME)
    model, tokenizer = loader.load()
    logger = MetricsLogger()

    for prompt in prompts:
        result = generate_response(model, tokenizer, prompt)
        result["prompt"] = prompt
        result["mode"] = "baseline"
        logger.log(result)
        print(f"✓ {prompt[:40]} | {result['tokens_per_sec']:.2f} tok/s")

if __name__ == "__main__":
    run_baseline()