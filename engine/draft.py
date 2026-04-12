import time

def generate_response(model, tokenizer, prompt, max_new_tokens=300):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start_time = time.time()

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7
    )

    end_time = time.time()

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    latency = end_time - start_time
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs.shape[1] - input_length
    tokens_per_sec = generated_tokens / latency

    return {
        "response": response,
        "latency": latency,
        "tokens_generated": generated_tokens,
        "tokens_per_sec": tokens_per_sec
    }