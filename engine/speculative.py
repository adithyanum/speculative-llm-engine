import torch
import torch.nn.functional as F
import time


def _get_probs(logits, token_id):
    """Get probability of a specific token from logits"""
    probs = F.softmax(logits, dim=-1)
    return probs[token_id].item()


def _resample(p_logits, q_logits):
    """Sample from max(0, p-q) normalized — the corrected distribution"""
    p_probs = F.softmax(p_logits, dim=-1)
    q_probs = F.softmax(q_logits, dim=-1)
    corrected = torch.clamp(p_probs - q_probs, min=0)
    corrected = corrected / corrected.sum()  # normalize
    return torch.multinomial(corrected, num_samples=1).item()


def speculative_decode(draft_model, target_model, tokenizer,
                       prompt, k=4, max_new_tokens=200):
    """
    Speculative decoding loop.
    Draft model generates k tokens, target model verifies in one forward pass.
    Acceptance/rejection preserves target model output distribution exactly.
    """

    device = next(target_model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(target_model.device)

    generated_tokens = 0
    accepted_total = 0
    cycles = 0
    start_time = time.time()

    while generated_tokens < max_new_tokens:
        # ── Step 1: Draft generates k tokens ──────────────────────────
        draft_input = input_ids.clone()
        draft_tokens = []
        draft_logits_list = []

        with torch.no_grad():
            for _ in range(k):
                out = draft_model(draft_input)
                next_logits = out.logits[0, -1, :]  # last token position
                next_token = torch.multinomial(
                    F.softmax(next_logits, dim=-1), num_samples=1
                ).item()
                draft_tokens.append(next_token)
                draft_logits_list.append(next_logits)
                draft_input = torch.cat([
                    draft_input,
                    torch.tensor([[next_token]]).to(draft_input.device)
                ], dim=1)

        # ── Step 2: Target verifies all k tokens in ONE forward pass ───
        target_input = torch.cat([
            input_ids,
            torch.tensor([draft_tokens]).to(input_ids.device)
        ], dim=1)

        with torch.no_grad():
            target_out = target_model(target_input)

        # target logits at each draft position
        # position -k-1 to -2 are the verification positions
        target_logits_list = [
            target_out.logits[0, -(k + 1) + i, :]
            for i in range(k)
        ]

        # ── Step 3: Accept or reject each draft token ──────────────────
        accepted = 0
        new_token = None

        for i in range(k):
            token_id = draft_tokens[i]
            q_prob = _get_probs(draft_logits_list[i], token_id)
            p_prob = _get_probs(target_logits_list[i], token_id)

            acceptance_prob = min(1.0, p_prob / (q_prob + 1e-8))
            u = torch.rand(1).item()

            if u < acceptance_prob:
                # accept this draft token
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[token_id]]).to(input_ids.device)
                ], dim=1)
                accepted += 1
                generated_tokens += 1
            else:
                # reject — resample from corrected distribution
                new_token = _resample(target_logits_list[i], draft_logits_list[i])
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[new_token]]).to(input_ids.device)
                ], dim=1)
                generated_tokens += 1
                break  # discard remaining draft tokens

        # ── Step 4: Bonus token if all k accepted ──────────────────────
        if accepted == k:
            bonus_logits = target_out.logits[0, -1, :]
            bonus_token = torch.multinomial(
                F.softmax(bonus_logits, dim=-1), num_samples=1
            ).item()
            input_ids = torch.cat([
                input_ids,
                torch.tensor([[bonus_token]]).to(input_ids.device)
            ], dim=1)
            generated_tokens += 1

        accepted_total += accepted
        cycles += 1

        # stop if EOS
        if input_ids[0, -1].item() == tokenizer.eos_token_id:
            break

    # ── Metrics ────────────────────────────────────────────────────────
    latency = time.time() - start_time
    total_draft_tokens = cycles * k
    acceptance_rate = accepted_total / total_draft_tokens if total_draft_tokens > 0 else 0

    response = tokenizer.decode(
        input_ids[0][tokenizer(prompt, return_tensors="pt").input_ids.shape[1]:],
        skip_special_tokens=True
    )

    return {
        "response": response,
        "latency": round(latency, 3),
        "tokens_generated": generated_tokens,
        "tokens_per_sec": round(generated_tokens / latency, 2),
        "acceptance_rate": round(acceptance_rate, 3),
        "cycles": cycles,
        "mode": "speculative"
    }