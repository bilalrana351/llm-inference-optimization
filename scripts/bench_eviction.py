"""Measure KV-cache eviction quality, memory, and decode speed.

Two policies operate on transformers.DynamicCache:

  sink_window  keep a few first tokens and the newest tokens
  score        keep sinks, a small recent region, and the remaining tokens
               with the largest cumulative attention mass

Quality is measured on one fixed WikiText-2 token stream. The first tokens fill
and warm the cache; perplexity is reported only on the following evaluation
tokens. Performance uses synthetic KV tensors at exact context lengths. Values
do not affect attention runtime, so this reaches long contexts without paying
the quadratic cost of replaying a 120k-token prompt for every configuration.

Usage:
    python scripts/bench_eviction.py --mode quality --overwrite
    python scripts/bench_eviction.py --mode speed --overwrite
    python scripts/bench_eviction.py --mode all --overwrite
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import statistics
import time
from datetime import datetime, timezone
from typing import Iterable

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

from bench_common import bytes_to_mib, print_env, reset_peak_vram


def write_rows(path: str, rows: list[dict], overwrite: bool) -> None:
    if os.path.exists(path) and not overwrite:
        raise SystemExit(f"{path} exists, pass --overwrite to replace it")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def cache_tensors(cache: DynamicCache) -> Iterable[torch.Tensor]:
    for key, value in zip(cache.key_cache, cache.value_cache):
        yield key
        yield value


def cache_bytes(cache: DynamicCache) -> int:
    return sum(t.numel() * t.element_size() for t in cache_tensors(cache))


def cache_tokens(cache: DynamicCache) -> int:
    return cache.key_cache[0].shape[-2] if cache.key_cache else 0


def chunk_causal_mask(cache: DynamicCache, width: int, dtype: torch.dtype) -> torch.Tensor:
    """A 4D mask for compacted old keys followed by one new token chunk.

    Retained keys may have large, non-contiguous absolute positions, but every
    one is older than the current chunk and must remain visible. Only the new
    chunk needs a triangular mask. Relying on the model's compact KV indices
    after eviction would mistake later tokens in this chunk for old tokens and
    leak future targets into the perplexity calculation.
    """
    old_length = cache_tokens(cache)
    mask = torch.zeros((1, 1, width, old_length + width), device="cuda", dtype=dtype)
    local = torch.triu(
        torch.full((width, width), torch.finfo(dtype).min, device="cuda", dtype=dtype),
        diagonal=1,
    )
    mask[:, :, :, old_length:] = local
    return mask


def index_cache_layer(cache: DynamicCache, layer: int, keep: torch.Tensor) -> None:
    cache.key_cache[layer] = cache.key_cache[layer].index_select(-2, keep)
    cache.value_cache[layer] = cache.value_cache[layer].index_select(-2, keep)


class SinkWindowEvictor:
    def __init__(self, sink_tokens: int = 4):
        self.sink_tokens = sink_tokens

    def prune(self, cache: DynamicCache, budget: int) -> None:
        if not cache.key_cache or cache_tokens(cache) <= budget:
            return
        if budget <= self.sink_tokens:
            raise ValueError("budget must exceed sink_tokens")
        length = cache_tokens(cache)
        device = cache.key_cache[0].device
        sinks = torch.arange(self.sink_tokens, device=device)
        recent = torch.arange(length - (budget - self.sink_tokens), length, device=device)
        keep = torch.cat((sinks, recent))
        for layer in range(len(cache.key_cache)):
            index_cache_layer(cache, layer, keep)


class AttentionScoreEvictor:
    """Cumulative-attention heavy hitters with pinned sinks and recent tokens."""

    def __init__(self, sink_tokens: int = 4, recent_tokens: int = 32):
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens
        self.scores: list[torch.Tensor] = []

    def _keep(self, score: torch.Tensor, budget: int) -> torch.Tensor:
        length = score.numel()
        if length <= budget:
            return torch.arange(length, device=score.device)
        sink_count = min(self.sink_tokens, budget)
        recent_count = min(self.recent_tokens, budget - sink_count)
        fixed = torch.cat((
            torch.arange(sink_count, device=score.device),
            torch.arange(length - recent_count, length, device=score.device),
        )).unique()
        remaining = budget - fixed.numel()
        if remaining > 0:
            candidates = torch.ones(length, device=score.device, dtype=torch.bool)
            candidates[fixed] = False
            candidate_idx = candidates.nonzero(as_tuple=False).flatten()
            chosen = candidate_idx[score[candidate_idx].topk(remaining).indices]
            fixed = torch.cat((fixed, chosen))
        return fixed.sort().values

    def prune(self, cache: DynamicCache, budget: int) -> None:
        if not cache.key_cache or cache_tokens(cache) <= budget:
            return
        for layer in range(len(cache.key_cache)):
            keep = self._keep(self.scores[layer], budget)
            index_cache_layer(cache, layer, keep)
            self.scores[layer] = self.scores[layer].index_select(0, keep)

    def update_and_prune(
        self,
        cache: DynamicCache,
        attentions: tuple[torch.Tensor, ...],
        budget: int,
    ) -> None:
        if not self.scores:
            self.scores = [
                torch.zeros(attention.shape[-1], device=attention.device, dtype=torch.float32)
                for attention in attentions
            ]
        for layer, attention in enumerate(attentions):
            length = attention.shape[-1]
            old = self.scores[layer]
            if old.numel() < length:
                old = F.pad(old, (0, length - old.numel()))
            elif old.numel() > length:
                raise RuntimeError(
                    f"score/cache mismatch at layer {layer}: {old.numel()} scores, {length} keys"
                )
            # Average batches and query heads, then add the mass from every
            # query in this chunk. The score stays aligned with the KV axis.
            mass = attention.detach().float().mean(dim=(0, 1)).sum(dim=0)
            self.scores[layer] = old + mass
        self.prune(cache, budget)


def safe_qwen2_eager_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position=None,
    position_embeddings=None,
):
    """Pinned Qwen2 eager attention with QK scores formed in fp32.

    transformers 4.46.3 forms the QK matmul in fp16, then upcasts only for the
    softmax. On torch 2.11.0+cu128 that materialized fp16 score tensor can
    overflow before the upcast and produces NaN logits even in the full-cache
    control. SDPA scales safely but cannot return attention weights. This is the
    same pinned eager implementation with only the QK matmul moved to fp32.
    """
    batch, query_length, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    query_states = query_states.view(
        batch, query_length, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        batch, query_length, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        batch, query_length, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    scores = torch.matmul(
        query_states.float(), key_states.transpose(2, 3).float()
    ) / math.sqrt(self.head_dim)
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : key_states.shape[-2]]
    probabilities = F.softmax(scores, dim=-1, dtype=torch.float32)
    probabilities = F.dropout(
        probabilities, p=self.attention_dropout, training=self.training
    )
    attention_output = torch.matmul(probabilities.to(value_states.dtype), value_states)
    attention_output = attention_output.transpose(1, 2).contiguous()
    attention_output = attention_output.reshape(batch, query_length, self.hidden_size)
    attention_output = self.o_proj(attention_output)
    returned_weights = probabilities if output_attentions else None
    return attention_output, returned_weights, past_key_value


def load_model(model_name: str, attention: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        attn_implementation=attention,
        device_map={"": 0},
    )
    model.eval()
    if attention == "eager":
        for layer in model.model.layers:
            layer.self_attn.forward = safe_qwen2_eager_forward.__get__(
                layer.self_attn, type(layer.self_attn)
            )
    return model


def wikitext_tokens(model_name: str, needed: int) -> torch.Tensor:
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    pieces: list[int] = []
    for part in dataset["text"]:
        if not part.strip():
            continue
        pieces.extend(tokenizer(part + "\n\n", add_special_tokens=False).input_ids)
        if len(pieces) >= needed:
            break
    tokens = torch.tensor(pieces[:needed], dtype=torch.long)
    if tokens.numel() < needed:
        raise RuntimeError(f"WikiText produced {tokens.numel()} tokens, need {needed}")
    return tokens[:needed].contiguous()


def run_quality_config(
    model,
    model_name: str,
    token_ids: torch.Tensor,
    policy_name: str,
    budget: int,
    warmup_tokens: int,
    eval_tokens: int,
    chunk_size: int,
    sink_tokens: int,
    recent_tokens: int,
) -> dict:
    cache = DynamicCache()
    sink = SinkWindowEvictor(sink_tokens)
    score = AttentionScoreEvictor(sink_tokens, recent_tokens)
    total_predictions = warmup_tokens + eval_tokens
    nll_sum = 0.0
    scored = 0
    started = time.perf_counter()
    reset_peak_vram()

    with torch.inference_mode():
        for offset in range(0, total_predictions, chunk_size):
            end = min(offset + chunk_size, total_predictions)
            width = end - offset
            if policy_name != "full":
                target_before = max(sink_tokens + 1, budget - width)
                if policy_name == "sink_window":
                    sink.prune(cache, target_before)
                else:
                    score.prune(cache, target_before)

            inputs = token_ids[offset:end].unsqueeze(0).cuda(non_blocking=True)
            targets = token_ids[offset + 1:end + 1].cuda(non_blocking=True)
            positions = torch.arange(offset, end, device="cuda", dtype=torch.long)
            attention_mask = chunk_causal_mask(cache, width, model.dtype)
            outputs = model(
                input_ids=inputs,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                cache_position=positions,
                output_attentions=policy_name == "score",
                return_dict=True,
            )
            cache = outputs.past_key_values
            if policy_name == "sink_window":
                sink.prune(cache, budget)
            elif policy_name == "score":
                if outputs.attentions is None:
                    raise RuntimeError("attention-score policy requires eager attentions")
                score.update_and_prune(cache, outputs.attentions, budget)

            target_positions = torch.arange(offset + 1, end + 1, device="cuda")
            selected = target_positions > warmup_tokens
            if selected.any():
                logits = outputs.logits[0, selected]
                labels = targets[selected]
                nll_sum += F.cross_entropy(logits.float(), labels, reduction="sum").item()
                scored += labels.numel()
            del outputs, inputs, targets, attention_mask

    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    nll = nll_sum / scored
    row = {
        "policy": policy_name,
        "budget": budget if policy_name != "full" else 0,
        "sink_tokens": sink_tokens if policy_name != "full" else 0,
        "recent_tokens": recent_tokens if policy_name == "score" else 0,
        "warmup_tokens": warmup_tokens,
        "eval_tokens": scored,
        "chunk_size": chunk_size,
        "nll": nll,
        "perplexity": math.exp(nll),
        "cache_tokens": cache_tokens(cache),
        "cache_mib": bytes_to_mib(cache_bytes(cache)),
        "allocated_mib": bytes_to_mib(torch.cuda.memory_allocated()),
        "peak_allocated_mib": bytes_to_mib(torch.cuda.max_memory_allocated()),
        "seconds": seconds,
        "evaluated_tokens_per_sec": scored / seconds,
        "model": model_name,
        "attention_backend": "eager_fp32_qk",
        "dataset": "wikitext-2-raw-v1:test",
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    del cache, sink, score
    gc.collect()
    torch.cuda.empty_cache()
    return row


def make_synthetic_cache(model, length: int) -> DynamicCache:
    config = model.config
    kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    cache = DynamicCache()
    for _ in range(config.num_hidden_layers):
        shape = (1, kv_heads, length, head_dim)
        cache.key_cache.append(torch.zeros(shape, device="cuda", dtype=torch.float16))
        cache.value_cache.append(torch.zeros(shape, device="cuda", dtype=torch.float16))
    cache._seen_tokens = length
    return cache


def run_speed_config(
    model,
    model_name: str,
    policy_name: str,
    budget: int,
    context_tokens: int,
    warmup: int,
    repeats: int,
    sink_tokens: int,
) -> list[dict]:
    retained = context_tokens if policy_name == "full" else min(context_tokens, budget)
    cache = make_synthetic_cache(model, retained)
    cache._seen_tokens = context_tokens
    sink = SinkWindowEvictor(sink_tokens)
    baseline_allocated = torch.cuda.memory_allocated() - cache_bytes(cache)
    reset_peak_vram()
    token = torch.tensor([[17]], device="cuda", dtype=torch.long)

    rows = []
    with torch.inference_mode():
        for step in range(warmup + repeats):
            position = context_tokens + step
            torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = model(
                input_ids=token,
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.tensor([position], device="cuda"),
                return_dict=True,
            )
            cache = outputs.past_key_values
            if policy_name == "sink_window":
                sink.prune(cache, budget)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000
            if step >= warmup:
                rows.append({
                    "policy": policy_name,
                    "budget": budget if policy_name != "full" else 0,
                    "context_tokens": context_tokens,
                    "repeat": step - warmup,
                    "decode_ms": elapsed_ms,
                    "decode_tokens_per_sec": 1000 / elapsed_ms,
                    "cache_tokens": cache_tokens(cache),
                    "cache_mib": bytes_to_mib(cache_bytes(cache)),
                    "allocated_above_model_mib": bytes_to_mib(
                        torch.cuda.memory_allocated() - baseline_allocated
                    ),
                    "reserved_mib": bytes_to_mib(torch.cuda.memory_reserved()),
                    "peak_above_model_mib": bytes_to_mib(
                        torch.cuda.max_memory_allocated() - baseline_allocated
                    ),
                    "model": model_name,
                    "attention_backend": "sdpa",
                    "gpu_name": torch.cuda.get_device_name(0),
                    "torch_version": torch.__version__,
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                })
            del outputs

    del cache, sink, token
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--mode", choices=("quality", "speed", "all"), default="all")
    parser.add_argument("--quality-csv", default="results/eviction_quality.csv")
    parser.add_argument("--speed-csv", default="results/eviction_speed.csv")
    parser.add_argument("--quality-budgets", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--quality-warmup", type=int, default=4096)
    parser.add_argument("--quality-eval", type=int, default=2048)
    parser.add_argument("--quality-chunk", type=int, default=32)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--score-recent", type=int, default=32)
    parser.add_argument("--speed-contexts", type=int, nargs="+",
                        default=[512, 4096, 16384, 65536, 120000])
    parser.add_argument("--speed-budgets", type=int, nargs="+", default=[512, 2048, 4096])
    parser.add_argument("--speed-warmup", type=int, default=3)
    parser.add_argument("--speed-repeats", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device.")
    if min(args.quality_budgets + args.speed_budgets) <= args.sink_tokens:
        raise SystemExit("every cache budget must exceed sink_tokens")
    if args.score_recent + args.sink_tokens >= min(args.quality_budgets):
        raise SystemExit("score_recent + sink_tokens must be smaller than every quality budget")

    torch.manual_seed(17)
    print_env()

    if args.mode in ("quality", "all"):
        needed = args.quality_warmup + args.quality_eval + 1
        tokens = wikitext_tokens(args.model, needed)
        model = load_model(args.model, "eager")
        quality_rows = []
        configurations = [("full", 0)]
        configurations += [("sink_window", budget) for budget in args.quality_budgets]
        configurations += [("score", budget) for budget in args.quality_budgets]
        for policy, budget in configurations:
            print(f"quality: {policy} budget={budget or 'full'}", flush=True)
            row = run_quality_config(
                model, args.model, tokens, policy, budget, args.quality_warmup,
                args.quality_eval, args.quality_chunk, args.sink_tokens,
                args.score_recent,
            )
            quality_rows.append(row)
            print(f"  ppl={row['perplexity']:.3f}, cache={row['cache_tokens']} tokens, "
                  f"{row['seconds']:.1f} s", flush=True)
        write_rows(args.quality_csv, quality_rows, args.overwrite)
        print(f"wrote {args.quality_csv}")
        del model, tokens
        gc.collect()
        torch.cuda.empty_cache()

    if args.mode in ("speed", "all"):
        model = load_model(args.model, "sdpa")
        speed_rows = []
        configurations = [("full", 0)]
        configurations += [("sink_window", budget) for budget in args.speed_budgets]
        for policy, budget in configurations:
            for context in args.speed_contexts:
                print(f"speed: {policy} budget={budget or 'full'}, context={context}", flush=True)
                rows = run_speed_config(
                    model, args.model, policy, budget, context, args.speed_warmup,
                    args.speed_repeats, args.sink_tokens,
                )
                speed_rows.extend(rows)
                median_ms = statistics.median(row["decode_ms"] for row in rows)
                print(f"  median={median_ms:.2f} ms, {1000 / median_ms:.2f} tok/s", flush=True)
        write_rows(args.speed_csv, speed_rows, args.overwrite)
        print(f"wrote {args.speed_csv}")


if __name__ == "__main__":
    main()
