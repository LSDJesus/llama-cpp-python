#!/usr/bin/env python3
"""
kv_injection_v2.py — KV Cache Semantic Injection (Fictional Prompt)
====================================================================
Tests whether factual details stored in the KV cache from a prefix can
leak into generation when the creative prompt contains NO details.

Key improvements over v1:
  - Pure fiction: zero pretraining contamination
  - /no_think: all output is story, no reasoning tokens
  - Highly specific, unusual details that wouldn't appear by chance
  - Simple name+situation prompt vs. rich character/setting details

Conditions:
  A) BASELINE:        bare prompt only (no details anywhere)
  B) FULL INJECTION:  details prefix + prompt (KV at all layers)
  C) CRITICAL-ONLY:   details prefix with resilient layers skipped
                      (KV for details exists only at critical layers)
  D) RESILIENT-ONLY:  details prefix with critical layers skipped
                      (KV for details exists only at resilient layers)
  E) DIRECT PROMPT:   details embedded directly in prompt text

Detection: Count which injected details appear in the output.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from typing import Dict, List, Optional

from llama_cpp import Llama

# ═══════════════════════════════════════════════════════════════════════
#  PROMPTS — Pure fiction, zero pretraining contamination
# ═══════════════════════════════════════════════════════════════════════

# Details to inject via KV cache — all highly specific and unusual
DETAILS_PREFIX = """\
Character and setting details for the story:
- The woman's name is Elara Voss. She is 34 years old.
- She has silver-streaked auburn hair and a crescent-shaped scar on her left palm.
- The forest is called Thornwood. It sits in a valley called the Greywander Basin.
- It is late November, the first frost has already come.
- Her companion is a wolfhound named Cassius with one blue eye and one amber eye.
- She carries a brass compass that belonged to her grandmother, Mirabel.\
"""

# Bare prompt — gives only a name and situation, zero specifics
BARE_PROMPT = """\
Write a short story about a woman named Elara walking through a forest at dusk. \
Three paragraphs. Rich sensory details. /no_think\
"""

# Direct prompt — details embedded in the instruction itself
DIRECT_PROMPT = """\
Write a short story about Elara Voss, a 34-year-old woman with silver-streaked \
auburn hair and a crescent-shaped scar on her left palm. She walks through a \
forest called Thornwood in the Greywander Basin at dusk in late November, after \
the first frost. Her companion is a wolfhound named Cassius with one blue eye \
and one amber eye. She carries a brass compass that belonged to her grandmother, \
Mirabel. Three paragraphs. Rich sensory details. /no_think\
"""

# Chat template wrapper for Qwen3
def wrap_chat(user_content: str) -> str:
    """Wrap user content in Qwen3 chat template for raw completion."""
    return (
        "<|im_start|>user\n"
        f"{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

KV_SEPARATOR = "\n\n---\n\n"


# ═══════════════════════════════════════════════════════════════════════
#  DETAIL DETECTION — highly specific tokens that wouldn't appear by chance
# ═══════════════════════════════════════════════════════════════════════

DETAIL_CHECKS = {
    "age_34":           (r"\b34\b", "Age: 34"),
    "silver_hair":      (r"silver", "Silver (hair)"),
    "auburn_hair":      (r"auburn", "Auburn (hair)"),
    "crescent_scar":    (r"crescent", "Crescent scar"),
    "thornwood":        (r"[Tt]hornwood", "Forest: Thornwood"),
    "greywander":       (r"[Gg]reywander", "Valley: Greywander"),
    "november":         (r"[Nn]ovember", "Month: November"),
    "frost":            (r"frost", "First frost"),
    "wolfhound":        (r"wolfhound", "Wolfhound"),
    "cassius":          (r"[Cc]assius", "Companion: Cassius"),
    "heterochromia":    (r"blue.{1,30}amber|amber.{1,30}blue|one.{1,10}eye", "Heterochromia eyes"),
    "brass_compass":    (r"brass.{1,15}compass|compass.{1,15}brass", "Brass compass"),
    "mirabel":          (r"[Mm]irabel", "Grandmother: Mirabel"),
}


def score_details(text: str) -> Dict[str, dict]:
    """Check which injected details appear in the output."""
    results = {}
    for key, (pattern, label) in DETAIL_CHECKS.items():
        match = re.search(pattern, text)
        results[key] = {
            "found": bool(match),
            "label": label,
            "match": match.group(0) if match else None,
        }
    return results


# ═══════════════════════════════════════════════════════════════════════
#  GENERATION WITH LAYER CONTROL
# ═══════════════════════════════════════════════════════════════════════

def generate_with_prefix(
    llm: Llama,
    prefix_text: str,
    prompt_text: str,
    max_tokens: int,
    skip_layers_during_prefix: Optional[List[int]] = None,
) -> str:
    """
    Process prefix + prompt sequentially with optional layer skipping
    during prefix processing only.

    1. Set layer_skip for prefix → KV entries only at non-skipped layers
    2. Clear layer_skip for prompt → full processing
    3. Generate with all layers active
    """
    llm._ctx.memory_clear(True)
    llm.reset()

    # Phase 1: Process prefix with optional layer skipping
    if skip_layers_during_prefix:
        llm.set_layer_skip(skip_layers_during_prefix)

    # Prefix gets BOS token
    prefix_tokens = llm.tokenize(prefix_text.encode(), add_bos=True)
    llm.eval(prefix_tokens)

    # Phase 2: Process prompt with all layers active
    llm.set_layer_skip(None)

    prompt_tokens = llm.tokenize(prompt_text.encode(), add_bos=False)
    llm.eval(prompt_tokens)

    # Phase 3: Generate
    output_tokens: List[int] = []
    eos_id = llm.token_eos()

    for _ in range(max_tokens):
        token = llm.sample(
            temp=0.0,
            top_k=-1,
            top_p=1.0,
            repeat_penalty=1.0,
            penalty_last_n=0,
        )
        if token == eos_id:
            break
        output_tokens.append(token)
        llm.eval([token])

    return llm.detokenize(output_tokens).decode("utf-8", errors="replace")


def generate_simple(llm: Llama, prompt: str, max_tokens: int) -> str:
    """Simple full-prompt generation, no prefix tricks."""
    llm._ctx.memory_clear(True)
    llm.reset()

    tokens = llm.tokenize(prompt.encode(), add_bos=True)
    llm.eval(tokens)

    output_tokens: List[int] = []
    eos_id = llm.token_eos()

    for _ in range(max_tokens):
        token = llm.sample(
            temp=0.0,
            top_k=-1,
            top_p=1.0,
            repeat_penalty=1.0,
            penalty_last_n=0,
        )
        if token == eos_id:
            break
        output_tokens.append(token)
        llm.eval([token])

    return llm.detokenize(output_tokens).decode("utf-8", errors="replace")


# ═══════════════════════════════════════════════════════════════════════
#  EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════

def run_condition(llm: Llama, label: str, generator, max_tokens: int) -> dict:
    """Run one experimental condition and collect metrics."""
    print(f"\n{'═' * 72}")
    print(f"  {label}")
    print(f"{'═' * 72}")

    t0 = time.time()
    text = generator()
    elapsed = time.time() - t0

    details = score_details(text)
    n_found = sum(1 for d in details.values() if d["found"])
    n_total = len(details)
    n_tok = len(llm.tokenize(text.encode(), add_bos=False))

    print(f"  {n_tok} tokens in {elapsed:.1f}s  |  Details found: {n_found}/{n_total}")
    for key, info in details.items():
        symbol = "+" if info["found"] else "-"
        extra = f' → "{info["match"]}"' if info["found"] else ""
        print(f"    {symbol} {info['label']}{extra}")

    # Print first ~500 chars
    preview = text.strip()[:500]
    print(f"  {'─' * 60}")
    for line in preview.split("\n"):
        print(f"  {line}")
    if len(text.strip()) > 500:
        print(f"  ...")

    return {
        "label": label,
        "text": text,
        "n_tokens": n_tok,
        "time_s": round(elapsed, 2),
        "details": {k: v["found"] for k, v in details.items()},
        "n_details_found": n_found,
        "n_details_total": n_total,
    }


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def parse_layer_list(s: str) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="KV Cache Semantic Injection v2 — Fictional Prompt"
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--main-gpu", type=int, default=0)
    ap.add_argument("--single-gpu", action="store_true")
    ap.add_argument("--critical-layers", type=str, default="")
    ap.add_argument("--resilient-layers", type=str, default="")
    ap.add_argument("--output", default="scripts/kv_injection_v2_results.json")
    args = ap.parse_args()

    critical = parse_layer_list(args.critical_layers)
    resilient = parse_layer_list(args.resilient_layers)

    # Load model
    print(f"Loading model: {args.model}")
    extra = {}
    if args.single_gpu:
        extra["tensor_split"] = [1.0, 0.0]

    llm = Llama(
        model_path=args.model,
        n_gpu_layers=-1,
        n_ctx=args.n_ctx,
        main_gpu=args.main_gpu,
        verbose=False,
        **extra,
    )

    n_layer = llm.get_n_layer()
    print(f"  Layers: {n_layer}  |  Embd: {llm.n_embd()}")
    print(f"  Critical:  {critical or '(auto)'}")
    print(f"  Resilient: {resilient or '(auto)'}")

    # Build chat-wrapped prompts
    bare_chat = wrap_chat(BARE_PROMPT)
    direct_chat = wrap_chat(DIRECT_PROMPT)
    details_prefix_chat = wrap_chat(DETAILS_PREFIX + KV_SEPARATOR + BARE_PROMPT)

    # For prefix injection, we split at the KV_SEPARATOR boundary:
    # prefix = details text (gets layer-skipped)
    # prompt = the actual chat-wrapped bare prompt
    prefix_part = DETAILS_PREFIX + KV_SEPARATOR
    prompt_part = wrap_chat(BARE_PROMPT)
    # But we need BOS at the very start, so prefix_part goes first with add_bos=True

    # Determine skip lists
    all_layers = set(range(n_layer))
    if critical and resilient:
        skip_for_critical_only = resilient
        skip_for_resilient_only = critical
    elif critical:
        skip_for_critical_only = sorted(all_layers - set(critical))
        skip_for_resilient_only = critical
    elif resilient:
        skip_for_critical_only = resilient
        skip_for_resilient_only = sorted(all_layers - set(resilient))
    else:
        skip_for_critical_only = []
        skip_for_resilient_only = []

    results = []

    # ── A) BASELINE: bare prompt, no details ─────────────────────────
    results.append(run_condition(
        llm,
        "A) BASELINE — bare prompt, no injected details",
        lambda: generate_simple(llm, bare_chat, args.max_tokens),
        args.max_tokens,
    ))

    # ── B) FULL INJECTION: details prefix → KV at all layers ─────────
    results.append(run_condition(
        llm,
        "B) FULL KV INJECTION — details prefix, all layers",
        lambda: generate_with_prefix(
            llm, prefix_part, prompt_part,
            args.max_tokens, skip_layers_during_prefix=None,
        ),
        args.max_tokens,
    ))

    # ── C) CRITICAL-ONLY: details KV only at critical layers ─────────
    if skip_for_critical_only:
        results.append(run_condition(
            llm,
            f"C) CRITICAL-ONLY — details KV at {len(set(range(n_layer)) - set(skip_for_critical_only))} critical layers",
            lambda: generate_with_prefix(
                llm, prefix_part, prompt_part,
                args.max_tokens, skip_layers_during_prefix=skip_for_critical_only,
            ),
            args.max_tokens,
        ))

    # ── D) RESILIENT-ONLY: details KV only at resilient layers ───────
    if skip_for_resilient_only:
        results.append(run_condition(
            llm,
            f"D) RESILIENT-ONLY — details KV at {len(set(range(n_layer)) - set(skip_for_resilient_only))} resilient layers",
            lambda: generate_with_prefix(
                llm, prefix_part, prompt_part,
                args.max_tokens, skip_layers_during_prefix=skip_for_resilient_only,
            ),
            args.max_tokens,
        ))

    # ── E) DIRECT PROMPT: details in prompt text itself ──────────────
    results.append(run_condition(
        llm,
        "E) DIRECT PROMPT — details embedded in prompt text",
        lambda: generate_simple(llm, direct_chat, args.max_tokens),
        args.max_tokens,
    ))

    # ── Save ─────────────────────────────────────────────────────────
    output = {
        "model": args.model,
        "n_layer": n_layer,
        "max_tokens": args.max_tokens,
        "critical_layers": critical,
        "resilient_layers": resilient,
        "prompts": {
            "details_prefix": DETAILS_PREFIX,
            "bare_prompt": BARE_PROMPT,
            "direct_prompt": DIRECT_PROMPT,
        },
        "conditions": results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print(f"  SUMMARY — KV Semantic Injection v2 (Fictional)")
    print(f"{'═' * 72}")
    print(f"  {'Condition':<58} {'Details':>7}")
    print(f"  {'─' * 58} {'─' * 7}")
    for r in results:
        print(f"  {r['label']:<58} {r['n_details_found']:>3}/{r['n_details_total']}")

    # Compare
    a_n = results[0]["n_details_found"]
    b_n = results[1]["n_details_found"]
    e_n = results[-1]["n_details_found"]

    print(f"\n  KEY COMPARISONS:")
    print(f"    Baseline (A):         {a_n}/{results[0]['n_details_total']} details")
    print(f"    Full injection (B):   {b_n}/{results[1]['n_details_total']} details")
    print(f"    Direct prompt (E):    {e_n}/{results[-1]['n_details_total']} details")

    if b_n > a_n:
        print(f"\n  >>> KV INJECTION SURFACED {b_n - a_n} ADDITIONAL DETAIL(S)! <<<")
        # Show which ones leaked through
        a_details = results[0]["details"]
        b_details = results[1]["details"]
        leaked = [k for k in b_details if b_details[k] and not a_details.get(k, False)]
        if leaked:
            for k in leaked:
                print(f"      Leaked: {DETAIL_CHECKS[k][1]}")
    elif b_n == a_n:
        print(f"\n  → No additional details from KV injection ({a_n}={b_n})")
    else:
        print(f"\n  → KV injection reduced details ({a_n}→{b_n})")

    if len(results) >= 4:
        c_n = results[2]["n_details_found"]
        d_n = results[3]["n_details_found"]
        print(f"\n  LAYER SELECTIVITY:")
        print(f"    Critical-only (C): {c_n}/{results[2]['n_details_total']}")
        print(f"    Resilient-only (D): {d_n}/{results[3]['n_details_total']}")

    print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
