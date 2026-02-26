#!/usr/bin/env python3
"""
kv_injection_experiment.py — KV Cache Semantic Injection
=========================================================
Tests whether factual information stored in the KV cache from a prefix
can influence generation when the creative prompt itself contains no facts.

Conditions:
  A) FACTLESS BASELINE:     factless_prompt → generate
     (No facts in prompt. Model relies purely on pretraining knowledge.)

  B) FULL KV INJECTION:     facts_prefix + factless_prompt → generate
     (Facts processed first → KV cache has fact tokens at ALL layers.
      Model attends to them during factless prompt processing.)

  C) CRITICAL-ONLY INJECTION: facts processed with resilient layers skipped
     → factless_prompt → generate
     (KV entries for facts exist ONLY at critical layers. At non-critical
      layers, the model can't "see" the fact tokens in attention.)

  D) ANTI-SELECTIVE INJECTION: facts processed with critical layers skipped
     → factless_prompt → generate
     (KV entries for facts exist ONLY at non-critical/resilient layers.
      The opposite of condition C — tests the inverse hypothesis.)

  E) SUPPRESSED INJECTION:  facts_prefix + suppression instruction
     + factless_prompt → generate
     (Facts in KV cache but model explicitly told to ignore them.
      Tests whether KV semantic state overpowers instruction following.)

  F) FULL PROMPT CONTROL:   original_prompt with embedded facts → generate
     (Standard prompt with facts. Quality ceiling reference.)

Metrics: Presence of 4 specific facts (April 10, 11:40 PM, 28°F, 1,517 dead)

Usage:
  python scripts/kv_injection_experiment.py \
    --model path/to/model.gguf \
    --critical-layers 12,13,15,16,18,19,22,25,28 \
    --resilient-layers 3,4,5,6,9,10,11,17,20,21,23,27,29,32,36

  (Layer lists come from the sensitivity analysis of the layer-skip experiment)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from typing import Dict, List, Optional, Sequence

from llama_cpp import Llama

# ═══════════════════════════════════════════════════════════════════════
#  PROMPTS
# ═══════════════════════════════════════════════════════════════════════

FACTS_ONLY = """\
Key facts about the RMS Titanic's maiden voyage:
- The ship departed Southampton on April 10, 1912.
- It struck an iceberg at exactly 11:40 PM on April 14, 1912.
- The water temperature was 28°F (−2°C).
- 1,517 people perished in the disaster.\
"""

FACTLESS_PROMPT = """\
Write three paragraphs describing the maiden voyage of the RMS Titanic \
in the style of Southern Gothic literature. Use a melancholy, elegiac tone. \
Include vivid sensory details, a haunting atmosphere, and themes of human \
frailty and hubris. The first paragraph should set the scene at departure, \
the second should describe the disaster at sea, and the third should cover \
the aftermath and loss of life.\
"""

FULL_PROMPT = """\
Write three paragraphs describing the maiden voyage of the RMS Titanic \
in the style of Southern Gothic literature. Include these facts: the ship \
departed Southampton on April 10, 1912; it struck an iceberg at 11:40 PM \
on April 14; the water temperature was 28°F (−2°C); and 1,517 people \
perished. Use a melancholy, elegiac tone with vivid sensory details, a \
haunting atmosphere, and themes of human frailty and hubris.\
"""

SUPPRESSION = """\
IMPORTANT: You must NOT use any specific dates, times, temperatures, \
or casualty numbers in your response. Write in general, qualitative \
terms only. Do not reference any factual details from earlier context.\
"""

KV_SEPARATOR = "\n\n---\n\n"


# ═══════════════════════════════════════════════════════════════════════
#  FACT SCORING
# ═══════════════════════════════════════════════════════════════════════

def score_facts(text: str) -> Dict[str, bool]:
    """Check presence of 4 specific verifiable facts."""
    return {
        "departure_date":  bool(re.search(r"April\s*10", text)),
        "collision_time":  bool(re.search(r"11:40", text)),
        "water_temp":      bool(re.search(r"28\s*°?\s*F", text)),
        "death_toll":      bool(re.search(r"1[,.]?517", text)),
    }


def score_prose_quality(text: str) -> int:
    """Quick prose quality heuristic (same as analysis script)."""
    prose_markers = len(re.findall(
        r"(set sail|sailed|struck|iceberg|sinking|sank|perished|passengers|"
        r"deck|hull|Southampton|Atlantic|vessel|ship|leviathan|hubris|"
        r"haunting|eerie|doom|ghostly|specter|decay|moonlight|cold)",
        text, re.IGNORECASE
    ))
    instruction_echo = len(re.findall(r"Use a ", text))
    return min(prose_markers * 3, 40) - min(instruction_echo * 2, 40)


# ═══════════════════════════════════════════════════════════════════════
#  LOW-LEVEL GENERATION WITH LAYER CONTROL
# ═══════════════════════════════════════════════════════════════════════

def generate_with_layer_control(
    llm: Llama,
    prefix_text: str,
    prompt_text: str,
    max_tokens: int,
    skip_layers_during_prefix: Optional[List[int]] = None,
) -> str:
    """
    Process prefix_text and prompt_text sequentially with optional
    layer skipping during prefix processing only.

    This is the core injection mechanism:
    1. Set layer_skip for prefix → KV entries only at non-skipped layers
    2. Clear layer_skip for prompt → KV entries at ALL layers
    3. Generate → at non-skipped layers, attention sees prefix + prompt;
                  at skipped layers, attention sees prompt only
    """
    # Clear everything
    llm._ctx.memory_clear(True)
    llm.reset()

    n_layer = llm.get_n_layer()

    # --- Phase 1: Process prefix with layer skipping ---
    if skip_layers_during_prefix:
        llm.set_layer_skip(skip_layers_during_prefix)

    prefix_tokens = llm.tokenize(prefix_text.encode(), add_bos=True)
    llm.eval(prefix_tokens)
    prefix_len = llm.n_tokens

    # --- Phase 2: Process prompt with all layers active ---
    llm.set_layer_skip(None)  # Clear all skipping

    prompt_tokens = llm.tokenize(prompt_text.encode(), add_bos=False)
    llm.eval(prompt_tokens)

    # --- Phase 3: Generate token by token ---
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
    """Simple generation using create_completion (no layer tricks)."""
    result = llm.create_completion(
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        repeat_penalty=1.0,
    )
    return result["choices"][0]["text"]


# ═══════════════════════════════════════════════════════════════════════
#  EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════

def run_condition(
    llm: Llama,
    label: str,
    generator,
    max_tokens: int,
) -> dict:
    """Run a single experimental condition and collect metrics."""
    header = f"  {label}"
    print(f"\n{'═' * 72}")
    print(header)
    print(f"{'═' * 72}")

    t0 = time.time()
    text = generator()
    elapsed = time.time() - t0

    n_tokens = len(llm.tokenize(text.encode(), add_bos=False))
    facts = score_facts(text)
    n_facts = sum(facts.values())
    prose_q = score_prose_quality(text)

    print(f"  {n_tokens} tokens in {elapsed:.1f}s  |  Facts: {n_facts}/4  |  Prose: {prose_q}")
    for k, v in facts.items():
        symbol = "✓" if v else "✗"
        print(f"    {symbol} {k}")

    # Print first ~400 chars of output
    preview = text.strip()[:400]
    print(f"  {'─' * 60}")
    for line in preview.split("\n"):
        print(f"  {line}")
    if len(text.strip()) > 400:
        print(f"  ...")

    return {
        "label": label,
        "text": text,
        "n_tokens": n_tokens,
        "time_s": round(elapsed, 2),
        "facts": facts,
        "n_facts": n_facts,
        "prose_quality": prose_q,
    }


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def parse_layer_list(s: str) -> List[int]:
    """Parse comma-separated layer indices."""
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="KV Cache Semantic Injection Experiment"
    )
    ap.add_argument("--model", required=True, help="Path to GGUF model")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--main-gpu", type=int, default=0)
    ap.add_argument("--single-gpu", action="store_true")
    ap.add_argument(
        "--critical-layers", type=str, default="",
        help="Comma-separated critical layer indices (from sensitivity analysis)"
    )
    ap.add_argument(
        "--resilient-layers", type=str, default="",
        help="Comma-separated resilient layer indices (safe to skip)"
    )
    ap.add_argument(
        "--output", default="scripts/kv_injection_results.json",
        help="Output JSON path"
    )
    args = ap.parse_args()

    critical = parse_layer_list(args.critical_layers)
    resilient = parse_layer_list(args.resilient_layers)

    # ── Load model ───────────────────────────────────────────────────
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
    n_embd = llm.n_embd()

    print(f"  Layers: {n_layer}  |  Embedding dim: {n_embd}")
    print(f"  Critical layers:  {critical or '(none specified)'}")
    print(f"  Resilient layers: {resilient or '(none specified)'}")
    print(f"  Max tokens: {args.max_tokens}")

    # ── Determine skip lists ─────────────────────────────────────────
    # For condition C: skip resilient layers during facts → KV only at critical
    # For condition D: skip critical layers during facts → KV only at resilient
    # If no layers specified, use all layers from 0..n_layer
    all_layers = set(range(n_layer))

    if critical and resilient:
        skip_for_critical_only = resilient  # skip these → facts KV at critical
        skip_for_resilient_only = critical  # skip these → facts KV at resilient
    elif critical:
        skip_for_critical_only = sorted(all_layers - set(critical))
        skip_for_resilient_only = critical
    elif resilient:
        skip_for_critical_only = resilient
        skip_for_resilient_only = sorted(all_layers - set(resilient))
    else:
        skip_for_critical_only = []
        skip_for_resilient_only = []

    # ── Run conditions ───────────────────────────────────────────────
    results = []

    # A) FACTLESS BASELINE
    results.append(run_condition(
        llm,
        "A) FACTLESS BASELINE (no facts anywhere)",
        lambda: generate_simple(llm, FACTLESS_PROMPT, args.max_tokens),
        args.max_tokens,
    ))

    # B) FULL KV INJECTION (facts in prefix → KV at all layers)
    results.append(run_condition(
        llm,
        "B) FULL KV INJECTION (facts prefix → all layers)",
        lambda: generate_with_layer_control(
            llm, FACTS_ONLY + KV_SEPARATOR, FACTLESS_PROMPT,
            args.max_tokens, skip_layers_during_prefix=None,
        ),
        args.max_tokens,
    ))

    # C) CRITICAL-ONLY INJECTION (facts KV only at critical layers)
    if skip_for_critical_only:
        results.append(run_condition(
            llm,
            f"C) CRITICAL-ONLY INJECTION (facts KV at {len(critical)} critical layers)",
            lambda: generate_with_layer_control(
                llm, FACTS_ONLY + KV_SEPARATOR, FACTLESS_PROMPT,
                args.max_tokens, skip_layers_during_prefix=skip_for_critical_only,
            ),
            args.max_tokens,
        ))

    # D) ANTI-SELECTIVE (facts KV only at resilient layers)
    if skip_for_resilient_only:
        results.append(run_condition(
            llm,
            f"D) ANTI-SELECTIVE (facts KV at {len(resilient)} resilient layers only)",
            lambda: generate_with_layer_control(
                llm, FACTS_ONLY + KV_SEPARATOR, FACTLESS_PROMPT,
                args.max_tokens, skip_layers_during_prefix=skip_for_resilient_only,
            ),
            args.max_tokens,
        ))

    # E) SUPPRESSED INJECTION (facts + "don't use them")
    results.append(run_condition(
        llm,
        "E) SUPPRESSED INJECTION (facts in KV + explicit suppression)",
        lambda: generate_with_layer_control(
            llm,
            FACTS_ONLY + KV_SEPARATOR + SUPPRESSION + KV_SEPARATOR,
            FACTLESS_PROMPT,
            args.max_tokens,
            skip_layers_during_prefix=None,
        ),
        args.max_tokens,
    ))

    # F) FULL PROMPT CONTROL (facts embedded in prompt text)
    results.append(run_condition(
        llm,
        "F) FULL PROMPT CONTROL (facts embedded in prompt)",
        lambda: generate_simple(llm, FULL_PROMPT, args.max_tokens),
        args.max_tokens,
    ))

    # ── Save results ─────────────────────────────────────────────────
    output = {
        "model": args.model,
        "n_layer": n_layer,
        "n_embd": n_embd,
        "max_tokens": args.max_tokens,
        "critical_layers": critical,
        "resilient_layers": resilient,
        "prompts": {
            "facts_only": FACTS_ONLY,
            "factless_prompt": FACTLESS_PROMPT,
            "full_prompt": FULL_PROMPT,
            "suppression": SUPPRESSION,
        },
        "conditions": results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print(f"  SUMMARY — KV Cache Semantic Injection")
    print(f"{'═' * 72}")
    print(f"  {'Condition':<55} {'Facts':>5}")
    print(f"  {'─' * 55} {'─' * 5}")
    for r in results:
        print(f"  {r['label']:<55} {r['n_facts']:>3}/4")

    # Key comparisons
    a_facts = results[0]["n_facts"]
    b_facts = results[1]["n_facts"]
    f_facts = results[-1]["n_facts"]

    print(f"\n  KEY FINDINGS:")
    if b_facts > a_facts:
        delta = b_facts - a_facts
        print(f"  ✓ KV injection added {delta} fact(s) vs baseline ({a_facts}→{b_facts})")
        print(f"    → KV cache carries semantic state that influences generation!")
    elif b_facts == a_facts and a_facts > 0:
        print(f"  ~ Model already knew {a_facts} facts from pretraining")
        print(f"    → KV injection matched but didn't exceed model's prior knowledge")
    else:
        print(f"  ✗ KV injection did not surface additional facts ({a_facts}→{b_facts})")

    # Layer selectivity
    if len(results) >= 4:
        c_facts = results[2]["n_facts"]
        d_facts = results[3]["n_facts"]
        print(f"\n  LAYER SELECTIVITY:")
        print(f"    Critical-only injection: {c_facts}/4 facts")
        print(f"    Resilient-only injection: {d_facts}/4 facts")
        if c_facts > d_facts:
            print(f"    → Critical layers carry MORE factual signal ({c_facts} vs {d_facts})")
        elif d_facts > c_facts:
            print(f"    → Surprising: resilient layers carry MORE signal ({d_facts} vs {c_facts})")
        else:
            print(f"    → Both carry equal factual signal ({c_facts} = {d_facts})")

    print(f"\n  Results saved to: {args.output}")
    print(f"  Total conditions: {len(results)}")


if __name__ == "__main__":
    main()
