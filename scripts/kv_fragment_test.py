#!/usr/bin/env python3
"""
KV Fragment Injection — Atomic Knowledge Fragments Test

Tests whether small, individual fact fragments injected as KV cache prefixes
produce targeted recall. This validates the hybrid RAG + KV cache architecture.

The idea:
- Break the full 117-token prefix into atomic fact fragments (~5-15 tokens each)
- Pre-compute KV cache for each fragment
- Test individual and combined fragment injection
- Measure whether each fragment's specific fact appears in generation

Usage:
    python scripts/kv_fragment_test.py --model <path.gguf> [options]
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

from llama_cpp import Llama


# ═══════════════════════════════════════════════════════════════════════
#  KNOWLEDGE FRAGMENTS — atomic facts with their expected recall checks
# ═══════════════════════════════════════════════════════════════════════

FRAGMENTS = {
    "age": {
        "text": "Elara Voss is 34 years old.\n",
        "checks": {"age_34": r"\b34\b"},
    },
    "hair": {
        "text": "Elara has silver-streaked auburn hair.\n",
        "checks": {"silver_hair": r"silver", "auburn_hair": r"auburn"},
    },
    "scar": {
        "text": "Elara has a crescent-shaped scar on her left palm.\n",
        "checks": {"crescent_scar": r"crescent"},
    },
    "forest": {
        "text": "The forest is called Thornwood in the Greywander Basin.\n",
        "checks": {"thornwood": r"[Tt]hornwood", "greywander": r"[Gg]reywander"},
    },
    "season": {
        "text": "It is late November and the first frost has come.\n",
        "checks": {"november": r"[Nn]ovember", "frost": r"frost"},
    },
    "dog": {
        "text": "Her companion is a wolfhound named Cassius with one blue eye and one amber eye.\n",
        "checks": {"wolfhound": r"wolfhound", "cassius": r"[Cc]assius",
                   "heterochromia": r"blue.{1,30}amber|amber.{1,30}blue"},
    },
    "compass": {
        "text": "She carries a brass compass that belonged to her grandmother Mirabel.\n",
        "checks": {"brass_compass": r"brass.*compass|compass.*brass", "mirabel": r"[Mm]irabel"},
    },
}

ALL_CHECKS = {
    "age_34": r"\b34\b",
    "silver_hair": r"silver",
    "auburn_hair": r"auburn",
    "crescent_scar": r"crescent",
    "thornwood": r"[Tt]hornwood",
    "greywander": r"[Gg]reywander",
    "november": r"[Nn]ovember",
    "frost": r"frost",
    "wolfhound": r"wolfhound",
    "cassius": r"[Cc]assius",
    "heterochromia": r"blue.{1,30}amber|amber.{1,30}blue",
    "brass_compass": r"brass.*compass|compass.*brass",
    "mirabel": r"[Mm]irabel",
}

FULL_PREFIX = """Character and setting details for the story:
- The woman's name is Elara Voss. She is 34 years old.
- She has silver-streaked auburn hair and a crescent-shaped scar on her left palm.
- The forest is called Thornwood. It sits in a valley called the Greywander Basin.
- It is late November, the first frost has already come.
- Her companion is a wolfhound named Cassius with one blue eye and one amber eye.
- She carries a brass compass that belonged to her grandmother, Mirabel.
---
"""

PROMPT = (
    "<|im_start|>user\n"
    "Write a short story about a woman named Elara walking through "
    "a forest at dusk. Three paragraphs. Rich sensory details. /no_think"
    "<|im_end|>\n<|im_start|>assistant\n"
)


# ═══════════════════════════════════════════════════════════════════════
#  GENERATION
# ═══════════════════════════════════════════════════════════════════════

def generate_with_prefix(
    llm: Llama,
    prefix_text: str,
    prompt_text: str,
    max_tokens: int = 512,
) -> str:
    """Generate with a prefix injected before the prompt."""
    llm._ctx.memory_clear(True)
    llm.reset()

    if prefix_text:
        prefix_tokens = llm.tokenize(prefix_text.encode(), add_bos=True)
        prompt_tokens = llm.tokenize(prompt_text.encode(), add_bos=False)
        llm.eval(prefix_tokens)
        llm.eval(prompt_tokens)
        n_prefix = len(prefix_tokens)
    else:
        tokens = llm.tokenize(prompt_text.encode(), add_bos=True)
        llm.eval(tokens)
        n_prefix = 0

    output = []
    eos = llm.token_eos()
    for _ in range(max_tokens):
        lp = llm._ctx.get_logits()
        logits = np.ctypeslib.as_array(lp, shape=(llm._n_vocab,)).copy()
        tok = int(np.argmax(logits))
        if tok == eos:
            break
        output.append(tok)
        llm.eval([tok])

    return llm.detokenize(output).decode("utf-8", errors="replace"), n_prefix


def check_recall(text: str, checks: Dict[str, str] = None) -> Dict[str, bool]:
    """Check which facts appear in generated text."""
    if checks is None:
        checks = ALL_CHECKS
    return {k: bool(re.search(p, text, re.I)) for k, p in checks.items()}


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="KV fragment injection test")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--main-gpu", type=int, default=0)
    parser.add_argument("--single-gpu", action="store_true")
    parser.add_argument("--n-ctx", type=int, default=4096)
    args = parser.parse_args()

    extra = {}
    if args.single_gpu:
        extra["tensor_split"] = [1.0, 0.0]

    print("Loading model...")
    llm = Llama(
        model_path=args.model,
        n_gpu_layers=-1,
        n_ctx=args.n_ctx,
        main_gpu=args.main_gpu,
        verbose=False,
        embeddings=False,
        **extra,
    )
    print(f"  n_embd={llm.n_embd()}, n_layer={llm.get_n_layer()}")

    # Show fragment sizes
    print(f"\n{'═' * 72}")
    print(f"  FRAGMENT INVENTORY")
    print(f"{'═' * 72}")
    total_frag_tokens = 0
    for name, frag in FRAGMENTS.items():
        toks = llm.tokenize(frag["text"].encode(), add_bos=False)
        total_frag_tokens += len(toks)
        checks_str = ", ".join(frag["checks"].keys())
        print(f"  {name:>10}: {len(toks):>3} tokens | checks: {checks_str}")

    full_tokens = llm.tokenize(FULL_PREFIX.encode(), add_bos=True)
    print(f"  {'':>10}  ───")
    print(f"  {'fragments':>10}: {total_frag_tokens:>3} tokens total")
    print(f"  {'full_prefix':>10}: {len(full_tokens):>3} tokens")
    print(f"  {'savings':>10}: {len(full_tokens) - total_frag_tokens:>3} tokens ({100*(len(full_tokens)-total_frag_tokens)/len(full_tokens):.0f}%)")

    results = {}

    # ─── Baseline: no prefix ────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print(f"  TEST: Baseline (no prefix)")
    print(f"{'═' * 72}")
    text, n_pre = generate_with_prefix(llm, "", PROMPT, args.max_tokens)
    hits = check_recall(text)
    total = sum(hits.values())
    print(f"  Recall: {total}/13  (0 prefix tokens)")
    results["bare"] = {"recall": total, "prefix_tokens": 0, "hits": hits}

    # ─── Full prefix ────────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print(f"  TEST: Full prefix")
    print(f"{'═' * 72}")
    text, n_pre = generate_with_prefix(llm, FULL_PREFIX, PROMPT, args.max_tokens)
    hits = check_recall(text)
    total = sum(hits.values())
    print(f"  Recall: {total}/13  ({n_pre} prefix tokens)")
    for k, v in hits.items():
        if v:
            print(f"    ✓ {k}")
    results["full_prefix"] = {"recall": total, "prefix_tokens": n_pre, "hits": hits}

    # ─── Individual fragments ───────────────────────────────────────
    print(f"\n{'═' * 72}")
    print(f"  TEST: Individual fragments (one at a time)")
    print(f"{'═' * 72}")
    for name, frag in FRAGMENTS.items():
        text, n_pre = generate_with_prefix(llm, frag["text"], PROMPT, args.max_tokens)
        # Check both the fragment's own checks AND all checks
        own_hits = check_recall(text, frag["checks"])
        all_hits = check_recall(text)
        own_total = sum(own_hits.values())
        all_total = sum(all_hits.values())
        own_max = len(frag["checks"])

        marks = " ".join(f"✓{k}" if v else f"-{k}" for k, v in own_hits.items())
        bonus = [k for k, v in all_hits.items() if v and k not in frag["checks"]]
        bonus_str = f"  +bonus: {', '.join(bonus)}" if bonus else ""

        print(f"  {name:>10}: {own_total}/{own_max} own, {all_total}/13 total  "
              f"({n_pre} tok)  [{marks}]{bonus_str}")

        results[f"frag_{name}"] = {
            "recall": all_total, "own_recall": own_total,
            "own_max": own_max, "prefix_tokens": n_pre,
            "hits": all_hits, "own_hits": own_hits,
        }

    # ─── All fragments concatenated ────────────────────────────────
    print(f"\n{'═' * 72}")
    print(f"  TEST: All fragments concatenated")
    print(f"{'═' * 72}")
    all_frags = "".join(frag["text"] for frag in FRAGMENTS.values())
    text, n_pre = generate_with_prefix(llm, all_frags, PROMPT, args.max_tokens)
    hits = check_recall(text)
    total = sum(hits.values())
    print(f"  Recall: {total}/13  ({n_pre} prefix tokens)")
    for k, v in hits.items():
        if v:
            print(f"    ✓ {k}")
    results["all_fragments"] = {"recall": total, "prefix_tokens": n_pre, "hits": hits}

    # ─── Selective combos (simulate keyword matching) ───────────────
    print(f"\n{'═' * 72}")
    print(f"  TEST: Selective fragment combos (simulated keyword retrieval)")
    print(f"{'═' * 72}")

    combos = {
        "hair+forest": ["hair", "forest"],
        "dog+compass": ["dog", "compass"],
        "hair+dog+season": ["hair", "dog", "season"],
        "forest+season+scar": ["forest", "season", "scar"],
        "hair+dog+forest+compass": ["hair", "dog", "forest", "compass"],
    }

    for combo_name, frag_names in combos.items():
        combo_text = "".join(FRAGMENTS[f]["text"] for f in frag_names)
        text, n_pre = generate_with_prefix(llm, combo_text, PROMPT, args.max_tokens)
        hits = check_recall(text)
        total = sum(hits.values())
        # Count which of the combo's own checks hit
        combo_checks = {}
        for f in frag_names:
            combo_checks.update(FRAGMENTS[f]["checks"])
        own_hits = {k: hits.get(k, False) for k in combo_checks}
        own_total = sum(own_hits.values())
        own_max = len(combo_checks)

        print(f"  {combo_name:>25}: {own_total}/{own_max} targeted, "
              f"{total}/13 total  ({n_pre} tok)")
        results[f"combo_{combo_name}"] = {
            "recall": total, "own_recall": own_total,
            "own_max": own_max, "prefix_tokens": n_pre, "hits": hits,
        }

    # ─── Summary ────────────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print(f"  SUMMARY")
    print(f"{'═' * 72}")
    print(f"  {'Condition':>30}  {'Tokens':>6}  {'Recall':>8}  {'Efficiency':>10}")
    print(f"  {'─' * 62}")
    for name, r in sorted(results.items(), key=lambda x: -x[1]["recall"]):
        tok = r["prefix_tokens"]
        rec = r["recall"]
        eff = f"{rec/max(tok,1):.3f}/tok" if tok > 0 else "N/A"
        print(f"  {name:>30}  {tok:>6}  {rec:>5}/13  {eff:>10}")

    del llm


if __name__ == "__main__":
    main()
