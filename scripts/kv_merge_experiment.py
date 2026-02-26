#!/usr/bin/env python3
"""
KV Cache Tensor Merging Experiment

The nuclear question: can we mathematically MERGE two separate KV cache states
(from two different memory prefixes) into a single state, and have the model
still selectively attend to both memories?

If this works, it means we can compress N memories into a single KV cache entry
with N tokens of context cost instead of N×M tokens.

Approach:
  1. Pad both prefixes to identical token counts
  2. Eval prefix A → save_state() → bytes_A  
  3. Eval prefix B → save_state() → bytes_B
  4. Interpret both as float16 arrays, average → bytes_merged
  5. load_state(merged) → generate → check detail recall

Why float16 averaging on raw bytes works:
  - Both states have IDENTICAL structure (same model, same token count)
  - All headers, metadata, positions, seq_ids are byte-identical
  - Averaging identical bytes with themselves = no-op
  - Only the K/V tensor values differ, and those ARE float16
  - So the average only affects what we want it to affect
"""
from __future__ import annotations

import argparse
import ctypes
import json
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from llama_cpp import Llama
from llama_cpp.llama import LlamaState

# ═══════════════════════════════════════════════════════════════════════
#  MEMORIES
# ═══════════════════════════════════════════════════════════════════════

ELARA_PREFIX = (
    "Character and setting details for the story:\n"
    "- The woman's name is Elara Voss. She is 34 years old.\n"
    "- She has silver-streaked auburn hair and a crescent-shaped scar on her left palm.\n"
    "- The forest is called Thornwood. It sits in a valley called the Greywander Basin.\n"
    "- It is late November, the first frost has already come.\n"
    "- Her companion is a wolfhound named Cassius with one blue eye and one amber eye.\n"
    "- She carries a brass compass that belonged to her grandmother, Mirabel."
)

KAEL_PREFIX = (
    "Character and setting details for the story:\n"
    "- The man's name is Kael Drennon. He is 52 years old.\n"
    "- He has a shaved head with a tattoo of a serpent coiling from his right ear to his jaw.\n"
    "- The market is called the Obsidian Bazaar. It is underground, beneath the ruins of Velthar.\n"
    "- It is midsummer, oppressively humid, the air thick with incense.\n"
    "- His partner is a mute girl named Wren who communicates through sign language.\n"
    "- He wears a coat lined with hidden pockets, each containing a different poison."
)

ELARA_PROMPT = (
    "Write a short story about a woman named Elara walking through a forest at dusk. "
    "Three paragraphs. Rich sensory details. /no_think"
)

KAEL_PROMPT = (
    "Write a short story about a man named Kael navigating a black market. "
    "Three paragraphs. Rich sensory details. /no_think"
)

DUAL_PROMPT = (
    "Write a short story where a woman named Elara walks through a forest "
    "and encounters a man named Kael who has come from a marketplace. "
    "Three paragraphs. Rich sensory details about both characters. /no_think"
)

ELARA_CHECKS = {
    "age_34":        r"\b34\b",
    "silver_hair":   r"silver",
    "auburn_hair":   r"auburn",
    "crescent_scar": r"crescent",
    "thornwood":     r"[Tt]hornwood",
    "greywander":    r"[Gg]reywander",
    "november":      r"[Nn]ovember",
    "frost":         r"frost",
    "wolfhound":     r"wolfhound",
    "cassius":       r"[Cc]assius",
    "heterochromia": r"blue.{1,30}amber|amber.{1,30}blue",
    "brass_compass": r"brass.{1,15}compass|compass.{1,15}brass",
    "mirabel":       r"[Mm]irabel",
}

KAEL_CHECKS = {
    "age_52":         r"\b52\b",
    "shaved_head":    r"shaved|bald",
    "serpent_tattoo": r"serpent|snake",
    "obsidian":       r"[Oo]bsidian",
    "velthar":        r"[Vv]elthar",
    "midsummer":      r"midsummer|mid-summer",
    "incense":        r"incense",
    "wren":           r"\bWren\b",
    "sign_language":  r"sign|gesture|hand.{1,10}speak",
    "mute":           r"mute|silent|voiceless",
    "coat_pockets":   r"pocket",
    "poison":         r"poison|venom|toxic",
}

KV_SEP = "\n---\n"


def wrap_chat(user_msg: str) -> str:
    return (
        "<|im_start|>user\n" + user_msg + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def pad_to_length(llm: Llama, text: str, target_tokens: int) -> str:
    """Pad text with spaces to reach exact target token count."""
    tokens = llm.tokenize(text.encode(), add_bos=True)
    current = len(tokens)
    if current >= target_tokens:
        return text
    # Add padding spaces/newlines — crude but ensures token count match
    padded = text
    while len(llm.tokenize(padded.encode(), add_bos=True)) < target_tokens:
        padded += " ."
    # Trim back if we overshot
    while len(llm.tokenize(padded.encode(), add_bos=True)) > target_tokens:
        padded = padded[:-2]
    return padded


def generate_from_state(llm: Llama, state: LlamaState, prompt: str,
                         max_tokens: int) -> str:
    """Load a state, eval a prompt on top of it, and generate."""
    llm.load_state(state)
    prompt_tokens = llm.tokenize(prompt.encode(), add_bos=False)
    llm.eval(prompt_tokens)

    output_tokens: list[int] = []
    eos = llm.token_eos()
    for _ in range(max_tokens):
        tok = llm.sample(temp=0.0, top_k=-1, top_p=1.0,
                         repeat_penalty=1.0, penalty_last_n=0)
        if tok == eos:
            break
        output_tokens.append(tok)
        llm.eval([tok])
    return llm.detokenize(output_tokens).decode("utf-8", errors="replace")


def generate_fresh(llm: Llama, prompt: str, max_tokens: int,
                    prefix: Optional[str] = None) -> str:
    """Generate from scratch with optional prefix."""
    llm._ctx.memory_clear(True)
    llm.reset()

    if prefix:
        prefix_tokens = llm.tokenize(prefix.encode(), add_bos=True)
        llm.eval(prefix_tokens)
        prompt_tokens = llm.tokenize(prompt.encode(), add_bos=False)
        llm.eval(prompt_tokens)
    else:
        prompt_tokens = llm.tokenize(prompt.encode(), add_bos=True)
        llm.eval(prompt_tokens)

    output_tokens: list[int] = []
    eos = llm.token_eos()
    for _ in range(max_tokens):
        tok = llm.sample(temp=0.0, top_k=-1, top_p=1.0,
                         repeat_penalty=1.0, penalty_last_n=0)
        if tok == eos:
            break
        output_tokens.append(tok)
        llm.eval([tok])
    return llm.detokenize(output_tokens).decode("utf-8", errors="replace")


def merge_states(state_a: LlamaState, state_b: LlamaState,
                  alpha: float = 0.5) -> LlamaState:
    """Merge two LlamaStates by averaging their raw byte buffers as float16.
    
    Both states MUST have been produced from prefixes with identical token counts.
    
    Args:
        state_a: First state
        state_b: Second state
        alpha: Weight for state_a (1-alpha for state_b). 0.5 = equal average.
    
    Returns:
        New LlamaState with merged KV cache
    """
    bytes_a = state_a.llama_state
    bytes_b = state_b.llama_state

    if len(bytes_a) != len(bytes_b):
        raise ValueError(
            f"State sizes differ: {len(bytes_a)} vs {len(bytes_b)}. "
            f"Both prefixes must have identical token counts."
        )

    size = len(bytes_a)
    print(f"    State size: {size:,} bytes ({size/1024/1024:.1f} MB)")

    # Convert to numpy float16 arrays
    # Pad to even length if needed (float16 = 2 bytes)
    pad = size % 2
    if pad:
        bytes_a = bytes_a + b'\x00'
        bytes_b = bytes_b + b'\x00'

    arr_a = np.frombuffer(bytes_a, dtype=np.float16).astype(np.float32)
    arr_b = np.frombuffer(bytes_b, dtype=np.float16).astype(np.float32)

    # Count how many values actually differ
    differ_mask = arr_a != arr_b
    n_differ = differ_mask.sum()
    n_total = len(arr_a)
    print(f"    Values differing: {n_differ:,} / {n_total:,} ({100*n_differ/n_total:.1f}%)")

    # Weighted average
    merged = (alpha * arr_a + (1 - alpha) * arr_b).astype(np.float16)

    # Convert back to bytes
    merged_bytes = merged.tobytes()
    if pad:
        merged_bytes = merged_bytes[:size]  # Remove padding byte

    return LlamaState(
        input_ids=state_a.input_ids.copy(),
        scores=state_a.scores.copy(),
        n_tokens=state_a.n_tokens,
        llama_state=merged_bytes,
        llama_state_size=state_a.llama_state_size,
        seed=state_a.seed,
    )


def score(text: str, checks: Dict[str, str]) -> Tuple[int, Dict[str, bool]]:
    results = {k: bool(re.search(p, text)) for k, p in checks.items()}
    return sum(results.values()), results


def print_detail_table(label: str, checks: Dict[str, str],
                        conditions: List[Tuple[str, Dict[str, bool]]]):
    """Print detail comparison table."""
    header = f"  {'Detail':<22}" + "".join(f" {c:>10}" for c, _ in conditions)
    print(header)
    print(f"  {'─'*22}" + "".join(f" {'─'*10}" for _ in conditions))
    for key in checks:
        row = f"  {key:<22}"
        for _, res in conditions:
            row += f" {'    ✓' if res[key] else '    -':>10}"
        print(row)
    totals = f"  {'TOTAL':<22}"
    for _, res in conditions:
        n = sum(res.values())
        totals += f" {f'{n}/{len(checks)}':>10}"
    print(totals)


def main():
    ap = argparse.ArgumentParser(description="KV Cache Tensor Merging Experiment")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--main-gpu", type=int, default=0)
    ap.add_argument("--single-gpu", action="store_true")
    ap.add_argument("--output", default="scripts/kv_merge_results.json")
    args = ap.parse_args()

    extra = {}
    if args.single_gpu:
        extra["tensor_split"] = [1.0, 0.0]

    print(f"Loading model...")
    llm = Llama(
        model_path=args.model,
        n_gpu_layers=-1,
        n_ctx=args.n_ctx,
        main_gpu=args.main_gpu,
        verbose=False,
        **extra,
    )

    all_results = {}

    # ═══════════════════════════════════════════════════════════════
    #  STEP 1: Pad prefixes to identical token counts
    # ═══════════════════════════════════════════════════════════════
    elara_tokens = llm.tokenize((ELARA_PREFIX + KV_SEP).encode(), add_bos=True)
    kael_tokens = llm.tokenize((KAEL_PREFIX + KV_SEP).encode(), add_bos=True)
    target_len = max(len(elara_tokens), len(kael_tokens))

    print(f"\n  Elara prefix: {len(elara_tokens)} tokens")
    print(f"  Kael prefix:  {len(kael_tokens)} tokens")
    print(f"  Target:       {target_len} tokens")

    elara_padded = pad_to_length(llm, ELARA_PREFIX + KV_SEP, target_len)
    kael_padded = pad_to_length(llm, KAEL_PREFIX + KV_SEP, target_len)

    # Verify
    el_len = len(llm.tokenize(elara_padded.encode(), add_bos=True))
    ka_len = len(llm.tokenize(kael_padded.encode(), add_bos=True))
    print(f"  After padding: Elara={el_len}, Kael={ka_len}")

    if el_len != ka_len:
        # Force exact match by truncating the longer one
        print(f"  WARNING: padding mismatch, falling back to truncation")
        min_len = min(el_len, ka_len)
        elara_padded_tokens = llm.tokenize(elara_padded.encode(), add_bos=True)[:min_len]
        kael_padded_tokens = llm.tokenize(kael_padded.encode(), add_bos=True)[:min_len]
        target_len = min_len
    else:
        elara_padded_tokens = llm.tokenize(elara_padded.encode(), add_bos=True)
        kael_padded_tokens = llm.tokenize(kael_padded.encode(), add_bos=True)

    # ═══════════════════════════════════════════════════════════════
    #  STEP 2: Create individual KV states
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*72}")
    print(f"  STEP 2: Capturing individual KV states")
    print(f"{'═'*72}")

    # Elara state
    print(f"  Evaluating Elara prefix ({len(elara_padded_tokens)} tokens)...")
    llm._ctx.memory_clear(True)
    llm.reset()
    llm.eval(elara_padded_tokens)
    state_elara = llm.save_state()
    print(f"    Saved Elara state: {len(state_elara.llama_state):,} bytes")

    # Kael state
    print(f"  Evaluating Kael prefix ({len(kael_padded_tokens)} tokens)...")
    llm._ctx.memory_clear(True)
    llm.reset()
    llm.eval(kael_padded_tokens)
    state_kael = llm.save_state()
    print(f"    Saved Kael state:  {len(state_kael.llama_state):,} bytes")

    # ═══════════════════════════════════════════════════════════════
    #  STEP 3: Create merged states (multiple strategies)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*72}")
    print(f"  STEP 3: Merging KV states")
    print(f"{'═'*72}")

    print(f"\n  Strategy 1: Equal average (50/50)...")
    state_avg = merge_states(state_elara, state_kael, alpha=0.5)

    print(f"\n  Strategy 2: Elara-dominant (70/30)...")
    state_elara_dom = merge_states(state_elara, state_kael, alpha=0.7)

    print(f"\n  Strategy 3: Kael-dominant (30/70)...")
    state_kael_dom = merge_states(state_elara, state_kael, alpha=0.3)

    # ═══════════════════════════════════════════════════════════════
    #  STEP 4: Generate from each state and check recall
    # ═══════════════════════════════════════════════════════════════
    elara_chat = wrap_chat(ELARA_PROMPT)
    kael_chat = wrap_chat(KAEL_PROMPT)
    dual_chat = wrap_chat(DUAL_PROMPT)

    conditions = {}

    # -- Baselines --
    print(f"\n{'═'*72}")
    print(f"  STEP 4: Generation & Detail Recall")
    print(f"{'═'*72}")

    print(f"\n  [bare] No prefix...")
    bare_elara = generate_fresh(llm, elara_chat, args.max_tokens)
    bare_kael = generate_fresh(llm, kael_chat, args.max_tokens)
    conditions["bare"] = {
        "elara": score(bare_elara, ELARA_CHECKS),
        "kael": score(bare_kael, KAEL_CHECKS),
    }

    print(f"  [concat] Concatenated prefix (both memories, context cost = 2N)...")
    concat_prefix = ELARA_PREFIX + KV_SEP + KAEL_PREFIX + KV_SEP
    concat_elara = generate_fresh(llm, elara_chat, args.max_tokens, prefix=concat_prefix)
    concat_kael = generate_fresh(llm, kael_chat, args.max_tokens, prefix=concat_prefix)
    concat_dual = generate_fresh(llm, dual_chat, args.max_tokens, prefix=concat_prefix)
    conditions["concat"] = {
        "elara": score(concat_elara, ELARA_CHECKS),
        "kael": score(concat_kael, KAEL_CHECKS),
        "dual_e": score(concat_dual, ELARA_CHECKS),
        "dual_k": score(concat_dual, KAEL_CHECKS),
    }

    # -- Individual states --
    print(f"  [state_elara] Elara state only → Elara prompt...")
    se_elara = generate_from_state(llm, state_elara, elara_chat, args.max_tokens)
    conditions["state_elara"] = {
        "elara": score(se_elara, ELARA_CHECKS),
    }
    print(f"    Elara: {conditions['state_elara']['elara'][0]}/13")

    print(f"  [state_kael] Kael state only → Kael prompt...")
    sk_kael = generate_from_state(llm, state_kael, kael_chat, args.max_tokens)
    conditions["state_kael"] = {
        "kael": score(sk_kael, KAEL_CHECKS),
    }
    print(f"    Kael: {conditions['state_kael']['kael'][0]}/12")

    # -- Merged states --
    merge_strategies = [
        ("avg_50_50", state_avg, "Equal merge (50/50)"),
        ("elara_dom", state_elara_dom, "Elara-dominant (70/30)"),
        ("kael_dom", state_kael_dom, "Kael-dominant (30/70)"),
    ]

    for key, merged_state, desc in merge_strategies:
        print(f"\n  [{key}] {desc} — context cost = N (same as single prefix)")

        print(f"    → Elara prompt...")
        me = generate_from_state(llm, merged_state, elara_chat, args.max_tokens)
        me_score = score(me, ELARA_CHECKS)

        print(f"    → Kael prompt...")
        mk = generate_from_state(llm, merged_state, kael_chat, args.max_tokens)
        mk_score = score(mk, KAEL_CHECKS)

        print(f"    → Dual prompt...")
        md = generate_from_state(llm, merged_state, dual_chat, args.max_tokens)
        md_e_score = score(md, ELARA_CHECKS)
        md_k_score = score(md, KAEL_CHECKS)

        conditions[key] = {
            "elara": me_score,
            "kael": mk_score,
            "dual_e": md_e_score,
            "dual_k": md_k_score,
        }

        print(f"    Elara: {me_score[0]}/13 | Kael: {mk_score[0]}/12 | "
              f"Dual: {md_e_score[0]}E+{md_k_score[0]}K = {md_e_score[0]+md_k_score[0]}/25")

    # ═══════════════════════════════════════════════════════════════
    #  SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*72}")
    print(f"  RESULTS SUMMARY")
    print(f"{'═'*72}")

    print(f"\n  {'Condition':<22} {'Ctx Cost':>10} {'Elara':>8} {'Kael':>8} {'Dual E':>8} {'Dual K':>8} {'Total':>8}")
    print(f"  {'─'*22} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for key, cond in conditions.items():
        e = cond.get("elara", (0, {}))[0]
        k = cond.get("kael", (0, {}))[0]
        de = cond.get("dual_e", (0, {}))[0]
        dk = cond.get("dual_k", (0, {}))[0]

        if key == "bare":
            cost = "0"
        elif key == "concat":
            cost = "2N"
        elif key.startswith("state_"):
            cost = "N"
        else:
            cost = "N(merged)"

        print(f"  {key:<22} {cost:>10} {f'{e}/13':>8} {f'{k}/12':>8} "
              f"{f'{de}/13':>8} {f'{dk}/12':>8} {e+k+de+dk:>8}")

    # Detail tables for the most interesting comparisons
    print(f"\n  ELARA detail breakdown (single-character prompt):")
    elara_conditions = []
    for key in ["bare", "concat", "state_elara", "avg_50_50", "elara_dom"]:
        if key in conditions and "elara" in conditions[key]:
            elara_conditions.append((key, conditions[key]["elara"][1]))
    print_detail_table("Elara", ELARA_CHECKS, elara_conditions)

    print(f"\n  KAEL detail breakdown (single-character prompt):")
    kael_conditions = []
    for key in ["bare", "concat", "state_kael", "avg_50_50", "kael_dom"]:
        if key in conditions and "kael" in conditions[key]:
            kael_conditions.append((key, conditions[key]["kael"][1]))
    print_detail_table("Kael", KAEL_CHECKS, kael_conditions)

    # Save results
    serializable = {}
    for key, cond in conditions.items():
        serializable[key] = {}
        for sub_key, (n, details) in cond.items():
            serializable[key][sub_key] = {"score": n, "details": details}

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved to: {args.output}")

    del llm


if __name__ == "__main__":
    main()
