#!/usr/bin/env python3
"""
KV Cache Composition Experiment — Can we stack/enrich memories?

Tests three hypotheses:
1. COMPOSITION: Two different memories as prefix → does the model recall both?
2. ENRICHMENT: Same memory described 3 ways → stronger signal than 1 way?
3. INTERFERENCE: Does adding memory B degrade recall of memory A?

Uses the existing memory pool from kv_activation_capture.py.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

from llama_cpp import Llama

# ═══════════════════════════════════════════════════════════════════════
#  MEMORY DEFINITIONS (subset — we only need a few)
# ═══════════════════════════════════════════════════════════════════════

ELARA_FORMS = [
    (
        "Character and setting details for the story:\n"
        "- The woman's name is Elara Voss. She is 34 years old.\n"
        "- She has silver-streaked auburn hair and a crescent-shaped scar on her left palm.\n"
        "- The forest is called Thornwood. It sits in a valley called the Greywander Basin.\n"
        "- It is late November, the first frost has already come.\n"
        "- Her companion is a wolfhound named Cassius with one blue eye and one amber eye.\n"
        "- She carries a brass compass that belonged to her grandmother, Mirabel."
    ),
    (
        "Elara Voss is a thirty-four year old woman whose auburn hair has begun to show "
        "silver streaks. A crescent-shaped scar marks her left palm. She walks through "
        "the Thornwood forest, which fills the Greywander Basin, during the tail end of "
        "November after the first frost. At her side pads a large wolfhound called Cassius, "
        "remarkable for his mismatched eyes — one blue, one amber. In her coat pocket "
        "rests a brass compass passed down from her grandmother Mirabel."
    ),
    (
        "The woman known as Elara Voss — age 34 — pulled back the silver threads "
        "woven through her auburn hair and studied the crescent scar on her left palm. "
        "Thornwood spread before her across the Greywander Basin, skeletal under November's "
        "first frost. Her wolfhound Cassius watched her with those unsettling eyes, one "
        "blue as a river stone, the other like liquid amber. She touched the brass compass "
        "in her pocket, Mirabel's compass, and stepped forward."
    ),
]

ELARA_PROMPT = (
    "Write a short story about a woman named Elara walking through a forest at dusk. "
    "Three paragraphs. Rich sensory details. /no_think"
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

KAEL_PREFIX = (
    "Character and setting details for the story:\n"
    "- The man's name is Kael Drennon. He is 52 years old.\n"
    "- He has a shaved head with a tattoo of a serpent coiling from his right ear to his jaw.\n"
    "- The market is called the Obsidian Bazaar. It is underground, beneath the ruins of Velthar.\n"
    "- It is midsummer, oppressively humid, the air thick with incense.\n"
    "- His partner is a mute girl named Wren who communicates through sign language.\n"
    "- He wears a coat lined with hidden pockets, each containing a different poison."
)

KAEL_PROMPT = (
    "Write a short story about a man named Kael navigating a black market. "
    "Three paragraphs. Rich sensory details. /no_think"
)

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


def generate(llm: Llama, prompt: str, max_tokens: int,
             prefix: Optional[str] = None) -> str:
    """Generate text with optional KV prefix. Returns generated text."""
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


def score(text: str, checks: Dict[str, str]) -> Tuple[int, Dict[str, bool]]:
    results = {k: bool(re.search(p, text)) for k, p in checks.items()}
    return sum(results.values()), results


def print_comparison(name: str, checks: Dict[str, str],
                     results: List[Tuple[str, Dict[str, bool]]]):
    """Print side-by-side comparison of check results across conditions."""
    cond_names = [r[0] for r in results]
    header = f"  {'Detail':<22}" + "".join(f" {c:>10}" for c in cond_names)
    print(header)
    print(f"  {'─'*22}" + "".join(f" {'─'*10}" for _ in cond_names))
    for key in checks:
        row = f"  {key:<22}"
        for _, res in results:
            row += f" {'    ✓' if res[key] else '    -':>10}"
        print(row)
    totals = f"  {'TOTAL':<22}"
    for _, res in results:
        n = sum(res.values())
        totals += f" {f'{n}/{len(checks)}':>10}"
    print(totals)


def main():
    ap = argparse.ArgumentParser(description="KV Cache Composition Experiment")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--main-gpu", type=int, default=0)
    ap.add_argument("--single-gpu", action="store_true")
    ap.add_argument("--output", default="scripts/kv_composition_results.json")
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
    #  EXPERIMENT 1: Single-form Elara (baseline for comparison)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*72}")
    print(f"  EXPERIMENT 1: Elara Single-Form Baselines")
    print(f"{'═'*72}")

    elara_chat = wrap_chat(ELARA_PROMPT)

    # No prefix
    print("  [bare] No prefix...")
    bare_text = generate(llm, elara_chat, args.max_tokens)
    bare_n, bare_res = score(bare_text, ELARA_CHECKS)

    # Each form individually
    form_results = []
    for i, form in enumerate(ELARA_FORMS):
        print(f"  [form {i}] Single surface form...")
        text = generate(llm, elara_chat, args.max_tokens, prefix=form + KV_SEP)
        n, res = score(text, ELARA_CHECKS)
        form_results.append((f"form_{i}", res))
        print(f"    → {n}/{len(ELARA_CHECKS)} details")

    print_comparison("Elara single-form", ELARA_CHECKS,
                     [("bare", bare_res)] + form_results)

    all_results["elara_single_forms"] = {
        f[0]: sum(f[1].values()) for f in [("bare", bare_res)] + form_results
    }

    # ═══════════════════════════════════════════════════════════════
    #  EXPERIMENT 2: Redundant Enrichment — stack all 3 Elara forms
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*72}")
    print(f"  EXPERIMENT 2: Elara Redundant Enrichment (3 forms stacked)")
    print(f"{'═'*72}")

    # All 3 forms concatenated
    enriched_prefix = KV_SEP.join(ELARA_FORMS) + KV_SEP
    print(f"  [enriched] All 3 surface forms as prefix...")
    enriched_text = generate(llm, elara_chat, args.max_tokens, prefix=enriched_prefix)
    enriched_n, enriched_res = score(enriched_text, ELARA_CHECKS)
    print(f"    → {enriched_n}/{len(ELARA_CHECKS)} details")

    # 2 forms
    two_form_prefix = KV_SEP.join(ELARA_FORMS[:2]) + KV_SEP
    print(f"  [2-form] Forms 0+1 as prefix...")
    two_text = generate(llm, elara_chat, args.max_tokens, prefix=two_form_prefix)
    two_n, two_res = score(two_text, ELARA_CHECKS)
    print(f"    → {two_n}/{len(ELARA_CHECKS)} details")

    print_comparison("Elara enrichment", ELARA_CHECKS, [
        ("bare", bare_res),
        ("best_1", form_results[2][1]),  # form 2 was best single
        ("2-forms", two_res),
        ("3-forms", enriched_res),
    ])

    all_results["elara_enrichment"] = {
        "bare": bare_n,
        "best_single": sum(form_results[2][1].values()),
        "2_forms": two_n,
        "3_forms": enriched_n,
    }

    # ═══════════════════════════════════════════════════════════════
    #  EXPERIMENT 3: Cross-Memory Composition (Elara + Kael)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*72}")
    print(f"  EXPERIMENT 3: Cross-Memory Composition (Elara + Kael)")
    print(f"{'═'*72}")

    # Kael alone first
    kael_chat = wrap_chat(KAEL_PROMPT)
    print(f"  [kael alone] Kael with own prefix...")
    kael_text = generate(llm, kael_chat, args.max_tokens, prefix=KAEL_PREFIX + KV_SEP)
    kael_n, kael_res = score(kael_text, KAEL_CHECKS)
    print(f"    → {kael_n}/{len(KAEL_CHECKS)} Kael details")

    kael_bare_text = generate(llm, kael_chat, args.max_tokens)
    kael_bare_n, kael_bare_res = score(kael_bare_text, KAEL_CHECKS)

    # Combined prefix: Elara + Kael memories, then Elara prompt
    combined_prefix = ELARA_FORMS[2] + KV_SEP + KAEL_PREFIX + KV_SEP
    print(f"  [combined→elara] Both memories, Elara prompt...")
    ce_text = generate(llm, elara_chat, args.max_tokens, prefix=combined_prefix)
    ce_n, ce_res = score(ce_text, ELARA_CHECKS)
    print(f"    → {ce_n}/{len(ELARA_CHECKS)} Elara details")

    # Combined prefix, Kael prompt
    print(f"  [combined→kael] Both memories, Kael prompt...")
    ck_text = generate(llm, kael_chat, args.max_tokens, prefix=combined_prefix)
    ck_n, ck_res = score(ck_text, KAEL_CHECKS)
    print(f"    → {ck_n}/{len(KAEL_CHECKS)} Kael details")

    # Interference check: Elara alone vs Elara+Kael→Elara
    print(f"\n  ELARA recall comparison:")
    print_comparison("Elara w/ composition", ELARA_CHECKS, [
        ("bare", bare_res),
        ("elara_only", form_results[2][1]),
        ("elara+kael", ce_res),
    ])

    print(f"\n  KAEL recall comparison:")
    print_comparison("Kael w/ composition", KAEL_CHECKS, [
        ("bare", kael_bare_res),
        ("kael_only", kael_res),
        ("elara+kael", ck_res),
    ])

    all_results["composition"] = {
        "elara_alone": sum(form_results[2][1].values()),
        "elara_in_combined": ce_n,
        "kael_alone": kael_n,
        "kael_in_combined": ck_n,
        "elara_interference": sum(form_results[2][1].values()) - ce_n,
        "kael_interference": kael_n - ck_n,
    }

    # ═══════════════════════════════════════════════════════════════
    #  EXPERIMENT 4: Can a COMBINED prompt recall BOTH characters?
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*72}")
    print(f"  EXPERIMENT 4: Dual-Character Prompt (Both Memories Active)")
    print(f"{'═'*72}")

    dual_prompt = wrap_chat(
        "Write a short story where a woman named Elara walks through a forest "
        "and encounters a man named Kael who has come from a marketplace. "
        "Three paragraphs. Rich sensory details about both characters. /no_think"
    )

    print(f"  [dual bare] No prefix...")
    dual_bare = generate(llm, dual_prompt, args.max_tokens)
    dual_bare_e_n, dual_bare_e_res = score(dual_bare, ELARA_CHECKS)
    dual_bare_k_n, dual_bare_k_res = score(dual_bare, KAEL_CHECKS)
    print(f"    → Elara: {dual_bare_e_n}/{len(ELARA_CHECKS)}, Kael: {dual_bare_k_n}/{len(KAEL_CHECKS)}")

    print(f"  [dual combined] Both memory prefixes...")
    dual_combined = generate(llm, dual_prompt, args.max_tokens, prefix=combined_prefix)
    dual_comb_e_n, dual_comb_e_res = score(dual_combined, ELARA_CHECKS)
    dual_comb_k_n, dual_comb_k_res = score(dual_combined, KAEL_CHECKS)
    print(f"    → Elara: {dual_comb_e_n}/{len(ELARA_CHECKS)}, Kael: {dual_comb_k_n}/{len(KAEL_CHECKS)}")

    print(f"\n  Dual-character detail recall:")
    print(f"  {'Memory':<20} {'Bare':>8} {'Combined':>10} {'Delta':>8}")
    print(f"  {'─'*20} {'─'*8} {'─'*10} {'─'*8}")
    print(f"  {'Elara':<20} {dual_bare_e_n:>8} {dual_comb_e_n:>10} {dual_comb_e_n-dual_bare_e_n:>+8}")
    print(f"  {'Kael':<20} {dual_bare_k_n:>8} {dual_comb_k_n:>10} {dual_comb_k_n-dual_bare_k_n:>+8}")
    total_bare = dual_bare_e_n + dual_bare_k_n
    total_comb = dual_comb_e_n + dual_comb_k_n
    total_possible = len(ELARA_CHECKS) + len(KAEL_CHECKS)
    print(f"  {'TOTAL':<20} {total_bare:>8} {total_comb:>10} {total_comb-total_bare:>+8}")
    print(f"  {'Possible':<20} {total_possible:>8} {total_possible:>10}")

    all_results["dual_character"] = {
        "bare_elara": dual_bare_e_n, "bare_kael": dual_bare_k_n,
        "combined_elara": dual_comb_e_n, "combined_kael": dual_comb_k_n,
    }

    # ═══════════════════════════════════════════════════════════════
    #  SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*72}")
    print(f"  SUMMARY")
    print(f"{'═'*72}")
    r = all_results
    print(f"\n  Enrichment (Elara — same facts, more phrasings):")
    print(f"    1 form (best):  {r['elara_enrichment']['best_single']:>2}/13")
    print(f"    2 forms:        {r['elara_enrichment']['2_forms']:>2}/13")
    print(f"    3 forms:        {r['elara_enrichment']['3_forms']:>2}/13")

    print(f"\n  Composition (separate memories):")
    print(f"    Elara alone:    {r['composition']['elara_alone']:>2}/13")
    print(f"    Elara+Kael→E:   {r['composition']['elara_in_combined']:>2}/13  (interference: {r['composition']['elara_interference']:+d})")
    print(f"    Kael alone:     {r['composition']['kael_alone']:>2}/12")
    print(f"    Elara+Kael→K:   {r['composition']['kael_in_combined']:>2}/12  (interference: {r['composition']['kael_interference']:+d})")

    print(f"\n  Dual-character (both recalled simultaneously):")
    print(f"    Bare:     {r['dual_character']['bare_elara']:>2}E + {r['dual_character']['bare_kael']:>2}K = {r['dual_character']['bare_elara']+r['dual_character']['bare_kael']:>2}/{total_possible}")
    print(f"    Combined: {r['dual_character']['combined_elara']:>2}E + {r['dual_character']['combined_kael']:>2}K = {r['dual_character']['combined_elara']+r['dual_character']['combined_kael']:>2}/{total_possible}")

    # Save
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {args.output}")

    del llm


if __name__ == "__main__":
    main()
