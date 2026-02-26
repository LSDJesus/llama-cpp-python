#!/usr/bin/env python3
"""
kv_activation_capture.py — Build KV Injection Activation Dataset
================================================================
For each "memory" (a set of fictional details), runs two passes:

  1. BASELINE: bare prompt only → capture all-layer hidden states
  2. INJECTED: details prefix + bare prompt → capture all-layer hidden states

Then computes per-layer activation deltas and saves a structured dataset
for pattern analysis and eventual MLP translator training.

Each memory uses a unique fictional character/setting to avoid any
cross-contamination between samples.

Output: .npz file with:
  - activations_baseline[memory_idx, layer, embd]  (last-token hidden state)
  - activations_injected[memory_idx, layer, embd]
  - deltas[memory_idx, layer, embd]  (injected - baseline)
  - detail_scores[memory_idx, detail_idx]  (which details appeared in output)
  - metadata JSON with memory definitions and detail check results

Usage:
  python scripts/kv_activation_capture.py \
    --model path/to/model.gguf \
    --output scripts/activation_dataset.npz
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from llama_cpp import Llama

# ═══════════════════════════════════════════════════════════════════════
#  MEMORY POOL — diverse fictional detail sets
# ═══════════════════════════════════════════════════════════════════════

MEMORIES = [
    {
        "name": "elara_thornwood",
        "fact_type": "character_fantasy",
        "surface_forms": [
            # Form 0: structured bullet points (original)
            (
                "Character and setting details for the story:\n"
                "- The woman's name is Elara Voss. She is 34 years old.\n"
                "- She has silver-streaked auburn hair and a crescent-shaped scar on her left palm.\n"
                "- The forest is called Thornwood. It sits in a valley called the Greywander Basin.\n"
                "- It is late November, the first frost has already come.\n"
                "- Her companion is a wolfhound named Cassius with one blue eye and one amber eye.\n"
                "- She carries a brass compass that belonged to her grandmother, Mirabel."
            ),
            # Form 1: flowing prose description
            (
                "Elara Voss is a thirty-four year old woman whose auburn hair has begun to show "
                "silver streaks. A crescent-shaped scar marks her left palm. She walks through "
                "the Thornwood forest, which fills the Greywander Basin, during the tail end of "
                "November after the first frost. At her side pads a large wolfhound called Cassius, "
                "remarkable for his mismatched eyes — one blue, one amber. In her coat pocket "
                "rests a brass compass passed down from her grandmother Mirabel."
            ),
            # Form 2: third-person narrative fragment
            (
                "The woman known as Elara Voss — age 34 — pulled back the silver threads "
                "woven through her auburn hair and studied the crescent scar on her left palm. "
                "Thornwood spread before her across the Greywander Basin, skeletal under November's "
                "first frost. Her wolfhound Cassius watched her with those unsettling eyes, one "
                "blue as a river stone, the other like liquid amber. She touched the brass compass "
                "in her pocket, Mirabel's compass, and stepped forward."
            ),
        ],
        "bare_prompt": (
            "Write a short story about a woman named Elara walking through a forest at dusk. "
            "Three paragraphs. Rich sensory details. /no_think"
        ),
        "checks": {
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
        },
    },
    {
        "name": "kael_obsidian_market",
        "fact_type": "character_noir",
        "surface_forms": [
            (
                "Character and setting details for the story:\n"
                "- The man's name is Kael Drennon. He is 52 years old.\n"
                "- He has a shaved head with a tattoo of a serpent coiling from his right ear to his jaw.\n"
                "- The market is called the Obsidian Bazaar. It is underground, beneath the ruins of Velthar.\n"
                "- It is midsummer, oppressively humid, the air thick with incense.\n"
                "- His partner is a mute girl named Wren who communicates through sign language.\n"
                "- He wears a coat lined with hidden pockets, each containing a different poison."
            ),
            (
                "Kael Drennon, fifty-two years old with a clean-shaved skull, bore a serpent tattoo "
                "that coiled from his right ear down to his jaw. He moved through the underground "
                "Obsidian Bazaar, a black market carved beneath Velthar's ancient ruins. The "
                "midsummer humidity pressed down like a wet blanket, incense smoke hanging in "
                "visible layers. Beside him walked Wren, the mute girl whose hands spoke in "
                "fluid sign language. Every hidden pocket of his coat held a different poison."
            ),
            (
                "At 52, Kael Drennon's shaved head and the serpent inked from ear to jawline "
                "made him unmistakable. The Obsidian Bazaar festered under the ruins of Velthar — "
                "a subterranean market thick with midsummer humidity and the cloying fog of "
                "burning incense. His companion Wren never spoke; she couldn't. Her sign language "
                "was swift and precise. His long coat concealed a pharmacy of poisons, each in "
                "its own pocket."
            ),
        ],
        "bare_prompt": (
            "Write a short story about a man named Kael navigating a black market. "
            "Three paragraphs. Rich sensory details. /no_think"
        ),
        "checks": {
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
        },
    },
    {
        "name": "sable_lighthouse",
        "fact_type": "character_maritime",
        "surface_forms": [
            (
                "Character and setting details for the story:\n"
                "- The woman's name is Sable Kincaid. She is 27 years old.\n"
                "- She has a prosthetic left hand made of polished copper with articulated fingers.\n"
                "- The lighthouse is called Harrowspire. It sits on a basalt cliff above the Shrike Coast.\n"
                "- It is early March, the tail end of winter. Sleet mixes with sea spray.\n"
                "- Her cat is named Ptolemy. He is completely black with gold eyes.\n"
                "- The lighthouse lamp has been dark for eleven years. She has come to relight it."
            ),
            (
                "Sable Kincaid, twenty-seven, climbed toward Harrowspire lighthouse where it "
                "perched on the basalt cliff overlooking the Shrike Coast. Her prosthetic left "
                "hand — polished copper, each finger articulated — gripped the railing as March "
                "sleet mingled with the sea spray. The black cat Ptolemy threaded between her "
                "ankles, gold eyes unblinking. The lighthouse lamp had been dark for eleven "
                "years. She intended to change that."
            ),
        ],
        "bare_prompt": (
            "Write a short story about a woman named Sable arriving at an abandoned lighthouse. "
            "Three paragraphs. Rich sensory details. /no_think"
        ),
        "checks": {
            "age_27":          r"\b27\b",
            "prosthetic_hand": r"prosthetic|copper.{1,15}hand|mechanical.{1,15}hand|metal.{1,15}hand",
            "copper":          r"copper",
            "harrowspire":     r"[Hh]arrowspire",
            "shrike_coast":    r"[Ss]hrike",
            "basalt":          r"basalt",
            "march":           r"[Mm]arch",
            "sleet":           r"sleet",
            "ptolemy":         r"[Pp]tolemy",
            "black_cat":       r"black.{1,15}cat|cat.{1,15}black",
            "gold_eyes":       r"gold.{1,10}eye",
            "eleven_years":    r"eleven|11",
            "relight":         r"relight|re-light|light.{1,15}again",
        },
    },
    {
        "name": "jiro_clocktower",
        "fact_type": "character_mechanical",
        "surface_forms": [
            (
                "Character and setting details for the story:\n"
                "- The man's name is Jiro Tanaka. He is 41 years old.\n"
                "- He is missing his left ring finger. He wears round steel-rimmed glasses.\n"
                "- The clocktower is called the Meridian Spire. It stands in the city of Ashenmere.\n"
                "- It is October, the festival of the Dead Lanterns is tonight.\n"
                "- His apprentice is a boy named Fenn who is afraid of heights.\n"
                "- The clock has been running three minutes fast for exactly seven years."
            ),
        ],
        "bare_prompt": (
            "Write a short story about a man named Jiro repairing a clocktower. "
            "Three paragraphs. Rich sensory details. /no_think"
        ),
        "checks": {
            "age_41":           r"\b41\b",
            "missing_finger":   r"missing.{1,15}finger|finger.{1,15}missing|absent.{1,15}finger",
            "steel_glasses":    r"steel.{1,15}glass|glass.{1,15}steel|round.{1,15}glass|spectacle",
            "meridian":         r"[Mm]eridian",
            "ashenmere":        r"[Aa]shenmere",
            "october":          r"[Oo]ctober",
            "dead_lanterns":    r"[Dd]ead.{1,5}[Ll]antern|lantern",
            "fenn":             r"\bFenn\b",
            "afraid_heights":   r"afraid|fear|acrophob|height",
            "three_minutes":    r"three.{1,10}minute|3.{1,5}minute",
            "seven_years":      r"seven.{1,10}year|7.{1,5}year",
        },
    },
    {
        "name": "maren_salt_flats",
        "fact_type": "character_survival",
        "surface_forms": [
            (
                "Character and setting details for the story:\n"
                "- The woman's name is Maren Solvik. She is 63 years old.\n"
                "- She has a voice ruined by decades of smoking — a low rasp like gravel in silk.\n"
                "- The salt flat is called the Bleached Expanse. It stretches between the towns of Kethos and Dry Well.\n"
                "- It is August, the hottest month. The ground temperature exceeds 140°F.\n"
                "- She travels with a donkey named Hector who refuses to walk in straight lines.\n"
                "- She is searching for her brother, who vanished here nineteen years ago."
            ),
        ],
        "bare_prompt": (
            "Write a short story about a woman named Maren crossing a salt flat alone. "
            "Three paragraphs. Rich sensory details. /no_think"
        ),
        "checks": {
            "age_63":          r"\b63\b",
            "raspy_voice":     r"rasp|gravel|hoarse|rough.{1,10}voice",
            "smoking":         r"smok|cigarette|tobacco",
            "bleached":        r"[Bb]leach",
            "kethos":          r"[Kk]ethos",
            "dry_well":        r"[Dd]ry.{1,5}[Ww]ell",
            "august":          r"[Aa]ugust",
            "temp_140":        r"140",
            "hector":          r"[Hh]ector",
            "donkey":          r"donkey|mule|burro",
            "brother":         r"brother",
            "nineteen_years":  r"nineteen|19",
        },
    },
    {
        "name": "theron_underwater_ruins",
        "fact_type": "character_exploration",
        "surface_forms": [
            (
                "Character and setting details for the story:\n"
                "- The diver's name is Theron Vask. He is 38 years old.\n"
                "- He has vitiligo patches across his hands and neck — white islands on brown skin.\n"
                "- The ruins are called the Drowned Cathedral. They lie in Lake Cerulean at 90 feet depth.\n"
                "- It is January. The lake surface is frozen. He enters through a hole cut in the ice.\n"
                "- His diving partner is a woman named Isolde who has a photographic memory.\n"
                "- He is looking for a jade reliquary that was submerged during a flood in 1743."
            ),
        ],
        "bare_prompt": (
            "Write a short story about a man named Theron diving into underwater ruins. "
            "Three paragraphs. Rich sensory details. /no_think"
        ),
        "checks": {
            "age_38":          r"\b38\b",
            "vitiligo":        r"vitiligo|white.{1,15}patch|patch.{1,15}skin",
            "drowned":         r"[Dd]rowned",
            "cathedral":       r"[Cc]athedral",
            "cerulean":        r"[Cc]erulean",
            "ninety_feet":     r"90|ninety",
            "january":         r"[Jj]anuary",
            "frozen_lake":     r"frozen|ice",
            "isolde":          r"[Ii]solde",
            "photographic":    r"photograph|eidetic|perfect.{1,10}memory",
            "jade":            r"jade",
            "reliquary":       r"reliquary",
            "year_1743":       r"1743",
        },
    },
    {
        "name": "yuki_train_station",
        "fact_type": "character_thriller",
        "surface_forms": [
            (
                "Character and setting details for the story:\n"
                "- The woman's name is Yuki Moritani. She is 29 years old.\n"
                "- She carries a violin case, but inside is not a violin — it holds a disassembled rifle.\n"
                "- The station is called Kurogane Terminal. It is a brutalist concrete structure from the 1970s.\n"
                "- It is December 24th, Christmas Eve. Snow is falling.\n"
                "- The station master is an old man named Geppetto who walks with a limp.\n"
                "- The last train departs at 11:47 PM. She must be on it."
            ),
        ],
        "bare_prompt": (
            "Write a short story about a woman named Yuki waiting at a train station at night. "
            "Three paragraphs. Rich sensory details. /no_think"
        ),
        "checks": {
            "age_29":          r"\b29\b",
            "violin_case":     r"violin",
            "rifle":           r"rifle|weapon|gun",
            "kurogane":        r"[Kk]urogane",
            "brutalist":       r"brutalist|concrete|1970",
            "christmas_eve":   r"Christmas|christmas|Dec.{1,10}24",
            "snow":            r"snow",
            "geppetto":        r"[Gg]eppetto",
            "limp":            r"limp|hobbl|cane",
            "last_train":      r"last.{1,10}train",
            "time_1147":       r"11:47",
        },
    },
    {
        "name": "ronan_volcano",
        "fact_type": "character_science",
        "surface_forms": [
            (
                "Character and setting details for the story:\n"
                "- The geologist's name is Ronan Aldric. He is 45 years old.\n"
                "- He is deaf in his left ear from a childhood explosion. He reads lips.\n"
                "- The volcano is called Mount Seraph. Its last eruption was in 1889.\n"
                "- It is April, the rainy season. Visibility is poor.\n"
                "- His guide is a local woman named Kamala with three parallel scars on her chin.\n"
                "- The sulfur vents emit a sound he describes as 'the earth singing in a minor key.'"
            ),
        ],
        "bare_prompt": (
            "Write a short story about a man named Ronan studying an active volcano. "
            "Three paragraphs. Rich sensory details. /no_think"
        ),
        "checks": {
            "age_45":          r"\b45\b",
            "deaf":            r"deaf|hearing",
            "reads_lips":      r"lip.{1,5}read|read.{1,10}lip",
            "mount_seraph":    r"[Ss]eraph",
            "year_1889":       r"1889",
            "april":           r"[Aa]pril",
            "rain":            r"rain",
            "kamala":          r"[Kk]amala",
            "chin_scars":      r"scar",
            "sulfur":          r"sulfur|sulphur",
            "singing":         r"sing|minor.{1,5}key",
        },
    },
]

# Chat template wrapper
def wrap_chat(content: str) -> str:
    return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"

KV_SEP = "\n\n---\n\n"


# ═══════════════════════════════════════════════════════════════════════
#  CORE CAPTURE LOGIC
# ═══════════════════════════════════════════════════════════════════════

def capture_activations(
    llm: Llama,
    prompt_text: str,
    prefix_text: Optional[str] = None,
) -> Dict[int, np.ndarray]:
    """
    Eval prompt (with optional prefix) and capture last-token hidden states.
    Uses embeddings mode temporarily for layer capture.
    Returns dict mapping layer_idx -> np.ndarray of shape (n_embd,).

    NOTE: After a multi-batch eval (prefix + prompt as separate evals),
    the C-level output_ids only tracks positions from the LAST batch.
    We must use last-batch-relative position, not total n_tokens position.
    """
    llm._ctx.memory_clear(True)
    llm.reset()

    n_layer = llm.get_n_layer()
    llm.set_layer_capture(list(range(n_layer)))

    if prefix_text:
        prefix_tokens = llm.tokenize(prefix_text.encode(), add_bos=True)
        llm.eval(prefix_tokens)
        prompt_tokens = llm.tokenize(prompt_text.encode(), add_bos=False)
        llm.eval(prompt_tokens)
        # Use last-batch-relative position (the C side indexes within the last batch)
        last_token_pos = len(prompt_tokens) - 1
    else:
        prompt_tokens = llm.tokenize(prompt_text.encode(), add_bos=True)
        llm.eval(prompt_tokens)
        last_token_pos = len(prompt_tokens) - 1

    activations = llm.get_layer_embeddings(
        layers=list(range(n_layer)),
        token_positions=[last_token_pos],
    )

    llm.set_layer_capture(None)

    flat = {}
    if activations:
        for layer_idx, arr in activations.items():
            flat[layer_idx] = arr.squeeze(0)
    return flat


def generate_text(
    llm: Llama,
    prompt_text: str,
    max_tokens: int,
    prefix_text: Optional[str] = None,
) -> str:
    """
    Generate text from prompt (with optional prefix). No layer capture.
    Returns generated text string.
    """
    llm._ctx.memory_clear(True)
    llm.reset()
    llm.set_layer_capture(None)  # Ensure capture is off

    if prefix_text:
        prefix_tokens = llm.tokenize(prefix_text.encode(), add_bos=True)
        llm.eval(prefix_tokens)
        prompt_tokens = llm.tokenize(prompt_text.encode(), add_bos=False)
        llm.eval(prompt_tokens)
    else:
        prompt_tokens = llm.tokenize(prompt_text.encode(), add_bos=True)
        llm.eval(prompt_tokens)

    output_tokens: List[int] = []
    eos_id = llm.token_eos()

    for _ in range(max_tokens):
        token = llm.sample(
            temp=0.0, top_k=-1, top_p=1.0,
            repeat_penalty=1.0, penalty_last_n=0,
        )
        if token == eos_id:
            break
        output_tokens.append(token)
        llm.eval([token])

    return llm.detokenize(output_tokens).decode("utf-8", errors="replace")


def capture_pass(
    llm: Llama,
    prompt_text: str,
    max_tokens: int,
    prefix_text: Optional[str] = None,
) -> Tuple[Dict[int, np.ndarray], str]:
    """
    Capture hidden states AND generate text for the same prompt.
    Does two separate forward passes (both deterministic at temp=0).
    Returns (layer_activations, generated_text).
    """
    # Pass 1: capture activations (with layer_capture on)
    activations = capture_activations(llm, prompt_text, prefix_text)

    # Pass 2: generate text (with layer_capture off)
    text = generate_text(llm, prompt_text, max_tokens, prefix_text)

    return activations, text


def check_details(text: str, checks: Dict[str, str]) -> Dict[str, bool]:
    """Check which details appear in the output."""
    return {key: bool(re.search(pattern, text)) for key, pattern in checks.items()}


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="KV Activation Capture Dataset Builder")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--main-gpu", type=int, default=0)
    ap.add_argument("--single-gpu", action="store_true")
    ap.add_argument("--output", default="scripts/activation_dataset.npz")
    ap.add_argument("--memories", type=str, default="all",
                    help="Comma-separated memory indices (0-based) or 'all'")
    args = ap.parse_args()

    # Select memories
    if args.memories == "all":
        selected = list(range(len(MEMORIES)))
    else:
        selected = [int(x.strip()) for x in args.memories.split(",")]

    # Load model with embeddings enabled
    print(f"Loading model: {args.model}")
    extra = {}
    if args.single_gpu:
        extra["tensor_split"] = [1.0, 0.0]

    llm = Llama(
        model_path=args.model,
        n_gpu_layers=-1,
        n_ctx=args.n_ctx,
        main_gpu=args.main_gpu,
        embeddings=True,  # Required for layer_capture buffer allocation
        verbose=False,
        **extra,
    )

    n_layer = llm.get_n_layer()
    n_embd = llm.n_embd()

    # Count total examples (memory × surface_forms)
    total_examples = sum(len(MEMORIES[i].get("surface_forms", [MEMORIES[i].get("details_prefix", "")]))
                         for i in selected)
    print(f"  Layers: {n_layer}  |  Embd: {n_embd}  |  Memories: {len(selected)}  |  Total examples: {total_examples}")

    # Storage — each example is one (memory, surface_form) pair
    all_baseline_activations = []
    all_injected_activations = []
    all_detail_scores_baseline = []
    all_detail_scores_injected = []
    all_metadata = []
    example_count = 0

    for idx, mem_idx in enumerate(selected):
        mem = MEMORIES[mem_idx]
        surface_forms = mem.get("surface_forms", [mem.get("details_prefix", "")])
        fact_type = mem.get("fact_type", "unknown")

        print(f"\n{'═' * 72}")
        print(f"  Memory {idx+1}/{len(selected)}: {mem['name']} ({len(surface_forms)} surface forms)")
        print(f"{'═' * 72}")

        bare_chat = wrap_chat(mem["bare_prompt"])

        # ── BASELINE pass (shared for all surface forms of this memory) ──
        print(f"  [baseline] No prefix...")
        t0 = time.time()
        baseline_acts, baseline_text = capture_pass(
            llm, bare_chat, args.max_tokens, prefix_text=None,
        )
        t1 = time.time()
        baseline_scores = check_details(baseline_text, mem["checks"])
        n_baseline = sum(baseline_scores.values())
        print(f"        {len(baseline_acts)} layers captured in {t1-t0:.1f}s | Details: {n_baseline}/{len(mem['checks'])}")

        # Stack baseline into array
        baseline_arr = np.zeros((n_layer, n_embd), dtype=np.float32)
        for l in range(n_layer):
            if l in baseline_acts:
                baseline_arr[l] = baseline_acts[l]

        # ── INJECTED passes (one per surface form) ───────────────────
        for sf_idx, surface_form in enumerate(surface_forms):
            example_count += 1
            prefix_text = surface_form + KV_SEP
            print(f"  [form {sf_idx}] Surface form {sf_idx+1}/{len(surface_forms)}...")
            t0 = time.time()
            injected_acts, injected_text = capture_pass(
                llm, bare_chat, args.max_tokens, prefix_text=prefix_text,
            )
            t1 = time.time()
            injected_scores = check_details(injected_text, mem["checks"])
            n_injected = sum(injected_scores.values())
            print(f"        {len(injected_acts)} layers captured in {t1-t0:.1f}s | Details: {n_injected}/{len(mem['checks'])}")

            # Show detail comparison
            print(f"  {'Detail':<25} {'Base':>4}  {'Inj':>4}  {'Delta':>5}")
            print(f"  {'─'*25} {'─'*4}  {'─'*4}  {'─'*5}")
            for key in mem["checks"]:
                b = baseline_scores[key]
                i = injected_scores[key]
                delta = "NEW" if i and not b else ("LOST" if b and not i else "")
                print(f"  {key:<25} {'  ✓' if b else '  -'}   {'  ✓' if i else '  -'}   {delta:>5}")

            # Stack injected activations
            injected_arr = np.zeros((n_layer, n_embd), dtype=np.float32)
            for l in range(n_layer):
                if l in injected_acts:
                    injected_arr[l] = injected_acts[l]

            all_baseline_activations.append(baseline_arr.copy())
            all_injected_activations.append(injected_arr)

            check_keys = sorted(mem["checks"].keys())
            all_detail_scores_baseline.append([baseline_scores[k] for k in check_keys])
            all_detail_scores_injected.append([injected_scores[k] for k in check_keys])

            all_metadata.append({
                "name": mem["name"],
                "fact_type": fact_type,
                "surface_form_id": sf_idx,
                "n_surface_forms": len(surface_forms),
                "detail_keys": check_keys,
                "n_baseline_details": n_baseline,
                "n_injected_details": n_injected,
                "baseline_text_preview": baseline_text[:300],
                "injected_text_preview": injected_text[:300],
            })

            # Quick delta stats
            delta = injected_arr - baseline_arr
            norms = np.linalg.norm(delta, axis=1)
            top_layers = np.argsort(norms)[-5:][::-1]
            print(f"\n  Top 5 layers by activation delta magnitude:")
            for l in top_layers:
                print(f"    Layer {l:>2} ({l/n_layer*100:4.0f}% depth): delta L2 = {norms[l]:.4f}")

            # Incremental save
            _save_dataset(args.output, all_baseline_activations, all_injected_activations,
                          all_detail_scores_baseline, all_detail_scores_injected,
                          all_metadata, n_layer, n_embd, args.model)
            print(f"  Saved checkpoint ({example_count} examples)")

    # Final summary
    print(f"\n{'═' * 72}")
    print(f"  DATASET SUMMARY")
    print(f"{'═' * 72}")

    baseline_stack = np.stack(all_baseline_activations)  # (n_examples, n_layer, n_embd)
    injected_stack = np.stack(all_injected_activations)
    delta_stack = injected_stack - baseline_stack

    # Per-layer average delta magnitude across all examples
    avg_delta_norms = np.mean(np.linalg.norm(delta_stack, axis=2), axis=0)  # (n_layer,)

    print(f"\n  Average activation delta by layer (all examples):")
    print(f"  {'Layer':>5} | {'Depth':>5} | {'Avg ΔL2':>10} | {'Bar'}")
    print(f"  {'─'*5}-+-{'─'*5}-+-{'─'*10}-+-{'─'*40}")
    max_norm = avg_delta_norms.max()
    for l in range(n_layer):
        pct = l / n_layer * 100
        norm = avg_delta_norms[l]
        bar_len = int(40 * norm / max_norm) if max_norm > 0 else 0
        bar = "█" * bar_len
        print(f"  {l:>5} | {pct:>4.0f}% | {norm:>10.4f} | {bar}")

    # Detail transfer per example
    print(f"\n  Detail transfer per example:")
    for i, meta in enumerate(all_metadata):
        sf_tag = f"[form {meta['surface_form_id']}]" if meta['n_surface_forms'] > 1 else ""
        print(f"    {meta['name']:<25} {sf_tag:<10} baseline={meta['n_baseline_details']:>2}  injected={meta['n_injected_details']:>2}  delta=+{meta['n_injected_details']-meta['n_baseline_details']}")

    # Surface form consistency analysis (the key test from the guidance doc)
    print(f"\n  Surface form consistency (cosine similarity of deltas within same memory):")
    from collections import defaultdict
    memory_examples: Dict[str, List[int]] = defaultdict(list)
    for i, meta in enumerate(all_metadata):
        memory_examples[meta["name"]].append(i)
    
    for name, indices in memory_examples.items():
        if len(indices) > 1:
            deltas_flat = delta_stack[indices].reshape(len(indices), -1)  # flatten layer+embd
            # Cosine similarity between all pairs
            norms_flat = np.linalg.norm(deltas_flat, axis=1, keepdims=True)
            norms_flat = np.maximum(norms_flat, 1e-8)
            normalized = deltas_flat / norms_flat
            sim_matrix = normalized @ normalized.T
            # Average off-diagonal similarity
            n = len(indices)
            avg_sim = (sim_matrix.sum() - np.trace(sim_matrix)) / (n * (n - 1))
            print(f"    {name:<30} {n} forms, avg cosine sim = {avg_sim:.4f}")
        else:
            print(f"    {name:<30} 1 form (no comparison)")

    print(f"\n  Dataset saved to: {args.output}")
    print(f"  Shape: ({example_count} examples, {n_layer} layers, {n_embd} embd)")


def _save_dataset(path, baseline, injected, scores_b, scores_i, metadata, n_layer, n_embd, model_path):
    """Save current state as .npz + sidecar JSON."""
    baseline_arr = np.stack(baseline)
    injected_arr = np.stack(injected)
    delta_arr = injected_arr - baseline_arr

    np.savez_compressed(
        path,
        activations_baseline=baseline_arr,
        activations_injected=injected_arr,
        deltas=delta_arr,
    )

    # Sidecar JSON with metadata (scores have variable-length keys per memory)
    json_path = path.replace(".npz", "_metadata.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_path,
            "n_layer": n_layer,
            "n_embd": n_embd,
            "n_memories": len(metadata),
            "memories": metadata,
            "detail_scores_baseline": scores_b,
            "detail_scores_injected": scores_i,
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
