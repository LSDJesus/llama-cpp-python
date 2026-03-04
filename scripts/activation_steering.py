#!/usr/bin/env python3
"""
Activation Steering via Control Vectors — Semantic Memory Injection Test

Uses llama.cpp's built-in control vector mechanism (llama_set_adapter_cvec)
to inject activation deltas into the residual stream during generation.

The experiment:
1. Run prompt WITH the Elara prefix → capture hidden states at critical layers
2. Run prompt WITHOUT the prefix → capture baseline hidden states
3. Compute delta = (with_prefix - without_prefix) at each layer
4. Apply delta as a control vector during generation (no prefix in prompt)
5. Check if the model "knows" about Elara despite never seeing the prefix

This tests the core premise of semantic memory injection:
can activation-level knowledge transfer replace explicit context?

Usage:
    python scripts/activation_steering.py --model <path.gguf> [options]
"""

from __future__ import annotations

import argparse
import ctypes
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

import llama_cpp
from llama_cpp import Llama

# ═══════════════════════════════════════════════════════════════════════
#  TEST CONTENT — zero-contamination fictional facts
# ═══════════════════════════════════════════════════════════════════════

ELARA_PREFIX = """Character and setting details for the story:
- The woman's name is Elara Voss. She is 34 years old.
- She has silver-streaked auburn hair and a crescent-shaped scar on her left palm.
- The forest is called Thornwood. It sits in a valley called the Greywander Basin.
- It is late November, the first frost has already come.
- Her companion is a wolfhound named Cassius with one blue eye and one amber eye.
- She carries a brass compass that belonged to her grandmother, Mirabel.
---
"""

ELARA_PROMPT = (
    "Write a short story about a woman named Elara walking through "
    "a forest at dusk. Three paragraphs. Rich sensory details. /no_think"
)

RECALL_CHECKS = {
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
    "brass_compass": r"brass.*compass|compass.*brass",
    "mirabel":       r"[Mm]irabel",
}

# All 40 layers of Qwen3-14B. Critical band is ~14-22 based on our experiments.
CRITICAL_LAYERS = list(range(14, 23))
ALL_LAYERS = list(range(0, 40))


def wrap_chat(user_text: str) -> str:
    """Wrap in Qwen3 chat template."""
    return (
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ═══════════════════════════════════════════════════════════════════════
#  ACTIVATION CAPTURE
# ═══════════════════════════════════════════════════════════════════════

def capture_activations(
    llm: Llama,
    tokens: List[int],
    layers: List[int],
) -> Dict[int, np.ndarray]:
    """Run tokens through the model and capture hidden states at specified layers.
    
    Returns dict mapping layer → 1D array of shape (n_embd,) for the LAST token.
    """
    llm._ctx.memory_clear(True)
    llm.reset()
    llm.set_layer_capture(layers)
    
    llm.eval(tokens)
    
    last_pos = len(tokens) - 1
    result = llm.get_layer_embeddings(layers=layers, token_positions=[last_pos])
    
    llm.set_layer_capture(None)
    
    out = {}
    for L in layers:
        if L in result and last_pos in result[L]:
            out[L] = result[L][last_pos].copy()
    
    return out


def compute_activation_deltas(
    llm: Llama,
    prefix_tokens: List[int],
    prompt_tokens: List[int],
    prompt_tokens_bos: List[int],
    layers: List[int],
) -> Dict[int, np.ndarray]:
    """Compute activation delta = (with_prefix - without_prefix) at each layer.
    
    The delta represents "what knowing the prefix does to the model's hidden states."
    """
    # WITH prefix: eval prefix, then prompt
    llm._ctx.memory_clear(True)
    llm.reset()
    llm.set_layer_capture(layers)
    llm.eval(prefix_tokens)
    llm.eval(prompt_tokens)
    last_pos_with = len(prompt_tokens) - 1  # within last batch
    states_with = llm.get_layer_embeddings(layers=layers, token_positions=[last_pos_with])
    llm.set_layer_capture(None)
    
    # WITHOUT prefix: eval prompt with BOS
    llm._ctx.memory_clear(True)
    llm.reset()
    llm.set_layer_capture(layers)
    llm.eval(prompt_tokens_bos)
    last_pos_without = len(prompt_tokens_bos) - 1
    states_without = llm.get_layer_embeddings(layers=layers, token_positions=[last_pos_without])
    llm.set_layer_capture(None)
    
    deltas = {}
    if states_with is not None and states_without is not None:
        for L in layers:
            if L in states_with and L in states_without:
                # get_layer_embeddings returns Dict[layer, array of shape (n_positions, n_embd)]
                # We requested one position, so index [0] gets the first (only) position
                deltas[L] = states_with[L][0] - states_without[L][0]
    
    return deltas


# ═══════════════════════════════════════════════════════════════════════
#  CONTROL VECTOR APPLICATION
# ═══════════════════════════════════════════════════════════════════════

def apply_steering_vector(
    llm: Llama,
    deltas: Dict[int, np.ndarray],
    layer_start: int,
    layer_end: int,
    scale: float = 1.0,
) -> None:
    """Apply activation deltas as a control vector via llama_set_adapter_cvec.
    
    The data format is: n_embd floats per layer, starting from layer 1,
    packed contiguously. Layers outside the delta dict get zeros.
    """
    n_embd = llm.n_embd()
    n_layers = llm.get_n_layer()
    
    # Build the control vector buffer: n_embd * (n_layers - 1) floats
    # (layer 0 is never included, buffer starts at layer 1)
    cvec_data = np.zeros(n_embd * (n_layers - 1), dtype=np.float32)
    
    for L, delta in deltas.items():
        if L < 1 or L >= n_layers:
            continue
        offset = (L - 1) * n_embd
        cvec_data[offset:offset + n_embd] = delta.ravel()[:n_embd] * scale
    
    # Apply via the C API
    data_ptr = cvec_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    ret = llama_cpp.llama_set_adapter_cvec(
        llm._ctx.ctx,
        data_ptr,
        len(cvec_data),
        n_embd,
        layer_start,
        layer_end,
    )
    
    if ret != 0:
        print(f"  WARNING: llama_set_adapter_cvec returned {ret}")


def clear_steering_vector(llm: Llama) -> None:
    """Remove any active control vector."""
    llama_cpp.llama_set_adapter_cvec(
        llm._ctx.ctx,
        None,  # NULL data = clear
        0,
        llm.n_embd(),
        0, 0,
    )


# ═══════════════════════════════════════════════════════════════════════
#  GENERATION & EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def generate(
    llm: Llama,
    prompt: str,
    max_tokens: int = 256,
) -> str:
    """Generate text from a prompt using greedy decoding."""
    llm._ctx.memory_clear(True)
    llm.reset()
    
    tokens = llm.tokenize(prompt.encode(), add_bos=True)
    llm.eval(tokens)
    
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
    
    return llm.detokenize(output).decode("utf-8", errors="replace")


def check_recall(text: str) -> Dict[str, bool]:
    """Check which Elara facts appear in the generated text."""
    return {k: bool(re.search(p, text, re.I)) for k, p in RECALL_CHECKS.items()}


def print_recall(hits: Dict[str, bool]) -> int:
    """Print recall results and return total count."""
    total = sum(hits.values())
    print(f"    Recall: {total}/13")
    for k, v in hits.items():
        mark = "✓" if v else "-"
        print(f"      {k}: {mark}")
    return total


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Activation steering via control vectors")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--main-gpu", type=int, default=0)
    parser.add_argument("--single-gpu", action="store_true")
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--scales", type=str, default="0.01,0.05,0.1,0.2,0.5,1.0",
                        help="Comma-separated scale factors to test")
    parser.add_argument("--layers", type=str, default="14-22",
                        help="Layer range for steering (e.g., '14-22' or '0-39')")
    args = parser.parse_args()
    
    extra = {}
    if args.single_gpu:
        extra["tensor_split"] = [1.0, 0.0]
    
    # Parse layer range
    if "-" in args.layers:
        start, end = args.layers.split("-")
        layers = list(range(int(start), int(end) + 1))
    else:
        layers = [int(x) for x in args.layers.split(",")]
    layer_start = min(layers)
    layer_end = max(layers)
    
    scales = [float(s) for s in args.scales.split(",")]
    
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
    
    n_embd = llm.n_embd()
    n_layer = llm.get_n_layer()
    print(f"  n_embd={n_embd}, n_layer={n_layer}")
    print(f"  Steering layers: {layer_start}-{layer_end}")
    print(f"  Scales: {scales}")
    
    chat_prompt = wrap_chat(ELARA_PROMPT)
    full_prompt = ELARA_PREFIX + chat_prompt
    
    prefix_tokens = llm.tokenize(ELARA_PREFIX.encode(), add_bos=True)
    prompt_tokens = llm.tokenize(chat_prompt.encode(), add_bos=False)
    prompt_tokens_bos = llm.tokenize(chat_prompt.encode(), add_bos=True)
    
    print(f"  Prefix: {len(prefix_tokens)} tokens")
    print(f"  Prompt: {len(prompt_tokens)} tokens")
    
    # ─── Step 1: Capture activation deltas ───────────────────────────
    print(f"\n{'═' * 72}")
    print(f"  STEP 1: Capture Activation Deltas")
    print(f"{'═' * 72}")
    
    deltas = compute_activation_deltas(
        llm, prefix_tokens, prompt_tokens, prompt_tokens_bos, layers
    )
    
    for L in sorted(deltas.keys()):
        d = deltas[L]
        print(f"  Layer {L}: delta norm={np.linalg.norm(d):.4f}, "
              f"mean={np.mean(d):.6f}, max={np.max(np.abs(d)):.4f}")
    
    # ─── Step 2: Baseline generation (no prefix, no steering) ───────
    print(f"\n{'═' * 72}")
    print(f"  STEP 2: Baseline Generation (no prefix)")
    print(f"{'═' * 72}")
    
    clear_steering_vector(llm)
    text_bare = generate(llm, chat_prompt, args.max_tokens)
    hits_bare = check_recall(text_bare)
    print_recall(hits_bare)
    print(f"    Preview: {text_bare[:200].replace(chr(10), ' ')}")
    
    # ─── Step 3: Full prefix generation (ground truth) ──────────────
    print(f"\n{'═' * 72}")
    print(f"  STEP 3: Full Prefix Generation (ground truth)")
    print(f"{'═' * 72}")
    
    clear_steering_vector(llm)
    text_full = generate(llm, full_prompt, args.max_tokens)
    hits_full = check_recall(text_full)
    print_recall(hits_full)
    print(f"    Preview: {text_full[:200].replace(chr(10), ' ')}")
    
    # ─── Step 4: Steered generation at various scales ───────────────
    print(f"\n{'═' * 72}")
    print(f"  STEP 4: Steered Generation (no prefix, control vector active)")
    print(f"{'═' * 72}")
    
    results = {}
    results["bare"] = sum(hits_bare.values())
    results["full_prefix"] = sum(hits_full.values())
    
    for scale in scales:
        print(f"\n  --- Scale {scale:.4f} ---")
        apply_steering_vector(llm, deltas, layer_start, layer_end, scale=scale)
        
        text = generate(llm, chat_prompt, args.max_tokens)
        hits = check_recall(text)
        total = print_recall(hits)
        print(f"    Preview: {text[:200].replace(chr(10), ' ')}")
        
        results[f"scale_{scale}"] = total
        
        # Clear for next iteration
        clear_steering_vector(llm)
    
    # ─── Summary ────────────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print(f"  SUMMARY")
    print(f"{'═' * 72}")
    
    print(f"  {'Condition':<25} {'Recall':>8}")
    print(f"  {'─' * 35}")
    for name, score in results.items():
        marker = " ★" if score == max(results.values()) and score > 0 else ""
        print(f"  {name:<25} {score:>5}/13{marker}")
    
    print(f"\n  Delta norms per layer:")
    for L in sorted(deltas.keys()):
        print(f"    Layer {L}: {np.linalg.norm(deltas[L]):.1f}")
    
    # Save deltas for further analysis
    np.savez(
        "scripts/steering_deltas.npz",
        **{f"layer_{L}": deltas[L] for L in sorted(deltas.keys())}
    )
    print(f"\n  Deltas saved to: scripts/steering_deltas.npz")
    
    del llm


if __name__ == "__main__":
    main()
