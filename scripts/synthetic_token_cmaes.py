#!/usr/bin/env python3
"""
Synthetic Token Optimization via CMA-ES
========================================

The nuclear experiment: compress an entire 117-token character sheet into
N synthetic embedding vectors (starting with N=1) that, when processed
through the model's normal forward pass, reproduce the activation pattern
of the full prefix at the critical transformer layers.

Architecture:
  1. Eval full prefix → capture layer activations (target)
  2. Eval bare prompt → capture layer activations (baseline)
  3. Target delta = target - baseline at critical layers
  4. CMA-ES loop:
     a. Propose candidate embedding vector(s)
     b. Inject via llama_batch.embd → llama_decode
     c. Eval same prompt on top → capture activations
     d. Compute loss = distance to target delta
     e. Feed loss back to CMA-ES
  5. Final test: inject best synthetic token → generate → score detail recall

Uses llama_batch with embd field for direct embedding injection.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import cma
import numpy as np
import numpy.typing as npt

import llama_cpp
from llama_cpp import Llama
from llama_cpp._internals import LlamaBatch
from llama_cpp._utils import suppress_stdout_stderr

# ═══════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

# Critical band from layer sensitivity experiments
# Full band for capture/validation: 14-22
# Optimization targets SEMANTIC layers (14-18) only — layers 19-22 carry
# the universal "context was present" signal (99.7% cosine match trivially)
# but NOT fact-specific content. Optimizing on them causes the synthetic
# token to encode structural markup rather than Elara facts.
CRITICAL_LAYERS = list(range(14, 23))  # Full band for capture
SEMANTIC_LAYERS = list(range(14, 19))  # Optimization target: semantic content layers

ELARA_PREFIX = (
    "Character and setting details for the story:\n"
    "- The woman's name is Elara Voss. She is 34 years old.\n"
    "- She has silver-streaked auburn hair and a crescent-shaped scar on her left palm.\n"
    "- The forest is called Thornwood. It sits in a valley called the Greywander Basin.\n"
    "- It is late November, the first frost has already come.\n"
    "- Her companion is a wolfhound named Cassius with one blue eye and one amber eye.\n"
    "- She carries a brass compass that belonged to her grandmother, Mirabel."
)

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

KV_SEP = "\n---\n"


def wrap_chat(user_msg: str) -> str:
    return (
        "<|im_start|>user\n" + user_msg + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# ═══════════════════════════════════════════════════════════════════════
#  EMBEDDING INJECTION
# ═══════════════════════════════════════════════════════════════════════

def eval_embedding(llm: Llama, embedding: np.ndarray, pos_start: int = 0,
                    seq_id: int = 0) -> None:
    """Inject a synthetic embedding vector and decode it through the model.
    
    Args:
        llm: Llama instance
        embedding: float32 array of shape (n_tokens, n_embd) or (n_embd,)
        pos_start: Starting position in the KV cache
        seq_id: Sequence ID
    """
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    
    n_tokens, n_embd = embedding.shape
    assert n_embd == llm.n_embd(), f"Embedding dim {n_embd} != model dim {llm.n_embd()}"
    
    # Allocate batch with embedding mode
    batch = llama_cpp.llama_batch_init(n_tokens, n_embd, 1)
    try:
        batch.n_tokens = n_tokens
        
        # Fill embedding data
        embd_arr = (ctypes.c_float * (n_tokens * n_embd))(*embedding.ravel().tolist())
        ctypes.memmove(batch.embd, embd_arr, n_tokens * n_embd * ctypes.sizeof(ctypes.c_float))
        
        # Set positions, seq_ids, logits
        for i in range(n_tokens):
            batch.pos[i] = pos_start + i
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = seq_id
            batch.logits[i] = 0
        batch.logits[n_tokens - 1] = 1  # Output logits for last token
        
        # Decode
        ret = llama_cpp.llama_decode(llm._ctx.ctx, batch)
        if ret != 0:
            raise RuntimeError(f"llama_decode failed with code {ret}")
        
        # Update internal token tracking
        llm.n_tokens = pos_start + n_tokens
        
    finally:
        llama_cpp.llama_batch_free(batch)


# ═══════════════════════════════════════════════════════════════════════
#  ACTIVATION CAPTURE
# ═══════════════════════════════════════════════════════════════════════

def capture_activation_delta(
    llm: Llama,
    prompt_tokens: List[int],
    prefix_tokens: Optional[List[int]],
    layers: List[int],
) -> Dict[int, np.ndarray]:
    """Capture per-layer activations for the last prompt token.
    
    NOTE: After multi-batch eval (prefix + prompt as separate evals),
    C-level output_ids only tracks positions from the LAST batch.
    Must use last-batch-relative position, not total n_tokens.
    
    Returns dict mapping layer -> activation vector (n_embd,)
    """
    llm._ctx.memory_clear(True)
    llm.reset()
    llm.set_layer_capture(layers)
    
    if prefix_tokens is not None:
        llm.eval(prefix_tokens)
    
    llm.eval(prompt_tokens)
    
    # Use last-batch-relative position (C side indexes within last batch)
    last_token_pos = len(prompt_tokens) - 1
    
    result = llm.get_layer_embeddings(
        layers=layers,
        token_positions=[last_token_pos],
    )
    
    llm.set_layer_capture(None)
    
    if result is None:
        return {}
    
    # Squeeze to (n_embd,) per layer
    return {layer: arr[0] for layer, arr in result.items()}


def capture_activation_with_embedding(
    llm: Llama,
    synthetic_emb: np.ndarray,
    prompt_tokens: List[int],
    layers: List[int],
) -> Dict[int, np.ndarray]:
    """Inject synthetic embedding, then eval prompt, capture activations.
    
    Layout: [BOS] [synthetic embedding] [prompt tokens]
    Matches the baseline/target layout which starts with BOS.
    
    NOTE: C-level output_ids tracks positions within the LAST batch only.
    """
    llm._ctx.memory_clear(True)
    llm.reset()
    llm.set_layer_capture(layers)
    
    # Eval BOS token first
    bos = llm.token_bos()
    bos_batch = llama_cpp.llama_batch_init(1, 0, 1)
    try:
        bos_batch.n_tokens = 1
        bos_batch.token[0] = bos
        bos_batch.pos[0] = 0
        bos_batch.n_seq_id[0] = 1
        bos_batch.seq_id[0][0] = 0
        bos_batch.logits[0] = 0
        ret = llama_cpp.llama_decode(llm._ctx.ctx, bos_batch)
        if ret != 0:
            raise RuntimeError(f"BOS decode failed: {ret}")
        llm.n_tokens = 1
    finally:
        llama_cpp.llama_batch_free(bos_batch)
    
    # Inject synthetic embedding after BOS
    n_synth = synthetic_emb.shape[0] if synthetic_emb.ndim > 1 else 1
    eval_embedding(llm, synthetic_emb, pos_start=1)
    pos_after_synth = 1 + n_synth
    
    # Eval prompt tokens after synthetic embedding
    n_prompt = len(prompt_tokens)
    batch = llama_cpp.llama_batch_init(n_prompt, 0, 1)
    try:
        batch.n_tokens = n_prompt
        for i in range(n_prompt):
            batch.token[i] = prompt_tokens[i]
            batch.pos[i] = pos_after_synth + i
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = 0
            batch.logits[i] = 0
        batch.logits[n_prompt - 1] = 1
        
        ret = llama_cpp.llama_decode(llm._ctx.ctx, batch)
        if ret != 0:
            raise RuntimeError(f"llama_decode failed: {ret}")
        llm.n_tokens = pos_after_synth + n_prompt
    finally:
        llama_cpp.llama_batch_free(batch)
    
    # Use last-batch-relative position (C side indexes within last batch)
    last_token_pos = n_prompt - 1
    
    result = llm.get_layer_embeddings(
        layers=layers,
        token_positions=[last_token_pos],
    )
    
    llm.set_layer_capture(None)
    
    if result is None:
        return {}
    
    return {layer: arr[0] for layer, arr in result.items()}


# ═══════════════════════════════════════════════════════════════════════
#  LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def compute_loss(
    produced: Dict[int, np.ndarray],
    target_delta: Dict[int, np.ndarray],
    baseline: Dict[int, np.ndarray],
) -> float:
    """Compute loss between produced activations and target.
    
    Loss = mean NORMALIZED cosine distance across critical layers.
    Each layer's delta is unit-normalized before comparison so that
    all layers contribute equally regardless of absolute magnitude.
    This prevents high-magnitude layers (19-22, norm~650) from
    dominating over lower-magnitude ones (14-18, norm~15-23).
    """
    losses = []
    for layer in target_delta:
        if layer not in produced or layer not in baseline:
            continue
        
        # The delta the synthetic token produced
        produced_delta = produced[layer] - baseline[layer]
        target = target_delta[layer]
        
        prod_norm = np.linalg.norm(produced_delta)
        targ_norm = np.linalg.norm(target)
        
        if prod_norm < 1e-8 or targ_norm < 1e-8:
            losses.append(2.0)  # Max loss if zero vector
            continue
        
        # Cosine distance (direction match) — each layer equally weighted
        cos_sim = np.dot(produced_delta, target) / (prod_norm * targ_norm)
        cos_loss = 1.0 - cos_sim  # 0 = perfect, 2 = opposite
        
        # Magnitude ratio penalty — log-scale so 10x off ≈ 2x off
        log_mag_ratio = abs(np.log(prod_norm / targ_norm))
        mag_loss = min(log_mag_ratio, 2.0)  # Cap at 2.0
        
        # Combined: 80% direction, 20% magnitude
        losses.append(0.8 * cos_loss + 0.2 * mag_loss)
    
    if not losses:
        return 4.0  # Max loss if no layers matched
    
    return float(np.mean(losses))


def capture_logits_with_embedding(
    llm: Llama,
    synthetic_emb: np.ndarray,
    prompt_tokens: List[int],
) -> np.ndarray:
    """Inject synthetic embedding, eval prompt, return logit vector for last token.
    
    Layout: [BOS] [synthetic embedding] [prompt tokens]
    Returns: logit array of shape (n_vocab,)
    """
    llm._ctx.memory_clear(True)
    llm.reset()
    llm.set_layer_capture(None)  # No layer capture needed for logits
    
    # BOS
    bos = llm.token_bos()
    bos_batch = llama_cpp.llama_batch_init(1, 0, 1)
    try:
        bos_batch.n_tokens = 1
        bos_batch.token[0] = bos
        bos_batch.pos[0] = 0
        bos_batch.n_seq_id[0] = 1
        bos_batch.seq_id[0][0] = 0
        bos_batch.logits[0] = 0
        ret = llama_cpp.llama_decode(llm._ctx.ctx, bos_batch)
        if ret != 0:
            raise RuntimeError(f"BOS decode failed: {ret}")
        llm.n_tokens = 1
    finally:
        llama_cpp.llama_batch_free(bos_batch)
    
    # Synthetic embedding
    n_synth = synthetic_emb.shape[0] if synthetic_emb.ndim > 1 else 1
    eval_embedding(llm, synthetic_emb, pos_start=1)
    pos_after = 1 + n_synth
    
    # Prompt tokens
    n_prompt = len(prompt_tokens)
    batch = llama_cpp.llama_batch_init(n_prompt, 0, 1)
    try:
        batch.n_tokens = n_prompt
        for i in range(n_prompt):
            batch.token[i] = prompt_tokens[i]
            batch.pos[i] = pos_after + i
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = 0
            batch.logits[i] = 0
        batch.logits[n_prompt - 1] = 1
        ret = llama_cpp.llama_decode(llm._ctx.ctx, batch)
        if ret != 0:
            raise RuntimeError(f"decode failed: {ret}")
        llm.n_tokens = pos_after + n_prompt
    finally:
        llama_cpp.llama_batch_free(batch)
    
    # Get logits
    logits_ptr = llm._ctx.get_logits()
    return np.ctypeslib.as_array(logits_ptr, shape=(llm._n_vocab,)).copy()


def capture_target_logits(
    llm: Llama,
    prefix_tokens: List[int],
    prompt_tokens: List[int],
) -> np.ndarray:
    """Capture the logit vector produced by [prefix + prompt] (full context).
    
    Returns: logit array of shape (n_vocab,)
    """
    llm._ctx.memory_clear(True)
    llm.reset()
    llm.set_layer_capture(None)
    
    # Prefix includes BOS
    llm.eval(prefix_tokens)
    llm.eval(prompt_tokens)
    
    logits_ptr = llm._ctx.get_logits()
    return np.ctypeslib.as_array(logits_ptr, shape=(llm._n_vocab,)).copy()


def compute_logit_loss(produced_logits: np.ndarray, target_logits: np.ndarray,
                        top_k: int = 200) -> float:
    """Compute KL divergence between target and produced logit distributions.
    
    Only considers top-K tokens from the target to focus on the
    tokens the model actually considers plausible.
    
    Returns loss in [0, ~20] range.
    """
    # Softmax with numerical stability
    def stable_softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()
    
    p_target = stable_softmax(target_logits)
    p_produced = stable_softmax(produced_logits)
    
    # Focus on top-K tokens from target distribution
    top_indices = np.argsort(p_target)[-top_k:]
    
    p_t = p_target[top_indices]
    p_p = p_produced[top_indices]
    
    # Clamp to avoid log(0)
    eps = 1e-10
    p_t = np.clip(p_t, eps, 1.0)
    p_p = np.clip(p_p, eps, 1.0)
    
    # KL(target || produced)
    kl = np.sum(p_t * np.log(p_t / p_p))
    
    return float(kl)


# ═══════════════════════════════════════════════════════════════════════
#  GENERATION & SCORING
# ═══════════════════════════════════════════════════════════════════════

def generate_with_embedding(llm: Llama, synthetic_emb: np.ndarray,
                              prompt: str, max_tokens: int) -> str:
    """Generate text using synthetic embedding as prefix.
    
    Layout: [BOS token] [synthetic embedding(s)] [prompt tokens...]
    BOS is eval'd as a real token first so the model has proper framing.
    """
    llm.set_layer_capture(None)  # Disable capture for generation
    llm._ctx.memory_clear(True)
    llm.reset()
    
    # Eval BOS token first at position 0
    bos = llm.token_bos()
    bos_batch = llama_cpp.llama_batch_init(1, 0, 1)
    try:
        bos_batch.n_tokens = 1
        bos_batch.token[0] = bos
        bos_batch.pos[0] = 0
        bos_batch.n_seq_id[0] = 1
        bos_batch.seq_id[0][0] = 0
        bos_batch.logits[0] = 0
        ret = llama_cpp.llama_decode(llm._ctx.ctx, bos_batch)
        if ret != 0:
            raise RuntimeError(f"BOS decode failed: {ret}")
        llm.n_tokens = 1
    finally:
        llama_cpp.llama_batch_free(bos_batch)
    
    # Inject synthetic embedding after BOS
    eval_embedding(llm, synthetic_emb, pos_start=1)
    
    # Eval prompt
    n_synth = synthetic_emb.shape[0] if synthetic_emb.ndim > 1 else 1
    pos_after_synth = 1 + n_synth
    prompt_tokens = llm.tokenize(prompt.encode(), add_bos=False)
    
    n_prompt = len(prompt_tokens)
    batch = llama_cpp.llama_batch_init(n_prompt, 0, 1)
    try:
        batch.n_tokens = n_prompt
        for i in range(n_prompt):
            batch.token[i] = prompt_tokens[i]
            batch.pos[i] = pos_after_synth + i
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = 0
            batch.logits[i] = 0
        batch.logits[n_prompt - 1] = 1
        
        ret = llama_cpp.llama_decode(llm._ctx.ctx, batch)
        if ret != 0:
            raise RuntimeError(f"llama_decode failed: {ret}")
        llm.n_tokens = pos_after_synth + n_prompt
    finally:
        llama_cpp.llama_batch_free(batch)
    
    # Generate
    output_tokens = []
    eos = llm.token_eos()
    for _ in range(max_tokens):
        logits_ptr = llm._ctx.get_logits()
        logits = np.ctypeslib.as_array(logits_ptr, shape=(llm._n_vocab,)).copy()
        
        # Greedy
        tok = int(np.argmax(logits))
        if tok == eos:
            break
        output_tokens.append(tok)
        
        # Eval the generated token
        next_batch = llama_cpp.llama_batch_init(1, 0, 1)
        try:
            next_batch.n_tokens = 1
            next_batch.token[0] = tok
            next_batch.pos[0] = llm.n_tokens
            next_batch.n_seq_id[0] = 1
            next_batch.seq_id[0][0] = 0
            next_batch.logits[0] = 1
            ret = llama_cpp.llama_decode(llm._ctx.ctx, next_batch)
            if ret != 0:
                break
            llm.n_tokens += 1
        finally:
            llama_cpp.llama_batch_free(next_batch)
    
    return llm.detokenize(output_tokens).decode("utf-8", errors="replace")


def generate_fresh(llm: Llama, prompt: str, max_tokens: int,
                    prefix: Optional[str] = None) -> str:
    """Standard generation for baselines."""
    llm.set_layer_capture(None)
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
    
    output_tokens = []
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


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Synthetic Token CMA-ES Optimizer")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-synthetic", type=int, default=1,
                    help="Number of synthetic tokens to optimize")
    ap.add_argument("--max-evals", type=int, default=500,
                    help="Max CMA-ES evaluations")
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="Max tokens for final generation test")
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--main-gpu", type=int, default=0)
    ap.add_argument("--single-gpu", action="store_true")
    ap.add_argument("--sigma", type=float, default=0.5,
                    help="CMA-ES initial step size")
    ap.add_argument("--output", default="scripts/synthetic_token_results.json")
    args = ap.parse_args()
    
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
        embeddings=True,  # Required for layer capture (plural!)
        **extra,
    )
    
    n_embd = llm.n_embd()
    n_layer = llm.get_n_layer()
    n_synth = args.n_synthetic
    opt_dim = n_synth * n_embd  # Total parameters to optimize
    
    print(f"  n_embd={n_embd}, n_layer={n_layer}")
    print(f"  Optimizing {n_synth} synthetic token(s) = {opt_dim} parameters")
    print(f"  Capturing layers: {CRITICAL_LAYERS}")
    
    full_prefix = ELARA_PREFIX + KV_SEP
    chat_prompt = wrap_chat(ELARA_PROMPT)
    
    # Tokenize
    prefix_tokens = llm.tokenize(full_prefix.encode(), add_bos=True)
    prompt_tokens = llm.tokenize(chat_prompt.encode(), add_bos=False)
    prompt_tokens_bos = llm.tokenize(chat_prompt.encode(), add_bos=True)
    
    print(f"  Prefix: {len(prefix_tokens)} tokens")
    print(f"  Prompt: {len(prompt_tokens)} tokens")
    
    # ═══════════════════════════════════════════════════════════════
    #  STEP 1: Capture target activations AND target logits
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*72}")
    print(f"  STEP 1: Capture Target Activations + Logits")
    print(f"{'═'*72}")
    
    # Baseline: prompt only, no prefix
    print("  Capturing baseline (no prefix)...")
    baseline = capture_activation_delta(
        llm, prompt_tokens_bos, prefix_tokens=None, layers=CRITICAL_LAYERS
    )
    print(f"    Got {len(baseline)} layers")
    for l in sorted(baseline)[:3]:
        print(f"    Layer {l}: norm={np.linalg.norm(baseline[l]):.4f}")
    
    # Target: full prefix + prompt
    print("  Capturing target (full prefix)...")
    target = capture_activation_delta(
        llm, prompt_tokens, prefix_tokens=prefix_tokens, layers=CRITICAL_LAYERS
    )
    print(f"    Got {len(target)} layers")
    
    # Target delta
    target_delta = {}
    for layer in target:
        if layer in baseline:
            delta = target[layer] - baseline[layer]
            target_delta[layer] = delta
            norm = np.linalg.norm(delta)
            if layer in list(sorted(target))[:3]:
                print(f"    Layer {layer}: delta norm={norm:.4f}")
    
    print(f"  Target delta computed for {len(target_delta)} layers")
    
    # Target logits: what the model predicts with the full prefix
    print("  Capturing target logits (full prefix)...")
    target_logits = capture_target_logits(llm, prefix_tokens, prompt_tokens)
    target_top5_ids = np.argsort(target_logits)[-5:][::-1]
    target_top5_vals = target_logits[target_top5_ids]
    print(f"    Target top-5 token IDs: {target_top5_ids.tolist()}")
    for tid, tv in zip(target_top5_ids, target_top5_vals):
        tok_str = llm.detokenize([tid]).decode('utf-8', errors='replace')
        print(f"      {tid}: {tv:.2f} ({tok_str!r})")
    
    # Baseline logits (no prefix)
    print("  Capturing baseline logits (no prefix)...")
    llm._ctx.memory_clear(True)
    llm.reset()
    llm.eval(prompt_tokens_bos)
    baseline_logits_ptr = llm._ctx.get_logits()
    baseline_logits = np.ctypeslib.as_array(baseline_logits_ptr, shape=(llm._n_vocab,)).copy()
    
    # ═══════════════════════════════════════════════════════════════
    #  STEP 2: CMA-ES Optimization
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*72}")
    print(f"  STEP 2: CMA-ES Optimization ({args.max_evals} max evaluations)")
    print(f"{'═'*72}")
    
    eval_count = 0
    best_loss = float('inf')
    best_embedding = None
    loss_history = []
    
    def objective(x: np.ndarray) -> float:
        """CMA-ES objective: inject synthetic embedding as system-prompt KV,
        then measure if the model makes the same next-token predictions.
        
        The synthetic embedding populates the KV cache at position 1 (after BOS).
        All prompt tokens attend to it through the attention mechanism.
        Loss = KL divergence between target logits (full prefix) and produced logits.
        
        This is the most direct objective: does the KV cache produced by
        the synthetic token cause the same output as the full 117-token prefix?
        """
        nonlocal eval_count, best_loss, best_embedding
        eval_count += 1
        
        emb = x.reshape(n_synth, n_embd).astype(np.float32)
        
        try:
            with suppress_stdout_stderr(disable=False):
                produced_logits = capture_logits_with_embedding(
                    llm, emb, prompt_tokens
                )
            loss = compute_logit_loss(produced_logits, target_logits, top_k=500)
        except Exception as e:
            print(f"    [eval {eval_count}] Error: {e}")
            loss = 100.0  # Max penalty
        
        if loss < best_loss:
            best_loss = loss
            best_embedding = emb.copy()
            # Show what the model's top prediction is now
            top_id = int(np.argmax(produced_logits))
            top_tok = llm.detokenize([top_id]).decode('utf-8', errors='replace')
            target_top_id = int(np.argmax(target_logits))
            target_top_tok = llm.detokenize([target_top_id]).decode('utf-8', errors='replace')
            match_str = "✓" if top_id == target_top_id else "✗"
            print(f"    [eval {eval_count}] NEW BEST: KL={loss:.4f}  "
                  f"top={top_tok!r} {match_str} (target={target_top_tok!r})")
        
        if eval_count % 50 == 0:
            print(f"    [eval {eval_count}] current best={best_loss:.4f}")
        
        loss_history.append(loss)
        return loss
    
    # Random start scaled to typical embedding magnitude
    x0 = np.random.normal(0, 0.02, opt_dim).astype(np.float64)
    
    # CMA-ES options
    opts = {
        'maxfevals': args.max_evals,
        'verb_disp': 0,        # Suppress default printing
        'seed': 42,
        'tolfun': 1e-6,        # Stop if loss plateaus
        'tolx': 1e-8,          # Stop if step size collapses
    }
    
    # For very high dimensions, use separable CMA-ES
    if opt_dim > 2000:
        print(f"  Using separable CMA-ES (dim={opt_dim} > 2000)")
        opts['CMA_diagonal'] = True
    
    print(f"  Starting CMA-ES (sigma={args.sigma}, dim={opt_dim})...")
    t_start = time.time()
    
    es = cma.CMAEvolutionStrategy(x0, args.sigma, opts)
    
    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(x) for x in solutions]
        es.tell(solutions, fitnesses)
    
    t_elapsed = time.time() - t_start
    print(f"\n  CMA-ES completed: {eval_count} evaluations in {t_elapsed:.1f}s")
    print(f"  Best loss: {best_loss:.6f}")
    print(f"  Mean time per eval: {t_elapsed/max(eval_count,1)*1000:.0f}ms")
    
    if best_embedding is None:
        print("  ERROR: No valid embedding found!")
        return
    
    # ═══════════════════════════════════════════════════════════════
    #  STEP 3: Validate best embedding — activation match
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*72}")
    print(f"  STEP 3: Activation Match Validation")
    print(f"{'═'*72}")
    
    final_act = capture_activation_with_embedding(
        llm, best_embedding, prompt_tokens, CRITICAL_LAYERS
    )
    
    print(f"\n  {'Layer':>8} {'Target Δ':>12} {'Produced Δ':>12} {'Cosine':>10} {'MagRatio':>10}")
    print(f"  {'─'*8} {'─'*12} {'─'*12} {'─'*10} {'─'*10}")
    
    for layer in sorted(target_delta):
        t_delta = target_delta[layer]
        t_norm = np.linalg.norm(t_delta)
        
        if layer not in final_act or layer not in baseline:
            print(f"  {layer:>8} {t_norm:>12.4f} {'N/A':>12} {'N/A':>10} {'N/A':>10}")
            continue
        
        p_delta = final_act[layer] - baseline[layer]
        p_norm = np.linalg.norm(p_delta)
        
        cos = np.dot(t_delta, p_delta) / (t_norm * p_norm + 1e-8)
        mag_ratio = min(t_norm, p_norm) / max(t_norm, p_norm + 1e-8)
        
        print(f"  {layer:>8} {t_norm:>12.4f} {p_norm:>12.4f} {cos:>10.4f} {mag_ratio:>10.4f}")
    
    # Logit comparison
    print(f"\n  Logit Distribution Comparison:")
    with suppress_stdout_stderr(disable=False):
        final_logits = capture_logits_with_embedding(llm, best_embedding, prompt_tokens)
    final_kl = compute_logit_loss(final_logits, target_logits, top_k=500)
    baseline_kl = compute_logit_loss(baseline_logits, target_logits, top_k=500)
    print(f"    Baseline KL (no prefix vs full):   {baseline_kl:.4f}")
    print(f"    Synthetic KL (synth vs full):       {final_kl:.4f}")
    print(f"    KL reduction:                       {baseline_kl - final_kl:.4f} ({(1 - final_kl/baseline_kl)*100:.1f}%)")
    
    # Top-5 token comparison
    print(f"\n  {'Rank':>6} {'Target':>20} {'Synthetic':>20} {'No-prefix':>20}")
    print(f"  {'─'*6} {'─'*20} {'─'*20} {'─'*20}")
    def stable_softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()
    p_tgt = stable_softmax(target_logits)
    p_syn = stable_softmax(final_logits)
    p_base = stable_softmax(baseline_logits)
    for rank in range(5):
        tid = target_top5_ids[rank]
        tok_str = llm.detokenize([tid]).decode('utf-8', errors='replace')
        t_prob = p_tgt[tid]
        s_prob = p_syn[tid]
        b_prob = p_base[tid]
        print(f"  {rank+1:>6} {tok_str!r:>12} {t_prob:.4f}  {s_prob:.4f}           {b_prob:.4f}")
    
    # ═══════════════════════════════════════════════════════════════
    #  STEP 4: Detail Recall Test
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*72}")
    print(f"  STEP 4: Detail Recall Test")
    print(f"{'═'*72}")
    
    # Baseline: no prefix
    print("  [bare] No prefix...")
    bare_text = generate_fresh(llm, chat_prompt, args.max_tokens)
    bare_n, bare_res = score(bare_text, ELARA_CHECKS)
    print(f"    → {bare_n}/13")
    
    # Full prefix
    print(f"  [full] Full prefix ({len(prefix_tokens)} tokens)...")
    full_text = generate_fresh(llm, chat_prompt, args.max_tokens,
                                prefix=full_prefix)
    full_n, full_res = score(full_text, ELARA_CHECKS)
    print(f"    → {full_n}/13")
    
    # Synthetic token
    print(f"  [synthetic] {n_synth} synthetic token(s)...")
    synth_text = generate_with_embedding(llm, best_embedding, chat_prompt,
                                          args.max_tokens)
    synth_n, synth_res = score(synth_text, ELARA_CHECKS)
    print(f"    → {synth_n}/13")
    
    # Detail comparison
    print(f"\n  {'Detail':<22} {'Bare':>8} {'Full':>8} {'Synth':>8}")
    print(f"  {'─'*22} {'─'*8} {'─'*8} {'─'*8}")
    for key in ELARA_CHECKS:
        b = '✓' if bare_res[key] else '-'
        f = '✓' if full_res[key] else '-'
        s = '✓' if synth_res[key] else '-'
        print(f"  {key:<22} {b:>8} {f:>8} {s:>8}")
    print(f"  {'TOTAL':<22} {f'{bare_n}/13':>8} {f'{full_n}/13':>8} {f'{synth_n}/13':>8}")
    
    # Show generated text preview
    print(f"\n  Synthetic token output (first 300 chars):")
    print(f"  {synth_text[:300]}...")
    
    # ═══════════════════════════════════════════════════════════════
    #  STEP 5: Compression Stats
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*72}")
    print(f"  COMPRESSION STATISTICS")
    print(f"{'═'*72}")
    
    full_kv_bytes = len(prefix_tokens) * n_layer * 2 * n_embd * 2  # approx
    synth_bytes = n_synth * n_embd * 4  # float32
    
    print(f"  Full prefix:     {len(prefix_tokens)} tokens, ~{full_kv_bytes/1024:.0f} KB KV cache")
    print(f"  Synthetic token: {n_synth} token(s), {synth_bytes/1024:.1f} KB (.mem file)")
    print(f"  Compression:     {len(prefix_tokens)/n_synth:.0f}× token reduction")
    print(f"  KV savings:      {full_kv_bytes/synth_bytes:.0f}× storage reduction")
    print(f"  Detail transfer: {synth_n}/13 vs {full_n}/13 (full prefix)")
    
    # ═══════════════════════════════════════════════════════════════
    #  Save results
    # ═══════════════════════════════════════════════════════════════
    results = {
        'config': {
            'n_synthetic': n_synth,
            'n_embd': n_embd,
            'opt_dim': opt_dim,
            'sigma': args.sigma,
            'max_evals': args.max_evals,
            'critical_layers': CRITICAL_LAYERS,
        },
        'optimization': {
            'total_evals': eval_count,
            'best_loss': float(best_loss),
            'time_seconds': t_elapsed,
            'loss_history_samples': loss_history[::max(1, len(loss_history)//50)],
        },
        'recall': {
            'bare': {'score': bare_n, 'details': bare_res},
            'full_prefix': {'score': full_n, 'tokens': len(prefix_tokens),
                           'details': full_res},
            'synthetic': {'score': synth_n, 'tokens': n_synth,
                         'details': synth_res},
        },
        'compression': {
            'full_prefix_tokens': len(prefix_tokens),
            'synthetic_tokens': n_synth,
            'token_ratio': len(prefix_tokens) / n_synth,
            'full_kv_bytes': full_kv_bytes,
            'synthetic_bytes': synth_bytes,
            'storage_ratio': full_kv_bytes / synth_bytes,
        },
    }
    
    # Save embedding as .mem file
    mem_path = args.output.replace('.json', '.mem.npy')
    np.save(mem_path, best_embedding)
    print(f"\n  Synthetic embedding saved to: {mem_path}")
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {args.output}")
    
    del llm


if __name__ == "__main__":
    main()
