#!/usr/bin/env python3
"""
Per-Layer Virtual Token Distillation
=====================================

Decomposes the "compress prefix into virtual token" problem into independent
per-layer sub-problems. For each transformer layer in the critical band,
independently optimize a virtual token embedding to match the hidden-state
effect of the full prefix at THAT specific layer only.

Why this is better than whole-model optimization:
  1. No cross-layer interference — layer 14 can't fight layer 22
  2. Each sub-problem is simpler — single-layer hidden-state matching
  3. Reveals which layers are "compressible" vs which resist distillation
  4. Shows natural groupings — similar optimal embeddings can be merged

After per-layer optimization, the script:
  - Computes inter-layer embedding similarity matrix
  - Tests generation quality with each per-layer embedding
  - Clusters compatible embeddings into groups
  - Tests averaged/clustered multi-token configurations

Architecture:
  [BOS pos=0] → [virtual_emb pos=1] → [prompt tokens pos=2..N]
  The virtual token populates KV cache at ALL layers, but we optimize
  using the loss at only ONE target layer at a time.
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
from llama_cpp._utils import suppress_stdout_stderr


# ═══════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

# Full critical band from layer sensitivity experiments
CRITICAL_LAYERS = list(range(14, 23))  # 14-22 inclusive

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


def score(text: str, checks: dict) -> Tuple[int, dict]:
    results = {}
    for name, pattern in checks.items():
        results[name] = bool(re.search(pattern, text, re.IGNORECASE))
    return sum(results.values()), results


# ═══════════════════════════════════════════════════════════════════════
#  EMBEDDING INJECTION
# ═══════════════════════════════════════════════════════════════════════

def eval_embedding(llm: Llama, embedding: np.ndarray, pos_start: int = 0,
                    seq_id: int = 0) -> None:
    """Inject a synthetic embedding vector and decode it through the model."""
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)

    n_tokens, n_embd = embedding.shape
    assert n_embd == llm.n_embd(), f"Embedding dim {n_embd} != model dim {llm.n_embd()}"

    batch = llama_cpp.llama_batch_init(n_tokens, n_embd, 1)
    try:
        batch.n_tokens = n_tokens
        embd_arr = (ctypes.c_float * (n_tokens * n_embd))(*embedding.ravel().tolist())
        ctypes.memmove(batch.embd, embd_arr, n_tokens * n_embd * ctypes.sizeof(ctypes.c_float))
        for i in range(n_tokens):
            batch.pos[i] = pos_start + i
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = seq_id
            batch.logits[i] = 0
        batch.logits[n_tokens - 1] = 1
        ret = llama_cpp.llama_decode(llm._ctx.ctx, batch)
        if ret != 0:
            raise RuntimeError(f"llama_decode failed: {ret}")
        llm.n_tokens = pos_start + n_tokens
    finally:
        llama_cpp.llama_batch_free(batch)


def eval_bos(llm: Llama) -> None:
    """Decode BOS token at position 0."""
    bos = llm.token_bos()
    batch = llama_cpp.llama_batch_init(1, 0, 1)
    try:
        batch.n_tokens = 1
        batch.token[0] = bos
        batch.pos[0] = 0
        batch.n_seq_id[0] = 1
        batch.seq_id[0][0] = 0
        batch.logits[0] = 0
        ret = llama_cpp.llama_decode(llm._ctx.ctx, batch)
        if ret != 0:
            raise RuntimeError(f"BOS decode failed: {ret}")
        llm.n_tokens = 1
    finally:
        llama_cpp.llama_batch_free(batch)


def eval_prompt_tokens(llm: Llama, prompt_tokens: List[int],
                        pos_start: int) -> None:
    """Decode prompt tokens at specified starting position."""
    n = len(prompt_tokens)
    batch = llama_cpp.llama_batch_init(n, 0, 1)
    try:
        batch.n_tokens = n
        for i in range(n):
            batch.token[i] = prompt_tokens[i]
            batch.pos[i] = pos_start + i
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = 0
            batch.logits[i] = 0
        batch.logits[n - 1] = 1  # Output for last token
        ret = llama_cpp.llama_decode(llm._ctx.ctx, batch)
        if ret != 0:
            raise RuntimeError(f"decode failed: {ret}")
        llm.n_tokens = pos_start + n
    finally:
        llama_cpp.llama_batch_free(batch)


# ═══════════════════════════════════════════════════════════════════════
#  ACTIVATION CAPTURE
# ═══════════════════════════════════════════════════════════════════════

def capture_hidden_states(
    llm: Llama,
    prompt_tokens: List[int],
    prefix_tokens: Optional[List[int]],
    layers: List[int],
) -> Dict[int, np.ndarray]:
    """Capture hidden states at specified layers for the last prompt token.

    Layout with prefix:  eval(prefix_tokens) → eval(prompt_tokens)
    Layout without:      eval(prompt_tokens_with_bos)

    Returns dict mapping layer → 1D array of shape (n_embd,).
    """
    llm._ctx.memory_clear(True)
    llm.reset()
    llm.set_layer_capture(layers)

    if prefix_tokens is not None:
        llm.eval(prefix_tokens)
        llm.eval(prompt_tokens)
        last_batch_size = len(prompt_tokens)
    else:
        llm.eval(prompt_tokens)
        last_batch_size = len(prompt_tokens)

    # C-side output_ids indexes within the LAST batch
    last_pos = last_batch_size - 1
    result = llm.get_layer_embeddings(layers=layers, token_positions=[last_pos])

    llm.set_layer_capture(None)
    if result is None:
        return {}
    return {layer: arr[0] for layer, arr in result.items()}


def capture_with_virtual_token(
    llm: Llama,
    virtual_emb: np.ndarray,
    prompt_tokens: List[int],
    layers: List[int],
) -> Dict[int, np.ndarray]:
    """Inject virtual token, eval prompt, capture hidden states at layers.

    Layout: [BOS pos=0] → [virtual_emb pos=1] → [prompt_tokens pos=2..N]
    Returns dict mapping layer → 1D array of shape (n_embd,).
    """
    llm._ctx.memory_clear(True)
    llm.reset()
    llm.set_layer_capture(layers)

    # BOS
    eval_bos(llm)

    # Virtual embedding at position 1
    n_synth = virtual_emb.shape[0] if virtual_emb.ndim > 1 else 1
    eval_embedding(llm, virtual_emb, pos_start=1)
    pos_after = 1 + n_synth

    # Prompt tokens
    eval_prompt_tokens(llm, prompt_tokens, pos_start=pos_after)

    # Capture (last batch = prompt tokens)
    last_pos = len(prompt_tokens) - 1
    result = llm.get_layer_embeddings(layers=layers, token_positions=[last_pos])

    llm.set_layer_capture(None)
    if result is None:
        return {}
    return {layer: arr[0] for layer, arr in result.items()}


def capture_logits_with_virtual(
    llm: Llama,
    virtual_emb: np.ndarray,
    prompt_tokens: List[int],
) -> np.ndarray:
    """Inject virtual token, eval prompt, return logits for last prompt token.
    
    """
    llm._ctx.memory_clear(True)
    llm.reset()
    llm.set_layer_capture(None)

    eval_bos(llm)
    n_synth = virtual_emb.shape[0] if virtual_emb.ndim > 1 else 1
    eval_embedding(llm, virtual_emb, pos_start=1)
    eval_prompt_tokens(llm, prompt_tokens, pos_start=1 + n_synth)

    logits_ptr = llm._ctx.get_logits()
    return np.ctypeslib.as_array(logits_ptr, shape=(llm._n_vocab,)).copy()


# ═══════════════════════════════════════════════════════════════════════
#  LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance: 0 = identical direction, 2 = opposite."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 2.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def magnitude_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """Ratio of magnitudes: 1.0 = same, 0 = huge difference."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if max(na, nb) < 1e-8:
        return 1.0
    return float(min(na, nb) / max(na, nb))


def single_layer_loss(produced: np.ndarray, target_delta: np.ndarray,
                       baseline: np.ndarray) -> float:
    """Loss for a single layer: cosine distance of activation deltas.

    produced: hidden state at layer L with virtual token
    baseline: hidden state at layer L without prefix
    target_delta: (hidden state with full prefix) - baseline
    """
    produced_delta = produced - baseline
    cos_d = cosine_distance(produced_delta, target_delta)

    # Add small magnitude penalty
    pn = np.linalg.norm(produced_delta)
    tn = np.linalg.norm(target_delta)
    if pn > 1e-8 and tn > 1e-8:
        log_mag = abs(np.log(pn / tn))
        mag_loss = min(log_mag, 2.0)
    else:
        mag_loss = 2.0

    return 0.85 * cos_d + 0.15 * mag_loss


def kl_divergence(produced_logits: np.ndarray, target_logits: np.ndarray,
                   top_k: int = 500) -> float:
    """KL(target || produced) on top-K tokens."""
    def softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    p_t = softmax(target_logits)
    p_p = softmax(produced_logits)
    top_idx = np.argsort(p_t)[-top_k:]
    pt = np.clip(p_t[top_idx], 1e-10, 1.0)
    pp = np.clip(p_p[top_idx], 1e-10, 1.0)
    return float(np.sum(pt * np.log(pt / pp)))


# ═══════════════════════════════════════════════════════════════════════
#  GENERATION
# ═══════════════════════════════════════════════════════════════════════

def generate_with_virtual(llm: Llama, virtual_emb: np.ndarray,
                           prompt: str, max_tokens: int) -> str:
    """Generate text: [BOS] [virtual_emb] [prompt] → greedy decode."""
    llm.set_layer_capture(None)
    llm._ctx.memory_clear(True)
    llm.reset()

    eval_bos(llm)
    n_synth = virtual_emb.shape[0] if virtual_emb.ndim > 1 else 1
    eval_embedding(llm, virtual_emb, pos_start=1)
    pos_after = 1 + n_synth

    prompt_tokens = llm.tokenize(prompt.encode(), add_bos=False)
    eval_prompt_tokens(llm, prompt_tokens, pos_start=pos_after)

    output_tokens = []
    eos = llm.token_eos()
    for _ in range(max_tokens):
        logits_ptr = llm._ctx.get_logits()
        logits = np.ctypeslib.as_array(logits_ptr, shape=(llm._n_vocab,)).copy()
        tok = int(np.argmax(logits))
        if tok == eos:
            break
        output_tokens.append(tok)

        batch = llama_cpp.llama_batch_init(1, 0, 1)
        try:
            batch.n_tokens = 1
            batch.token[0] = tok
            batch.pos[0] = llm.n_tokens
            batch.n_seq_id[0] = 1
            batch.seq_id[0][0] = 0
            batch.logits[0] = 1
            ret = llama_cpp.llama_decode(llm._ctx.ctx, batch)
            if ret != 0:
                break
            llm.n_tokens += 1
        finally:
            llama_cpp.llama_batch_free(batch)

    return llm.detokenize(output_tokens).decode("utf-8", errors="replace")


def generate_fresh(llm: Llama, prompt: str, max_tokens: int,
                    prefix: Optional[str] = None) -> str:
    """Standard generation (no virtual token)."""
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
        logits_ptr = llm._ctx.get_logits()
        logits = np.ctypeslib.as_array(logits_ptr, shape=(llm._n_vocab,)).copy()
        tok = int(np.argmax(logits))
        if tok == eos:
            break
        output_tokens.append(tok)

        batch = llama_cpp.llama_batch_init(1, 0, 1)
        try:
            batch.n_tokens = 1
            batch.token[0] = tok
            batch.pos[0] = llm.n_tokens
            batch.n_seq_id[0] = 1
            batch.seq_id[0][0] = 0
            batch.logits[0] = 1
            ret = llama_cpp.llama_decode(llm._ctx.ctx, batch)
            if ret != 0:
                break
            llm.n_tokens += 1
        finally:
            llama_cpp.llama_batch_free(batch)

    return llm.detokenize(output_tokens).decode("utf-8", errors="replace")


# ═══════════════════════════════════════════════════════════════════════
#  PER-LAYER CMA-ES OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════

def optimize_single_layer(
    llm: Llama,
    target_layer: int,
    target_delta: np.ndarray,
    baseline_state: np.ndarray,
    prompt_tokens: List[int],
    n_embd: int,
    max_evals: int = 500,
    sigma: float = 0.5,
) -> Tuple[np.ndarray, float, List[float]]:
    """CMA-ES optimization for a single layer.

    Finds the virtual token embedding that, when injected at position 1,
    causes the hidden state at target_layer to best match target_delta.

    Returns: (best_embedding, best_loss, loss_history)
    """
    eval_count = 0
    best_loss = float('inf')
    best_emb = None
    history = []

    def objective(x: np.ndarray) -> float:
        nonlocal eval_count, best_loss, best_emb
        eval_count += 1

        emb = x.astype(np.float32).reshape(1, n_embd)

        try:
            with suppress_stdout_stderr(disable=False):
                states = capture_with_virtual_token(
                    llm, emb, prompt_tokens, [target_layer]
                )
            if target_layer not in states:
                return 4.0

            loss = single_layer_loss(
                states[target_layer], target_delta, baseline_state
            )
        except Exception:
            loss = 4.0

        if loss < best_loss:
            best_loss = loss
            best_emb = emb.copy()

        history.append(loss)
        return loss

    x0 = np.random.normal(0, 0.02, n_embd).astype(np.float64)

    opts = {
        'maxfevals': max_evals,
        'verb_disp': 0,
        'seed': target_layer * 100 + 42,  # Different seed per layer
        'tolfun': 1e-6,
        'tolx': 1e-8,
    }
    if n_embd > 2000:
        opts['CMA_diagonal'] = True

    es = cma.CMAEvolutionStrategy(x0, sigma, opts)
    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(x) for x in solutions]
        es.tell(solutions, fitnesses)

    return best_emb if best_emb is not None else x0.reshape(1, n_embd).astype(np.float32), best_loss, history


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Per-layer virtual token distillation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--main-gpu", type=int, default=0)
    parser.add_argument("--single-gpu", action="store_true")
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--evals-per-layer", type=int, default=500,
                        help="CMA-ES evaluations per layer")
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layers to optimize (default: 14-22)")
    parser.add_argument("--output", type=str, default="scripts/per_layer_results.json")
    args = parser.parse_args()

    extra = {}
    if args.single_gpu:
        extra["tensor_split"] = [1.0, 0.0]

    layers = CRITICAL_LAYERS
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]

    print("Loading model...")
    llm = Llama(
        model_path=args.model,
        n_gpu_layers=-1,
        n_ctx=args.n_ctx,
        main_gpu=args.main_gpu,
        verbose=False,
        embeddings=False,  # Luna C patch: layer capture works without embeddings mode
        **extra,
    )

    n_embd = llm.n_embd()
    n_layer = llm.get_n_layer()
    print(f"  n_embd={n_embd}, n_layer={n_layer}")
    print(f"  Target layers: {layers}")
    print(f"  Evals/layer: {args.evals_per_layer}")

    full_prefix = ELARA_PREFIX + KV_SEP
    chat_prompt = wrap_chat(ELARA_PROMPT)

    prefix_tokens = llm.tokenize(full_prefix.encode(), add_bos=True)
    prompt_tokens = llm.tokenize(chat_prompt.encode(), add_bos=False)
    prompt_tokens_bos = llm.tokenize(chat_prompt.encode(), add_bos=True)

    print(f"  Prefix: {len(prefix_tokens)} tokens")
    print(f"  Prompt: {len(prompt_tokens)} tokens")

    # ═══════════════════════════════════════════════════════════════
    #  STEP 1: Capture targets and baselines for all layers
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*72}")
    print(f"  STEP 1: Capture Target & Baseline Hidden States")
    print(f"{'═'*72}")

    print("  Capturing baseline (no prefix)...")
    with suppress_stdout_stderr(disable=False):
        baseline = capture_hidden_states(
            llm, prompt_tokens_bos, prefix_tokens=None, layers=layers
        )
    print(f"    Got {len(baseline)} layers")

    print("  Capturing target (full prefix)...")
    with suppress_stdout_stderr(disable=False):
        target = capture_hidden_states(
            llm, prompt_tokens, prefix_tokens=prefix_tokens, layers=layers
        )
    print(f"    Got {len(target)} layers")

    # Compute target deltas
    target_delta = {}
    for L in layers:
        if L in target and L in baseline:
            delta = target[L] - baseline[L]
            target_delta[L] = delta
            print(f"    Layer {L}: delta norm={np.linalg.norm(delta):.4f}")

    # Capture target logits
    print("  Capturing target logits...")
    with suppress_stdout_stderr(disable=False):
        llm._ctx.memory_clear(True)
        llm.reset()
        llm.set_layer_capture(None)
        llm.eval(prefix_tokens)
        llm.eval(prompt_tokens)
        logits_ptr = llm._ctx.get_logits()
        target_logits = np.ctypeslib.as_array(logits_ptr, shape=(llm._n_vocab,)).copy()

    target_top_id = int(np.argmax(target_logits))
    target_top_tok = llm.detokenize([target_top_id]).decode('utf-8', errors='replace')
    print(f"    Target top token: {target_top_tok!r} (id={target_top_id})")

    # ═══════════════════════════════════════════════════════════════
    #  STEP 2: Per-Layer CMA-ES Optimization
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*72}")
    print(f"  STEP 2: Per-Layer CMA-ES ({args.evals_per_layer} evals/layer)")
    print(f"{'═'*72}")

    per_layer_embeddings: Dict[int, np.ndarray] = {}
    per_layer_losses: Dict[int, float] = {}
    per_layer_histories: Dict[int, List[float]] = {}

    total_t_start = time.time()

    for L in layers:
        if L not in target_delta:
            print(f"\n  Layer {L}: SKIPPED (no target delta)")
            continue

        t0 = time.time()
        print(f"\n  Layer {L}: optimizing...", end="", flush=True)

        emb, loss, hist = optimize_single_layer(
            llm=llm,
            target_layer=L,
            target_delta=target_delta[L],
            baseline_state=baseline[L],
            prompt_tokens=prompt_tokens,
            n_embd=n_embd,
            max_evals=args.evals_per_layer,
            sigma=args.sigma,
        )

        dt = time.time() - t0
        per_layer_embeddings[L] = emb
        per_layer_losses[L] = loss
        per_layer_histories[L] = hist

        # Quick validation: compute actual cosine similarity
        with suppress_stdout_stderr(disable=False):
            val_states = capture_with_virtual_token(
                llm, emb, prompt_tokens, [L]
            )
        if L in val_states:
            p_delta = val_states[L] - baseline[L]
            cos = 1.0 - cosine_distance(p_delta, target_delta[L])
            mag = magnitude_ratio(p_delta, target_delta[L])
        else:
            cos, mag = 0.0, 0.0

        # Also compute logit KL for this embedding
        with suppress_stdout_stderr(disable=False):
            emb_logits = capture_logits_with_virtual(llm, emb, prompt_tokens)
        kl = kl_divergence(emb_logits, target_logits)
        top_id = int(np.argmax(emb_logits))
        top_tok = llm.detokenize([top_id]).decode('utf-8', errors='replace')
        match = "✓" if top_id == target_top_id else "✗"

        print(f"  loss={loss:.4f}  cos={cos:.4f}  mag={mag:.4f}  "
              f"KL={kl:.3f}  top={top_tok!r} {match}  ({dt:.1f}s, {len(hist)} evals)")

    total_dt = time.time() - total_t_start
    print(f"\n  Total optimization time: {total_dt:.1f}s")

    # ═══════════════════════════════════════════════════════════════
    #  STEP 3: Inter-Layer Embedding Analysis
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*72}")
    print(f"  STEP 3: Inter-Layer Embedding Similarity")
    print(f"{'═'*72}")

    opt_layers = sorted(per_layer_embeddings.keys())
    n = len(opt_layers)

    # Cosine similarity matrix
    embeds = [per_layer_embeddings[L].ravel() for L in opt_layers]
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = 1.0 - cosine_distance(embeds[i], embeds[j])

    # Print similarity matrix
    header = "       " + "".join(f"  L{L:>2}" for L in opt_layers)
    print(header)
    for i, Li in enumerate(opt_layers):
        row = f"  L{Li:>2}  "
        for j in range(n):
            val = sim_matrix[i, j]
            if i == j:
                row += "  1.00"
            else:
                row += f"  {val:.2f}"
        row += f"   loss={per_layer_losses[Li]:.4f}"
        print(row)

    # Find clusters (greedy: group layers with cosine > threshold)
    threshold = 0.5
    assigned = set()
    clusters: List[List[int]] = []
    for i, Li in enumerate(opt_layers):
        if Li in assigned:
            continue
        cluster = [Li]
        assigned.add(Li)
        for j, Lj in enumerate(opt_layers):
            if Lj in assigned:
                continue
            if sim_matrix[i, j] > threshold:
                cluster.append(Lj)
                assigned.add(Lj)
        clusters.append(cluster)

    print(f"\n  Clusters (cosine > {threshold}):")
    for ci, cluster in enumerate(clusters):
        print(f"    Cluster {ci}: layers {cluster}")

    # Compute mean embedding per cluster
    cluster_embeddings: Dict[str, np.ndarray] = {}
    for ci, cluster in enumerate(clusters):
        mean_emb = np.mean([per_layer_embeddings[L].ravel() for L in cluster], axis=0)
        mean_emb = mean_emb.reshape(1, n_embd).astype(np.float32)
        cluster_embeddings[f"cluster_{ci}"] = mean_emb

    # ═══════════════════════════════════════════════════════════════
    #  STEP 4: Generation Quality Tests
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*72}")
    print(f"  STEP 4: Generation Quality Tests")
    print(f"{'═'*72}")

    results_table = {}

    # Bare (no prefix)
    print("  [bare] No prefix...")
    with suppress_stdout_stderr(disable=False):
        bare_text = generate_fresh(llm, chat_prompt, args.max_tokens)
    bare_n, bare_res = score(bare_text, ELARA_CHECKS)
    results_table["bare"] = {"score": bare_n, "details": bare_res}
    print(f"    → {bare_n}/13")

    # Full prefix
    print(f"  [full] Full prefix ({len(prefix_tokens)} tokens)...")
    with suppress_stdout_stderr(disable=False):
        full_text = generate_fresh(llm, chat_prompt, args.max_tokens,
                                    prefix=full_prefix)
    full_n, full_res = score(full_text, ELARA_CHECKS)
    results_table["full_prefix"] = {"score": full_n, "details": full_res}
    print(f"    → {full_n}/13")

    # Per-layer embeddings (test top 3 by lowest loss)
    sorted_by_loss = sorted(per_layer_losses.items(), key=lambda x: x[1])
    test_layers = [L for L, _ in sorted_by_loss[:5]]

    for L in test_layers:
        print(f"  [layer_{L}] Virtual token optimized for layer {L}...")
        with suppress_stdout_stderr(disable=False):
            text = generate_with_virtual(
                llm, per_layer_embeddings[L], chat_prompt, args.max_tokens
            )
        n_score, res = score(text, ELARA_CHECKS)
        results_table[f"layer_{L}"] = {
            "score": n_score, "details": res,
            "loss": per_layer_losses[L],
            "text_preview": text[:200],
        }
        print(f"    → {n_score}/13  (loss={per_layer_losses[L]:.4f})")

    # Mean of ALL per-layer embeddings
    mean_all_emb = np.mean(
        [per_layer_embeddings[L].ravel() for L in opt_layers], axis=0
    ).reshape(1, n_embd).astype(np.float32)

    print(f"  [mean_all] Mean of all {len(opt_layers)} per-layer embeddings...")
    with suppress_stdout_stderr(disable=False):
        text = generate_with_virtual(llm, mean_all_emb, chat_prompt, args.max_tokens)
    n_score, res = score(text, ELARA_CHECKS)
    results_table["mean_all"] = {"score": n_score, "details": res}
    print(f"    → {n_score}/13")

    # Cluster averages
    for name, emb in cluster_embeddings.items():
        print(f"  [{name}] Cluster average embedding...")
        with suppress_stdout_stderr(disable=False):
            text = generate_with_virtual(llm, emb, chat_prompt, args.max_tokens)
        n_score, res = score(text, ELARA_CHECKS)
        results_table[name] = {"score": n_score, "details": res}
        print(f"    → {n_score}/13")

    # Best per-layer by logit KL
    print(f"\n  [best_kl] Testing layer with best KL divergence...")
    kl_scores = {}
    for L in opt_layers:
        with suppress_stdout_stderr(disable=False):
            logits = capture_logits_with_virtual(llm, per_layer_embeddings[L], prompt_tokens)
        kl_scores[L] = kl_divergence(logits, target_logits)
    best_kl_layer = min(kl_scores, key=kl_scores.get)
    best_kl_val = kl_scores[best_kl_layer]
    print(f"    Best KL: layer {best_kl_layer} (KL={best_kl_val:.4f})")
    with suppress_stdout_stderr(disable=False):
        text = generate_with_virtual(
            llm, per_layer_embeddings[best_kl_layer], chat_prompt, args.max_tokens
        )
    n_score, res = score(text, ELARA_CHECKS)
    results_table["best_kl"] = {
        "score": n_score, "details": res,
        "layer": best_kl_layer, "kl": best_kl_val,
    }
    print(f"    → {n_score}/13  (layer {best_kl_layer})")

    # ═══════════════════════════════════════════════════════════════
    #  SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*72}")
    print(f"  SUMMARY")
    print(f"{'═'*72}")

    # Detail recall table
    all_checks = list(ELARA_CHECKS.keys())
    conditions = list(results_table.keys())

    header = f"  {'Detail':<22}"
    for cond in conditions:
        header += f" {cond:>8}"
    print(header)
    print("  " + "─" * (22 + 9 * len(conditions)))

    for check in all_checks:
        row = f"  {check:<22}"
        for cond in conditions:
            val = results_table[cond]["details"].get(check, False)
            row += f" {'✓':>8}" if val else f" {'-':>8}"
        print(row)

    row_total = f"  {'TOTAL':<22}"
    for cond in conditions:
        row_total += f" {results_table[cond]['score']:>5}/13"
    print(row_total)

    # Per-layer KL summary
    print(f"\n  Per-Layer KL Divergence:")
    print(f"  {'Layer':>8} {'Loss':>10} {'KL':>10} {'Status':>10}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10}")
    for L in opt_layers:
        status = "★ BEST" if L == best_kl_layer else ""
        print(f"  {L:>8} {per_layer_losses[L]:>10.4f} {kl_scores[L]:>10.4f} {status:>10}")

    # Similarity matrix summary
    print(f"\n  Embedding similarity extremes:")
    max_sim, max_pair = -1, (0, 0)
    min_sim, min_pair = 2, (0, 0)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] > max_sim:
                max_sim = sim_matrix[i, j]
                max_pair = (opt_layers[i], opt_layers[j])
            if sim_matrix[i, j] < min_sim:
                min_sim = sim_matrix[i, j]
                min_pair = (opt_layers[i], opt_layers[j])
    print(f"    Most similar:    layers {max_pair[0]},{max_pair[1]} cos={max_sim:.4f}")
    print(f"    Most different:  layers {min_pair[0]},{min_pair[1]} cos={min_sim:.4f}")

    # ═══════════════════════════════════════════════════════════════
    #  SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════

    # Save per-layer embeddings
    emb_path = args.output.replace(".json", "_embeddings.npz")
    np.savez(
        emb_path,
        **{f"layer_{L}": per_layer_embeddings[L] for L in opt_layers},
        mean_all=mean_all_emb,
        **{name: emb for name, emb in cluster_embeddings.items()},
    )
    print(f"\n  Embeddings saved to: {emb_path}")

    # Save JSON results
    json_data = {
        "config": {
            "layers": layers,
            "evals_per_layer": args.evals_per_layer,
            "sigma": args.sigma,
            "n_embd": n_embd,
            "prefix_tokens": len(prefix_tokens),
            "prompt_tokens": len(prompt_tokens),
        },
        "per_layer": {
            str(L): {
                "loss": per_layer_losses[L],
                "kl": kl_scores[L],
                "n_evals": len(per_layer_histories[L]),
            }
            for L in opt_layers
        },
        "similarity_matrix": {
            f"{opt_layers[i]},{opt_layers[j]}": float(sim_matrix[i, j])
            for i in range(n) for j in range(i + 1, n)
        },
        "clusters": {str(ci): cluster for ci, cluster in enumerate(clusters)},
        "generation_scores": {
            cond: results_table[cond]["score"]
            for cond in results_table
        },
        "best_kl_layer": best_kl_layer,
        "total_time_s": total_dt,
    }

    # Convert numpy types for JSON serialization
    def to_json_safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            val = to_json_safe(obj)
            if val is not obj:
                return val
            return super().default(obj)

    with open(args.output, "w") as f:
        json.dump(json_data, f, indent=2, cls=NumpyEncoder)
    print(f"  Results saved to: {args.output}")

    llm.set_layer_capture(None)
    del llm


if __name__ == "__main__":
    main()
