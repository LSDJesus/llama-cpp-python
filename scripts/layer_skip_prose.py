#!/usr/bin/env python
"""Layer skip prose comparison.

Skip each transformer layer individually and save the full generated text output
plus penultimate hidden-layer embeddings for qualitative/semantic comparison.

No token-match metrics — this is about reading the actual outputs and judging
coherence, factual accuracy, and writing quality.

Usage:
    python scripts/layer_skip_prose.py \
        --model path/to/model.gguf \
        --max-tokens 256 \
        --output scripts/prose_results.json
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from llama_cpp import Llama, llama_cpp


# ─── Default prompt ─────────────────────────────────────────────────────────────
# Tests: creative writing, style adherence, factual grounding, sustained coherence
DEFAULT_PROMPT = (
    "Write three paragraphs describing the maiden voyage of the RMS Titanic "
    "in the style of Southern Gothic literature. Include these specific facts: "
    "the ship departed Southampton on April 10, 1912; it struck an iceberg at "
    "11:40 PM on April 14; the water temperature was 28°F (−2°C); and 1,517 "
    "people perished. Weave these details naturally into atmospheric, literary prose."
)


# ─── Helpers ─────────────────────────────────────────────────────────────────────

def generate_greedy(llm: Llama, prompt: str, max_tokens: int) -> Tuple[str, List[int], float]:
    """Generate text with temp=0 greedy decoding.

    Returns (text, token_ids, elapsed_seconds).
    """
    t0 = time.perf_counter()
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        echo=False,
    )
    elapsed = time.perf_counter() - t0
    text: str = output["choices"][0]["text"]
    token_ids: List[int] = llm.tokenize(text.encode("utf-8"), add_bos=False)
    return text, token_ids, elapsed


def grab_penultimate(llm: Llama) -> Optional[npt.NDArray[np.float32]]:
    """Read the penultimate-layer embedding for the last decoded token (batch position 0).

    Works during normal generation — the C++ graph builder stores this
    independently of the embeddings=True context flag.
    """
    try:
        ptr = llama_cpp.llama_get_embeddings_penultimate_ith(
            llm._ctx.ctx, ctypes.c_int32(0)
        )
        if ptr and bool(ptr):
            n_embd = llm.n_embd()
            return np.ctypeslib.as_array(ptr, shape=(n_embd,)).copy()
    except Exception:
        pass
    return None


def cosine_sim(a: npt.NDArray, b: npt.NDArray) -> float:
    """Cosine similarity between two vectors."""
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


def emb_to_list(emb: Optional[npt.NDArray]) -> Optional[List[float]]:
    """Convert embedding to JSON-serialisable list (or None)."""
    if emb is None:
        return None
    return [round(float(v), 6) for v in emb]


# ─── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Layer-skip prose comparison")
    ap.add_argument("--model", required=True, help="Path to GGUF model")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="Generation prompt")
    ap.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    ap.add_argument("--n-ctx", type=int, default=2048, help="Context window size")
    ap.add_argument("--layers", type=str, default=None,
                    help="Comma-separated layer indices to test (default: all)")
    ap.add_argument("--main-gpu", type=int, default=0, help="Main GPU index")
    ap.add_argument("--single-gpu", action="store_true",
                    help="Force all layers onto main GPU (tensor_split=[1,0])")
    ap.add_argument("--output", default="scripts/prose_results.json",
                    help="Output JSON path")
    ap.add_argument("--save-embeddings", action="store_true",
                    help="Include raw penultimate embeddings in JSON (large!)")
    args = ap.parse_args()

    # ── Load model ───────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    extra_kwargs = {}
    if args.single_gpu:
        extra_kwargs["tensor_split"] = [1.0, 0.0]
    llm = Llama(
        model_path=args.model,
        n_gpu_layers=-1,
        n_ctx=args.n_ctx,
        main_gpu=args.main_gpu,
        verbose=False,
        **extra_kwargs,
    )

    n_layer = llm.get_n_layer()
    n_embd = llm.n_embd()

    # ── Layer list ───────────────────────────────────────────────────
    if args.layers is not None:
        test_layers = sorted(int(x.strip()) for x in args.layers.split(","))
    else:
        # Skip first 2 and last 2 layers -- these are always catastrophic
        # (embedding/unembedding adjacent layers crash or produce garbage)
        safe_start = 2
        safe_end = n_layer - 2
        test_layers = list(range(safe_start, safe_end))

    total_runs = 1 + len(test_layers)  # baseline + each skip

    print(f"  Layers: {n_layer}  |  Embedding dim: {n_embd}")
    print(f"  Runs: {total_runs} (1 baseline + {len(test_layers)} skips)")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Prompt ({len(args.prompt)} chars): {args.prompt[:100]}...")
    print()

    # ── Baseline ─────────────────────────────────────────────────────
    print(f"{'='*72}")
    print("  BASELINE (no layers skipped)")
    print(f"{'='*72}")

    llm.set_layer_skip(None)
    base_text, base_ids, base_time = generate_greedy(llm, args.prompt, args.max_tokens)
    base_emb = grab_penultimate(llm)

    print(f"  {len(base_ids)} tokens in {base_time:.1f}s")
    print(f"  ──────────")
    # Print full baseline text wrapped
    for line in base_text.split("\n"):
        print(f"  {line}")
    print()

    baseline_entry = {
        "text": base_text,
        "n_tokens": len(base_ids),
        "time_s": round(base_time, 2),
    }
    if args.save_embeddings:
        baseline_entry["penultimate_embedding"] = emb_to_list(base_emb)

    # ── Per-layer skips ──────────────────────────────────────────────
    layer_results: List[Dict] = []
    all_embeddings: List[Tuple[int, Optional[npt.NDArray]]] = []
    # Store baseline embedding for similarity calc
    all_embeddings.append((-1, base_emb))  # -1 = baseline sentinel

    def save_incremental():
        """Save current results to disk (crash protection)."""
        results = {
            "model": args.model,
            "n_layer": n_layer,
            "n_embd": n_embd,
            "max_tokens": args.max_tokens,
            "prompt": args.prompt,
            "baseline": baseline_entry,
            "layer_skips": layer_results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # Save baseline immediately
    save_incremental()

    for idx, layer in enumerate(test_layers):
        pct = (idx + 1) / len(test_layers) * 100
        print(f"{'='*72}")
        print(f"  LAYER {layer} SKIPPED  ({idx+1}/{len(test_layers)}, {pct:.0f}%)")
        print(f"{'='*72}")

        try:
            llm.set_layer_skip([layer])
            text, ids, elapsed = generate_greedy(llm, args.prompt, args.max_tokens)
            emb = grab_penultimate(llm)
        except Exception as e:
            print(f"  *** CRASHED: {e}")
            entry = {
                "layer": layer,
                "text": f"[CRASH: {e}]",
                "n_tokens": 0,
                "time_s": 0.0,
                "cosine_sim_vs_baseline": None,
            }
            layer_results.append(entry)
            save_incremental()
            llm.set_layer_skip(None)
            continue

        # Cosine similarity vs baseline
        cos = None
        if emb is not None and base_emb is not None:
            cos = round(cosine_sim(emb, base_emb), 6)

        entry = {
            "layer": layer,
            "text": text,
            "n_tokens": len(ids),
            "time_s": round(elapsed, 2),
            "cosine_sim_vs_baseline": cos,
        }
        if args.save_embeddings:
            entry["penultimate_embedding"] = emb_to_list(emb)

        layer_results.append(entry)
        all_embeddings.append((layer, emb))

        sim_str = f"cos_sim={cos:.4f}" if cos is not None else "no embedding"
        print(f"  {len(ids)} tokens in {elapsed:.1f}s  |  {sim_str}")
        print(f"  ──────────")
        for line in text.split("\n"):
            print(f"  {line}")
        print()

        # Save after every layer (crash protection)
        save_incremental()

    llm.set_layer_skip(None)

    # ── Pairwise cosine similarity matrix (compact) ──────────────────
    # Compute all-vs-all for the embeddings we have
    valid_embs = [(lbl, e) for lbl, e in all_embeddings if e is not None]
    similarity_matrix: Optional[Dict] = None
    if len(valid_embs) >= 2:
        labels = ["baseline" if lbl == -1 else f"skip_{lbl}" for lbl, _ in valid_embs]
        vecs = np.stack([e for _, e in valid_embs])
        # Normalise
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normed = vecs / norms
        sim = normed @ normed.T  # [N, N]

        similarity_matrix = {
            "labels": labels,
            "matrix": [[round(float(sim[i, j]), 6) for j in range(len(labels))]
                       for i in range(len(labels))],
        }

    # ── Save ──────────────────────────────────────────────────────────
    results = {
        "model": args.model,
        "n_layer": n_layer,
        "n_embd": n_embd,
        "max_tokens": args.max_tokens,
        "prompt": args.prompt,
        "baseline": baseline_entry,
        "layer_skips": layer_results,
    }
    if similarity_matrix:
        results["penultimate_cosine_similarity"] = similarity_matrix

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Layer':>6} | {'Tokens':>6} | {'CosSim':>8} | {'Time':>6} | First 80 chars")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*60}")
    print(f"  {'base':>6} | {baseline_entry['n_tokens']:>6} | {'1.0000':>8} | "
          f"{baseline_entry['time_s']:>5.1f}s | {base_text[:60]}")

    for entry in layer_results:
        cs = f"{entry['cosine_sim_vs_baseline']:.4f}" if entry['cosine_sim_vs_baseline'] is not None else "   N/A "
        print(f"  {entry['layer']:>6} | {entry['n_tokens']:>6} | {cs:>8} | "
              f"{entry['time_s']:>5.1f}s | {entry['text'][:60]}")

    print(f"\n  Results saved to: {args.output}")
    print(f"  Total runs: {total_runs}")


if __name__ == "__main__":
    main()
