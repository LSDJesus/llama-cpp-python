#!/usr/bin/env python3
"""
Layer Redundancy Analysis for Qwen3 Models
==========================================

Uses the [Luna] layer capture/skip infrastructure to:
  1. Extract hidden states from every transformer layer
  2. Build a cosine similarity heatmap (which layers produce near-identical outputs?)
  3. Test layer skipping and measure quality degradation
  4. Find the maximum safe pruning set

Requirements:
  - A Qwen3 or Qwen3-VL GGUF model
  - llama-cpp-python built with the [Luna] extensions
  - embeddings=True, pooling_type=NONE

Usage:
  python scripts/layer_analysis.py --model "D:/AI/SD Models/text_encoders/Qwen3-4B.i1-Q4_K_M.gguf"
  python scripts/layer_analysis.py --model "path/to/model.gguf" --phase all
  python scripts/layer_analysis.py --model "path/to/model.gguf" --phase similarity
  python scripts/layer_analysis.py --model "path/to/model.gguf" --phase skip --skip-layers 15,16,17
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent to path so we can import llama_cpp
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llama_cpp import Llama, llama_cpp


# ─── Test prompts covering different reasoning patterns ─────────────────────
TEST_PROMPTS = [
    "The capital of France is",
    "In quantum mechanics, the uncertainty principle states that",
    "To solve a quadratic equation ax² + bx + c = 0, you can use the formula",
    "The process of photosynthesis converts sunlight into",
    "Once upon a time, in a kingdom far away, there lived a",
    "The key differences between TCP and UDP are",
    "A recursive function must have a base case to",
    "The mitochondria is often called the powerhouse of the cell because",
]


def load_model(model_path: str, n_gpu_layers: int = -1, n_ctx: int = 512,
               embeddings: bool = True, main_gpu: int = 0,
               split_mode: Optional[int] = None) -> Llama:
    """Load a model. Set embeddings=True for layer capture, False for generation."""
    print(f"Loading model: {model_path}")
    print(f"  n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx}, embeddings={embeddings}, main_gpu={main_gpu}")

    kwargs: Dict = dict(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        main_gpu=main_gpu,
        verbose=False,
    )
    if split_mode is not None:
        kwargs["split_mode"] = split_mode
    else:
        kwargs["split_mode"] = llama_cpp.LLAMA_SPLIT_MODE_NONE  # default: pin to single GPU
    if embeddings:
        kwargs["embeddings"] = True
        kwargs["pooling_type"] = llama_cpp.LLAMA_POOLING_TYPE_NONE

    llm = Llama(**kwargs)

    n_layer = llm.get_n_layer()
    n_embd = llm.n_embd()
    print(f"  Model loaded: {n_layer} layers, {n_embd} dims")
    return llm


def tokenize_prompt(llm: Llama, prompt: str) -> List[int]:
    """Tokenize a prompt and return token IDs."""
    return llm.tokenize(prompt.encode("utf-8"), add_bos=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_sim_matrix(embeddings: Dict[int, np.ndarray]) -> np.ndarray:
    """Compute pairwise cosine similarity matrix across layers.

    Args:
        embeddings: Dict mapping layer_idx -> np.ndarray of shape [n_tokens, n_embd]

    Returns:
        np.ndarray of shape [n_layers, n_layers] with cosine similarities
        (averaged across tokens, using the last token representation)
    """
    layers = sorted(embeddings.keys())
    n = len(layers)
    sim = np.zeros((n, n))

    # Use last token embedding for similarity (most information-rich)
    vecs = {il: embeddings[il][-1] for il in layers}

    for i, li in enumerate(layers):
        for j, lj in enumerate(layers):
            sim[i, j] = cosine_similarity(vecs[li], vecs[lj])

    return sim


# ─── Phase 1: Layer Similarity Analysis ─────────────────────────────────────

def phase_similarity(llm: Llama, prompts: List[str]) -> Dict:
    """Capture all layer hidden states and compute similarity heatmap."""
    n_layer = llm.get_n_layer()
    n_embd = llm.n_embd()
    all_layers = list(range(n_layer))

    print(f"\n{'='*60}")
    print(f"Phase 1: Layer Similarity Analysis")
    print(f"{'='*60}")
    print(f"  Capturing {n_layer} layers across {len(prompts)} prompts...")

    # Enable capture on all layers
    llm.set_layer_capture(all_layers)

    # Accumulate per-layer similarities across prompts
    consecutive_sims = np.zeros(n_layer - 1)  # sim between layer i and i+1
    all_sim_matrices = []
    prompt_count = 0

    for pi, prompt in enumerate(prompts):
        tokens = tokenize_prompt(llm, prompt)
        n_tokens = len(tokens)

        print(f"  [{pi+1}/{len(prompts)}] \"{prompt[:50]}...\" ({n_tokens} tokens)")

        # Reset context and evaluate
        llm.reset()
        llm.eval(tokens)

        # Extract all layer embeddings
        layer_data = llm.get_layer_embeddings(layers=all_layers)

        if layer_data is None:
            print(f"    WARNING: No layer data returned, skipping")
            continue

        # Compute similarity matrix for this prompt
        sim_matrix = cosine_sim_matrix(layer_data)
        all_sim_matrices.append(sim_matrix)

        # Track consecutive layer similarities
        for il in range(n_layer - 1):
            if il in layer_data and (il + 1) in layer_data:
                last_tok_a = layer_data[il][-1]
                last_tok_b = layer_data[il + 1][-1]
                consecutive_sims[il] += cosine_similarity(last_tok_a, last_tok_b)

        prompt_count += 1

    # Disable layer capture
    llm.set_layer_capture(None)

    if prompt_count == 0:
        print("  ERROR: No prompts produced layer data!")
        return {}

    # Average across prompts
    consecutive_sims /= prompt_count
    avg_sim_matrix = np.mean(all_sim_matrices, axis=0)

    # ─── Report ─────────────────────────────────────────────────────
    print(f"\n  Consecutive Layer Cosine Similarities (layer i → i+1):")
    print(f"  {'Layer':>6}  {'Similarity':>10}  {'Bar'}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*40}")

    redundant_pairs = []
    for il in range(n_layer - 1):
        sim = consecutive_sims[il]
        bar_len = int(sim * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        marker = " ← REDUNDANT" if sim > 0.99 else (" ← high" if sim > 0.95 else "")
        print(f"  {il:>3}→{il+1:<3} {sim:>10.6f}  {bar}{marker}")

        if sim > 0.95:
            redundant_pairs.append((il, il + 1, float(sim)))

    # Find most skippable layers (the SECOND layer in each high-similarity pair)
    skip_candidates = sorted(
        [(il + 1, float(consecutive_sims[il])) for il in range(n_layer - 1) if consecutive_sims[il] > 0.95],
        key=lambda x: -x[1],
    )

    print(f"\n  Skip Candidates (layers whose output ≈ their predecessor):")
    if skip_candidates:
        for layer_idx, sim in skip_candidates:
            print(f"    Layer {layer_idx:3d}: cos_sim with layer {layer_idx-1} = {sim:.6f}")
    else:
        print(f"    None found with threshold > 0.95")
        # Lower threshold report
        mild_candidates = sorted(
            [(il + 1, float(consecutive_sims[il])) for il in range(n_layer - 1) if consecutive_sims[il] > 0.90],
            key=lambda x: -x[1],
        )
        if mild_candidates:
            print(f"    With relaxed threshold > 0.90:")
            for layer_idx, sim in mild_candidates:
                print(f"      Layer {layer_idx:3d}: cos_sim = {sim:.6f}")

    results = {
        "n_layer": n_layer,
        "n_embd": n_embd,
        "n_prompts": prompt_count,
        "consecutive_similarities": consecutive_sims.tolist(),
        "skip_candidates": skip_candidates,
        "redundant_pairs": redundant_pairs,
        "avg_similarity_matrix": avg_sim_matrix.tolist(),
    }

    return results


# ─── Phase 2: Layer Skip Quality Testing ────────────────────────────────────

def phase_skip(
    llm: Llama,
    prompts: List[str],
    skip_layers: Optional[List[int]] = None,
    similarity_results: Optional[Dict] = None,
) -> Dict:
    """Test the quality impact of skipping specific layers.

    If skip_layers is not specified, uses the top candidates from phase 1.
    """
    n_layer = llm.get_n_layer()

    print(f"\n{'='*60}")
    print(f"Phase 2: Layer Skip Quality Testing")
    print(f"{'='*60}")

    # Determine which layers to test skipping
    if skip_layers is None:
        if similarity_results and similarity_results.get("skip_candidates"):
            # Take top 5 candidates from similarity analysis
            skip_layers = [c[0] for c in similarity_results["skip_candidates"][:5]]
        else:
            # Default: test middle layers (more likely redundant than early/late)
            third = n_layer // 3
            skip_layers = list(range(third, 2 * third))

    print(f"  Testing skip candidates: {skip_layers}")

    # ─── Baseline: full model output ────────────────────────────────
    print(f"\n  Collecting baseline (no skips)...")
    llm.set_layer_skip(None)

    baseline_outputs = {}
    for pi, prompt in enumerate(prompts):
        tokens = tokenize_prompt(llm, prompt)
        llm.reset()
        llm.eval(tokens)

        # Get penultimate embeddings as quality reference
        pen_emb = llm.get_penultimate_embeddings(token_positions=[-1])
        if pen_emb is not None:
            baseline_outputs[pi] = pen_emb[0]  # last token embedding

    if not baseline_outputs:
        print("  ERROR: Could not get baseline embeddings")
        return {}

    # ─── Test each layer individually ───────────────────────────────
    print(f"\n  Testing individual layer skips...")
    individual_results = []

    for skip_il in skip_layers:
        if skip_il < 0 or skip_il >= n_layer:
            continue

        llm.set_layer_skip([skip_il])

        total_sim = 0.0
        count = 0

        for pi, prompt in enumerate(prompts):
            if pi not in baseline_outputs:
                continue

            tokens = tokenize_prompt(llm, prompt)
            llm.reset()
            llm.eval(tokens)

            pen_emb = llm.get_penultimate_embeddings(token_positions=[-1])
            if pen_emb is not None:
                sim = cosine_similarity(pen_emb[0], baseline_outputs[pi])
                total_sim += sim
                count += 1

        avg_sim = total_sim / count if count > 0 else 0.0
        degradation = 1.0 - avg_sim
        individual_results.append({
            "layer": skip_il,
            "avg_cosine_sim": avg_sim,
            "degradation": degradation,
        })
        status = "SAFE" if avg_sim > 0.999 else ("MILD" if avg_sim > 0.99 else "RISKY")
        print(f"    Skip layer {skip_il:3d}: baseline_sim={avg_sim:.6f}  degradation={degradation:.6f}  [{status}]")

    llm.set_layer_skip(None)

    # Sort by safety (highest similarity = safest to skip)
    individual_results.sort(key=lambda x: -x["avg_cosine_sim"])

    safe_skips = [r["layer"] for r in individual_results if r["avg_cosine_sim"] > 0.999]
    mild_skips = [r["layer"] for r in individual_results if 0.99 < r["avg_cosine_sim"] <= 0.999]
    risky_skips = [r["layer"] for r in individual_results if r["avg_cosine_sim"] <= 0.99]

    print(f"\n  Summary:")
    print(f"    Safe to skip  (>0.999 sim): {safe_skips or 'none'}")
    print(f"    Mild impact   (>0.99  sim): {mild_skips or 'none'}")
    print(f"    Risky         (≤0.99  sim): {risky_skips or 'none'}")

    results = {
        "skip_candidates_tested": skip_layers,
        "individual_results": individual_results,
        "safe_skips": safe_skips,
        "mild_skips": mild_skips,
        "risky_skips": risky_skips,
    }

    return results


# ─── Phase 3: Progressive Pruning ───────────────────────────────────────────

def phase_progressive(
    llm: Llama,
    prompts: List[str],
    skip_results: Optional[Dict] = None,
    threshold: float = 0.995,
) -> Dict:
    """Progressively add more layer skips until quality drops below threshold.

    Starts with the safest layers and adds one at a time.
    """
    n_layer = llm.get_n_layer()

    print(f"\n{'='*60}")
    print(f"Phase 3: Progressive Pruning (threshold={threshold})")
    print(f"{'='*60}")

    # Build candidate order: safe first, then mild
    if skip_results:
        candidates = [r["layer"] for r in skip_results.get("individual_results", [])]
    else:
        # Default: middle layers
        third = n_layer // 3
        candidates = list(range(third, 2 * third))

    if not candidates:
        print("  No candidates to test")
        return {}

    print(f"  Candidate order (safest first): {candidates}")

    # ─── Baseline ───────────────────────────────────────────────────
    print(f"  Collecting baseline...")
    llm.set_layer_skip(None)

    baseline_outputs = {}
    for pi, prompt in enumerate(prompts):
        tokens = tokenize_prompt(llm, prompt)
        llm.reset()
        llm.eval(tokens)
        pen_emb = llm.get_penultimate_embeddings(token_positions=[-1])
        if pen_emb is not None:
            baseline_outputs[pi] = pen_emb[0]

    # ─── Progressive addition ───────────────────────────────────────
    current_skips: List[int] = []
    pruning_log = []
    max_safe_set: List[int] = []

    for candidate in candidates:
        test_skips = current_skips + [candidate]
        llm.set_layer_skip(test_skips)

        total_sim = 0.0
        count = 0
        for pi, prompt in enumerate(prompts):
            if pi not in baseline_outputs:
                continue
            tokens = tokenize_prompt(llm, prompt)
            llm.reset()
            llm.eval(tokens)
            pen_emb = llm.get_penultimate_embeddings(token_positions=[-1])
            if pen_emb is not None:
                sim = cosine_similarity(pen_emb[0], baseline_outputs[pi])
                total_sim += sim
                count += 1

        avg_sim = total_sim / count if count > 0 else 0.0
        passed = avg_sim >= threshold

        entry = {
            "skipped_layers": list(test_skips),
            "n_skipped": len(test_skips),
            "n_active": n_layer - len(test_skips),
            "avg_cosine_sim": avg_sim,
            "passed": passed,
        }
        pruning_log.append(entry)

        status = "PASS" if passed else "FAIL"
        pct = len(test_skips) / n_layer * 100
        print(f"    Skip {len(test_skips):2d} layers ({pct:5.1f}%): sim={avg_sim:.6f} [{status}]  skipping={test_skips}")

        if passed:
            current_skips = test_skips
            max_safe_set = list(test_skips)
        else:
            # This candidate pushed us over — skip it and try next
            pass

    llm.set_layer_skip(None)

    reduction_pct = len(max_safe_set) / n_layer * 100

    print(f"\n  Maximum safe pruning set ({len(max_safe_set)}/{n_layer} layers, {reduction_pct:.1f}% reduction):")
    print(f"    Layers to skip: {max_safe_set}")
    print(f"    Active layers:  {[i for i in range(n_layer) if i not in max_safe_set]}")

    results = {
        "threshold": threshold,
        "max_safe_skips": max_safe_set,
        "n_skipped": len(max_safe_set),
        "n_active": n_layer - len(max_safe_set),
        "reduction_pct": reduction_pct,
        "pruning_log": pruning_log,
    }

    return results


# ─── Phase 4: Generation Quality Test ──────────────────────────────────────

GENERATION_PROMPTS = [
    "The capital of France is",
    "To solve x² - 5x + 6 = 0, we factor it as",
    "The three primary colors of light are",
    "Water boils at a temperature of",
    "The largest planet in our solar system is",
    "In Python, a list comprehension is",
    "The speed of light in a vacuum is approximately",
    "Photosynthesis is the process by which plants",
]


def generate_greedy(llm: Llama, prompt: str, max_tokens: int = 64) -> Tuple[str, List[int]]:
    """Generate text with temp=0 greedy decoding. Returns (text, token_ids)."""
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        echo=False,
    )
    text = output["choices"][0]["text"]
    # Tokenize the output to get token IDs for comparison
    token_ids = llm.tokenize(text.encode("utf-8"), add_bos=False)
    return text, token_ids


def token_match_rate(baseline_ids: List[int], test_ids: List[int]) -> float:
    """Compute fraction of tokens that match between two sequences."""
    if not baseline_ids and not test_ids:
        return 1.0
    if not baseline_ids or not test_ids:
        return 0.0
    min_len = min(len(baseline_ids), len(test_ids))
    if min_len == 0:
        return 0.0
    matches = sum(1 for a, b in zip(baseline_ids[:min_len], test_ids[:min_len]) if a == b)
    # Penalize length differences
    max_len = max(len(baseline_ids), len(test_ids))
    return matches / max_len


def phase_generation(
    llm: Llama,
    skip_layers: Optional[List[int]] = None,
    similarity_results: Optional[Dict] = None,
    max_tokens: int = 64,
) -> Dict:
    """Compare actual generated text with and without layer skips.

    Uses temp=0 greedy decoding so results are deterministic.
    Tests individual layers AND progressive combinations.
    """
    n_layer = llm.get_n_layer()

    print(f"\n{'='*60}")
    print(f"Phase 4: Generation Quality Test (temp=0, max_tokens={max_tokens})")
    print(f"{'='*60}")

    # Determine which layers to test
    if skip_layers is None:
        if similarity_results and similarity_results.get("skip_candidates"):
            skip_layers = [c[0] for c in similarity_results["skip_candidates"]]
        else:
            third = n_layer // 3
            skip_layers = list(range(third, 2 * third))

    # ─── Baseline: full model generation ────────────────────────────
    print(f"\n  Generating baselines (no skips)...")
    llm.set_layer_skip(None)

    baselines: Dict[int, Tuple[str, List[int]]] = {}
    for pi, prompt in enumerate(GENERATION_PROMPTS):
        text, ids = generate_greedy(llm, prompt, max_tokens)
        baselines[pi] = (text, ids)
        preview = text.replace("\n", "\\n")[:60]
        print(f"    [{pi}] \"{prompt[:40]}...\" → \"{preview}...\"")

    # ─── Test individual layer skips ────────────────────────────────
    print(f"\n  Testing individual layer skips on generation...")
    individual_results = []

    for skip_il in skip_layers:
        if skip_il < 0 or skip_il >= n_layer:
            continue

        llm.set_layer_skip([skip_il])

        total_match = 0.0
        comparisons = []

        for pi, prompt in enumerate(GENERATION_PROMPTS):
            text, ids = generate_greedy(llm, prompt, max_tokens)
            base_text, base_ids = baselines[pi]

            match = token_match_rate(base_ids, ids)
            total_match += match
            exact = text == base_text

            # Measure error compounding: compare first quarter vs last quarter
            q_len = max(1, min(len(base_ids), len(ids)) // 4)
            early_match = token_match_rate(base_ids[:q_len], ids[:q_len]) if q_len > 0 else 0.0
            late_match = token_match_rate(base_ids[-q_len:], ids[-q_len:]) if q_len > 0 else 0.0

            comparisons.append({
                "prompt_idx": pi,
                "prompt": prompt,
                "exact_match": exact,
                "token_match": match,
                "early_match": early_match,
                "late_match": late_match,
                "baseline_text": base_text,
                "skipped_text": text,
                "baseline_len": len(base_ids),
                "skipped_len": len(ids),
            })

        avg_match = total_match / len(GENERATION_PROMPTS)
        exact_count = sum(1 for c in comparisons if c["exact_match"])
        avg_early = sum(c["early_match"] for c in comparisons) / len(comparisons)
        avg_late = sum(c["late_match"] for c in comparisons) / len(comparisons)
        drift = avg_early - avg_late  # positive = errors compound, negative = self-corrects

        individual_results.append({
            "layer": skip_il,
            "avg_token_match": avg_match,
            "avg_early_match": avg_early,
            "avg_late_match": avg_late,
            "drift": drift,
            "exact_matches": exact_count,
            "total_prompts": len(GENERATION_PROMPTS),
            "comparisons": comparisons,
        })

        status = "PERFECT" if exact_count == len(GENERATION_PROMPTS) else (
            "GOOD" if avg_match > 0.9 else ("OK" if avg_match > 0.7 else "DEGRADED")
        )
        drift_arrow = "↘ compounds" if drift > 0.05 else ("↗ self-corrects" if drift < -0.05 else "→ stable")
        print(f"    Skip layer {skip_il:3d}: match={avg_match:.3f}  early={avg_early:.3f}  late={avg_late:.3f}  [{drift_arrow}]  exact={exact_count}/{len(GENERATION_PROMPTS)}  [{status}]")

        # Show divergences
        for c in comparisons:
            if not c["exact_match"]:
                bp = c["baseline_text"].replace("\n", "\\n")[:50]
                sp = c["skipped_text"].replace("\n", "\\n")[:50]
                print(f"      prompt {c['prompt_idx']}: \"{bp}\" vs \"{sp}\"")

    llm.set_layer_skip(None)

    # Sort by quality
    individual_results.sort(key=lambda x: -x["avg_token_match"])

    # ─── Progressive: stack the safest skips ────────────────────────
    print(f"\n  Progressive layer stacking (greedy generation)...")
    safe_order = [r["layer"] for r in individual_results]
    current_skips: List[int] = []
    progressive_log = []
    best_set: List[int] = []

    for candidate in safe_order:
        test_skips = current_skips + [candidate]
        llm.set_layer_skip(test_skips)

        total_match = 0.0
        exact_count = 0

        for pi, prompt in enumerate(GENERATION_PROMPTS):
            text, ids = generate_greedy(llm, prompt, max_tokens)
            base_text, base_ids = baselines[pi]
            match = token_match_rate(base_ids, ids)
            total_match += match
            if text == base_text:
                exact_count += 1

        avg_match = total_match / len(GENERATION_PROMPTS)

        entry = {
            "skipped_layers": list(test_skips),
            "n_skipped": len(test_skips),
            "avg_token_match": avg_match,
            "exact_matches": exact_count,
        }
        progressive_log.append(entry)

        pct = len(test_skips) / n_layer * 100
        passed = avg_match >= 0.85  # 85% token match threshold for generation
        status = "PASS" if passed else "FAIL"
        print(f"    Skip {len(test_skips):2d} layers ({pct:5.1f}%): match={avg_match:.3f}  exact={exact_count}/{len(GENERATION_PROMPTS)}  [{status}]")

        if passed:
            current_skips = test_skips
            best_set = list(test_skips)
        # Even on fail, keep trying (some layers might be fine to skip even when others aren't)

    llm.set_layer_skip(None)

    reduction_pct = len(best_set) / n_layer * 100

    print(f"\n  Best pruning set by generation quality ({len(best_set)}/{n_layer} layers, {reduction_pct:.1f}% reduction):")
    print(f"    Layers to skip: {best_set}")
    if best_set:
        print(f"    Active layers:  {[i for i in range(n_layer) if i not in best_set]}")

        # Show sample outputs with best set
        print(f"\n  Sample outputs with {len(best_set)} layers skipped:")
        llm.set_layer_skip(best_set)
        for pi in range(min(3, len(GENERATION_PROMPTS))):
            prompt = GENERATION_PROMPTS[pi]
            text, _ = generate_greedy(llm, prompt, max_tokens)
            base_text = baselines[pi][0]
            print(f"    Prompt:   \"{prompt}\"")
            print(f"    Baseline: \"{base_text[:100]}\"")
            print(f"    Pruned:   \"{text[:100]}\"")
            print()
        llm.set_layer_skip(None)

    results = {
        "individual_results": individual_results,
        "progressive_log": progressive_log,
        "best_skip_set": best_set,
        "n_skipped": len(best_set),
        "reduction_pct": reduction_pct,
        "baselines": {str(k): {"prompt": GENERATION_PROMPTS[k], "text": v[0]} for k, v in baselines.items()},
    }

    return results


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Layer Redundancy Analysis for Qwen3 Models")
    parser.add_argument("--model", type=str, required=True, help="Path to GGUF model file")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["similarity", "skip", "progressive", "generation", "all"],
                        help="Which analysis phase to run")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="Layers to offload to GPU (-1 = all)")
    parser.add_argument("--n-ctx", type=int, default=512, help="Context size")
    parser.add_argument("--main-gpu", type=int, default=0, help="GPU index to use (0 = first GPU)")
    parser.add_argument("--split-mode", type=int, default=None,
                        help="Split mode (0=none/single GPU, 1=layer, 2=row)")
    parser.add_argument("--skip-layers", type=str, default=None,
                        help="Comma-separated layer indices to test skipping (for --phase skip/generation)")
    parser.add_argument("--threshold", type=float, default=0.995,
                        help="Quality threshold for progressive pruning (cosine sim)")
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="Max tokens to generate per prompt in generation test")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    # Validate model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    # Auto-size context for generation phase
    n_ctx = args.n_ctx
    if args.phase in ("generation", "all") and n_ctx < args.max_tokens + 256:
        n_ctx = args.max_tokens + 256
        print(f"  Auto-sizing n_ctx to {n_ctx} (max_tokens={args.max_tokens} + 256 headroom)")

    # Load model — generation phase needs non-embedding mode
    t0 = time.time()
    needs_embeddings = args.phase in ("similarity", "skip", "progressive", "all")
    llm = load_model(str(model_path), n_gpu_layers=args.n_gpu_layers, n_ctx=n_ctx,
                     embeddings=needs_embeddings, main_gpu=args.main_gpu,
                     split_mode=args.split_mode)
    print(f"  Load time: {time.time() - t0:.1f}s")

    all_results = {"model": str(model_path), "n_layer": llm.get_n_layer(), "n_embd": llm.n_embd()}

    # Parse skip layers if provided
    skip_layers = None
    if args.skip_layers:
        skip_layers = [int(x.strip()) for x in args.skip_layers.split(",")]

    # Run phases
    sim_results = None
    skip_results = None

    if args.phase in ("similarity", "all"):
        t1 = time.time()
        sim_results = phase_similarity(llm, TEST_PROMPTS)
        all_results["similarity"] = sim_results
        print(f"\n  Phase 1 time: {time.time() - t1:.1f}s")

    if args.phase in ("skip", "all"):
        t2 = time.time()
        skip_results = phase_skip(llm, TEST_PROMPTS, skip_layers=skip_layers, similarity_results=sim_results)
        all_results["skip"] = skip_results
        print(f"\n  Phase 2 time: {time.time() - t2:.1f}s")

    if args.phase in ("progressive", "all"):
        t3 = time.time()
        prog_results = phase_progressive(llm, TEST_PROMPTS, skip_results=skip_results, threshold=args.threshold)
        all_results["progressive"] = prog_results
        print(f"\n  Phase 3 time: {time.time() - t3:.1f}s")

    if args.phase in ("generation", "all"):
        # Generation needs a non-embedding model — reload if we loaded in embedding mode
        if needs_embeddings:
            print(f"\n  Reloading model for generation (non-embedding mode)...")
            del llm
            llm = load_model(str(model_path), n_gpu_layers=args.n_gpu_layers,
                             n_ctx=n_ctx, embeddings=False,
                             main_gpu=args.main_gpu, split_mode=args.split_mode)
        t4 = time.time()
        gen_results = phase_generation(
            llm, skip_layers=skip_layers, similarity_results=sim_results,
            max_tokens=args.max_tokens,
        )
        all_results["generation"] = gen_results
        print(f"\n  Phase 4 time: {time.time() - t4:.1f}s")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Results saved to: {output_path}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print("Done!")

    return all_results


if __name__ == "__main__":
    main()
