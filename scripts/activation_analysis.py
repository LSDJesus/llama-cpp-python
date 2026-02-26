#!/usr/bin/env python3
"""
Activation Delta Analysis — UMAP + PCA + Cosine Similarity

Analyzes the activation delta dataset produced by kv_activation_capture.py.
Implements the Phase 2 analysis from kv_to_delta_w_guidance.md:
  1. PCA — how many components explain the variance?
  2. UMAP — do deltas cluster by memory/fact_type?
  3. Cosine similarity — within-type vs between-type
  4. Layer-band analysis — which layers carry the signal?

Usage:
    python scripts/activation_analysis.py --dataset scripts/activation_dataset_full.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ═══════════════════════════════════════════════════════════════════════
#  ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def load_dataset(npz_path: str):
    """Load activation dataset and metadata."""
    data = np.load(npz_path)
    meta_path = npz_path.replace(".npz", "_metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return data, meta


def pca_analysis(deltas: np.ndarray, label: str = ""):
    """PCA variance explained analysis.
    
    Args:
        deltas: shape (n_examples, feature_dim)
        label: description for printing
    """
    from sklearn.decomposition import PCA
    
    print(f"\n{'─'*60}")
    print(f"  PCA Analysis {f'— {label}' if label else ''}")
    print(f"{'─'*60}")
    
    # Center the data
    deltas_centered = deltas - deltas.mean(axis=0)
    
    pca = PCA()
    pca.fit(deltas_centered)
    
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    
    # Key thresholds
    for target in [0.50, 0.75, 0.90, 0.95, 0.99]:
        n = np.argmax(cumvar >= target) + 1
        print(f"  {target*100:5.0f}% variance in {n:>4} components")
    
    print(f"  Total components: {len(cumvar)}")
    print(f"  Top 10 individual: {pca.explained_variance_ratio_[:10].round(4)}")
    
    return pca, cumvar


def cosine_similarity_analysis(deltas: np.ndarray, labels: List[str], label_type: str = "memory"):
    """Within-type vs between-type cosine similarity.
    
    Args:
        deltas: shape (n_examples, feature_dim)
        labels: list of group labels per example
        label_type: name for the grouping (for display)
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    print(f"\n{'─'*60}")
    print(f"  Cosine Similarity Analysis (by {label_type})")
    print(f"{'─'*60}")
    
    unique_labels = sorted(set(labels))
    
    # Full similarity matrix
    sim_matrix = cosine_similarity(deltas)
    
    # Within-group similarities
    within_sims = []
    between_sims = []
    
    print(f"\n  Within-{label_type} similarities:")
    for lbl in unique_labels:
        mask = [l == lbl for l in labels]
        indices = [i for i, m in enumerate(mask) if m]
        if len(indices) < 2:
            print(f"    {lbl:<30} only 1 example")
            continue
        sub_matrix = sim_matrix[np.ix_(indices, indices)]
        n = len(indices)
        avg_within = (sub_matrix.sum() - np.trace(sub_matrix)) / (n * (n - 1))
        within_sims.append(avg_within)
        print(f"    {lbl:<30} {n} examples, avg sim = {avg_within:.4f}")
    
    # Between-group similarities
    print(f"\n  Between-{label_type} similarities:")
    for i, lbl_a in enumerate(unique_labels):
        for j, lbl_b in enumerate(unique_labels):
            if j <= i:
                continue
            idx_a = [k for k, l in enumerate(labels) if l == lbl_a]
            idx_b = [k for k, l in enumerate(labels) if l == lbl_b]
            sub_matrix = sim_matrix[np.ix_(idx_a, idx_b)]
            avg_between = sub_matrix.mean()
            between_sims.append(avg_between)
            print(f"    {lbl_a:<20} vs {lbl_b:<20} avg sim = {avg_between:.4f}")
    
    if within_sims and between_sims:
        print(f"\n  Summary:")
        print(f"    Avg within-{label_type} similarity:  {np.mean(within_sims):.4f}")
        print(f"    Avg between-{label_type} similarity: {np.mean(between_sims):.4f}")
        print(f"    Separation ratio:                  {np.mean(within_sims) / max(np.mean(between_sims), 1e-8):.2f}x")


def layer_band_analysis(deltas_3d: np.ndarray, n_layer: int, labels: List[str]):
    """Analyze which layer bands carry the strongest semantic signal.
    
    Tests the guidance doc hypothesis that 30-55% depth is the sweet spot.
    
    Args:
        deltas_3d: shape (n_examples, n_layer, n_embd)
        n_layer: total layer count
        labels: group labels per example
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    print(f"\n{'─'*60}")
    print(f"  Layer Band Analysis")
    print(f"{'─'*60}")
    
    bands = [
        ("Early (0-20%)",    0, int(n_layer * 0.20)),
        ("Lower-mid (20-35%)", int(n_layer * 0.20), int(n_layer * 0.35)),
        ("Core (35-55%)",    int(n_layer * 0.35), int(n_layer * 0.55)),
        ("Upper-mid (55-75%)", int(n_layer * 0.55), int(n_layer * 0.75)),
        ("Late (75-90%)",    int(n_layer * 0.75), int(n_layer * 0.90)),
        ("Final (90-100%)",  int(n_layer * 0.90), n_layer),
    ]
    
    unique_labels = sorted(set(labels))
    
    for band_name, start, end in bands:
        # Flatten the layer band into a single feature vector per example
        band_deltas = deltas_3d[:, start:end, :].reshape(deltas_3d.shape[0], -1)
        
        # Average L2 magnitude
        avg_l2 = np.mean(np.linalg.norm(band_deltas, axis=1))
        
        # Within-label cosine similarity
        sim_matrix = cosine_similarity(band_deltas)
        
        within_sims = []
        for lbl in unique_labels:
            indices = [i for i, l in enumerate(labels) if l == lbl]
            if len(indices) < 2:
                continue
            sub = sim_matrix[np.ix_(indices, indices)]
            n = len(indices)
            avg_w = (sub.sum() - np.trace(sub)) / (n * (n - 1))
            within_sims.append(avg_w)
        
        between_sims = []
        for i, a in enumerate(unique_labels):
            for j, b in enumerate(unique_labels):
                if j <= i:
                    continue
                ia = [k for k, l in enumerate(labels) if l == a]
                ib = [k for k, l in enumerate(labels) if l == b]
                sub = sim_matrix[np.ix_(ia, ib)]
                between_sims.append(sub.mean())
        
        w = np.mean(within_sims) if within_sims else 0
        b = np.mean(between_sims) if between_sims else 0
        ratio = w / max(abs(b), 1e-8)
        
        print(f"  {band_name:<22} layers {start:>2}-{end:>2} | "
              f"L2={avg_l2:>8.1f} | within={w:.3f} | between={b:.3f} | ratio={ratio:.2f}")


def umap_projection(deltas: np.ndarray, labels: List[str], colors: List[str],
                     title: str, output_path: str):
    """UMAP 2D projection with matplotlib.
    
    Args:
        deltas: shape (n_examples, feature_dim)
        labels: group label per example
        colors: color label per example (can differ from labels)
        title: plot title
        output_path: where to save the PNG
    """
    try:
        import umap
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"\n  UMAP skipped — missing dependency: {e}")
        print(f"  Install with: pip install umap-learn matplotlib")
        return
    
    print(f"\n{'─'*60}")
    print(f"  UMAP Projection — {title}")
    print(f"{'─'*60}")
    
    n_neighbors = min(15, len(deltas) - 1)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(deltas)
    
    plt.figure(figsize=(14, 10))
    
    unique_colors = sorted(set(colors))
    color_map = plt.cm.Set1(np.linspace(0, 1, len(unique_colors)))
    
    for ci, clr_label in enumerate(unique_colors):
        mask = [c == clr_label for c in colors]
        indices = [i for i, m in enumerate(mask) if m]
        pts = embedding[indices]
        plt.scatter(pts[:, 0], pts[:, 1], 
                   c=[color_map[ci]], label=clr_label, 
                   alpha=0.8, s=100, edgecolors='black', linewidths=0.5)
        
        # Annotate with labels
        for idx, pt in zip(indices, pts):
            plt.annotate(labels[idx], (pt[0], pt[1]), 
                        fontsize=7, alpha=0.6,
                        xytext=(5, 5), textcoords='offset points')
    
    plt.legend(loc='best', fontsize=9)
    plt.title(title, fontsize=14)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def detail_transfer_analysis(meta: dict):
    """Analyze relationship between detail transfer and data quality."""
    print(f"\n{'─'*60}")
    print(f"  Detail Transfer Quality")
    print(f"{'─'*60}")
    
    memories = meta["memories"]
    for m in memories:
        sf_tag = f"[form {m.get('surface_form_id', 0)}]" if m.get('n_surface_forms', 1) > 1 else ""
        score = m["n_injected_details"]
        total = len(m["detail_keys"])
        pct = 100 * score / total if total > 0 else 0
        bar = "█" * int(20 * pct / 100) + "░" * (20 - int(20 * pct / 100))
        status = "✓" if pct >= 30 else "⚠" if pct > 0 else "✗"
        print(f"  {status} {m['name']:<25} {sf_tag:<10} {score:>2}/{total:<2} ({pct:>5.1f}%) {bar}")
    
    # Filter recommendation
    scores = [m["n_injected_details"] / max(len(m["detail_keys"]), 1) 
              for m in memories]
    good = sum(1 for s in scores if s >= 0.3)
    print(f"\n  Examples with ≥30% detail transfer: {good}/{len(scores)}")
    print(f"  Recommended: filter to these for MLP training data")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Activation Delta Analysis")
    ap.add_argument("--dataset", required=True, help="Path to .npz dataset")
    ap.add_argument("--output-dir", default=None, help="Directory for output plots")
    ap.add_argument("--layer-band", type=str, default=None,
                    help="Restrict analysis to layer band, e.g. '12-22' (0-indexed)")
    args = ap.parse_args()
    
    # Load data
    data, meta = load_dataset(args.dataset)
    
    baseline = data["activations_baseline"]  # (n_examples, n_layer, n_embd)
    injected = data["activations_injected"]  
    deltas_3d = data["deltas"]               # (n_examples, n_layer, n_embd)
    
    n_examples, n_layer, n_embd = deltas_3d.shape
    print(f"Dataset: {n_examples} examples, {n_layer} layers, {n_embd} embd")
    print(f"Model: {meta['model']}")
    
    # Build labels
    memory_labels = [m["name"] for m in meta["memories"]]
    fact_type_labels = [m.get("fact_type", m["name"]) for m in meta["memories"]]
    form_labels = [f"{m['name']}_f{m.get('surface_form_id', 0)}" for m in meta["memories"]]
    
    # Output directory
    out_dir = args.output_dir or str(Path(args.dataset).parent)
    
    # ── Detail transfer quality ──────────────────────────────────
    detail_transfer_analysis(meta)
    
    # ── Determine layer band for flattened analysis ──────────────
    if args.layer_band:
        start, end = [int(x) for x in args.layer_band.split("-")]
        band_label = f"layers {start}-{end}"
    else:
        # Use all layers
        start, end = 0, n_layer
        band_label = "all layers"
    
    # Flatten deltas for selected layer band
    deltas_flat = deltas_3d[:, start:end, :].reshape(n_examples, -1)
    print(f"\nAnalysis band: {band_label} → feature dim = {deltas_flat.shape[1]}")
    
    # ── PCA ──────────────────────────────────────────────────────
    pca_analysis(deltas_flat, label=band_label)
    
    # ── Cosine similarity by memory ──────────────────────────────
    cosine_similarity_analysis(deltas_flat, memory_labels, label_type="memory")
    
    # ── Cosine similarity by fact type ───────────────────────────
    if len(set(fact_type_labels)) > 1:
        cosine_similarity_analysis(deltas_flat, fact_type_labels, label_type="fact_type")
    
    # ── Layer band analysis ──────────────────────────────────────
    layer_band_analysis(deltas_3d, n_layer, memory_labels)
    
    # ── UMAP projections ─────────────────────────────────────────
    umap_projection(
        deltas_flat, form_labels, memory_labels,
        title=f"Delta Activations by Memory ({band_label})",
        output_path=f"{out_dir}/umap_by_memory.png",
    )
    
    if len(set(fact_type_labels)) > 1:
        umap_projection(
            deltas_flat, form_labels, fact_type_labels,
            title=f"Delta Activations by Fact Type ({band_label})",
            output_path=f"{out_dir}/umap_by_fact_type.png",
        )
    
    # ── Layer-specific UMAP (core band only) ─────────────────────
    core_start = int(n_layer * 0.35)
    core_end = int(n_layer * 0.55)
    core_flat = deltas_3d[:, core_start:core_end, :].reshape(n_examples, -1)
    umap_projection(
        core_flat, form_labels, memory_labels,
        title=f"Delta Activations — Core Band (layers {core_start}-{core_end})",
        output_path=f"{out_dir}/umap_core_band.png",
    )
    
    print(f"\n{'═'*60}")
    print(f"  Analysis complete. Plots saved to: {out_dir}/")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
