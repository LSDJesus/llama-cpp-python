# KV Cache Pruning & Synthetic Token Optimization: Implementation Spec
### Research Continuation Document — February 25, 2026

---

## Context

This document defines the next research track: **Synthetic Token Compression**. The goal is to take a full KV cache prefix (e.g., 80 tokens describing a character) and compress it down to a single synthetic embedding vector — a virtual token — that when processed through the model's normal forward pass reproduces the semantic effect of the full prefix.

This builds directly on the activation delta analysis already completed. All filter criteria defined here are derived from existing experimental data — no new assumptions are being introduced.

---

## Why This Approach

Previous thinking treated the MLP translator as mapping from raw KV state to weight deltas (ΔW). This document proposes a cleaner two-stage pipeline:

**Stage 1 — Compress:** Find a synthetic token embedding that reproduces the pruned semantic KV state of a full prefix. Output is a single vector (~5120 floats, ~20KB). This is the portable `.mem` file — usable immediately for inference with zero context window token cost beyond the synthetic token itself.

**Stage 2 — Translate (existing plan, unchanged):** The MLP translator now operates on the compressed synthetic token representation rather than the raw 32M-number KV state. The compression step pre-solves the translator's dimensionality problem. Same ΔW output, far cleaner input.

The key architectural insight: rather than injecting pre-computed KV pairs around the model's attention mechanism, we inject a synthetic embedding vector *into* it. The model computes its own K and V projections from our synthetic token through its normal forward pass. This means the representation is fully integrated with the model's own processing rather than being bolted on externally.

---

## Stage 1: KV Cache Pruning

### Purpose

The full KV cache for an 80-token prefix contains approximately 32 million numbers (80 tokens × 40 layers × 2 (K+V) × 5120 dims). The actual semantic information content has been empirically measured at approximately 9 principal components — roughly 9% of variance after subtracting the universal injection signal. Pruning discards everything that isn't carrying semantic content about the target facts.

### The Three-Gate Filter

Every KV pair must pass all three gates to be retained. Apply in order — each gate reduces the candidate set before the next gate runs.

---

#### Gate 1: Layer Band Selection

**Keep:** Layers 14–22 only (the critical band, 35–55% depth in a 40-layer model).

**Discard:** All KV pairs from layers 0–13 (syntactic processing, low semantic content) and layers 23–39 (output formatting, already committed to generation trajectory).

**Basis:** Layer sensitivity mapping experiment (February 25, 2026). The layer 19 spike at 48% depth confirmed this band as the model's syntax-to-meaning transition zone. The band analysis table showed core band (layers 14–22) ΔL2 of 645.3 vs. 16.5 for early layers.

**Implementation:**
```python
CRITICAL_BAND_START = 14  # inclusive
CRITICAL_BAND_END = 22    # inclusive

def gate1_layer_filter(kv_cache):
    """Keep only KV pairs from critical band layers."""
    return {
        layer_id: kv_pairs 
        for layer_id, kv_pairs in kv_cache.items()
        if CRITICAL_BAND_START <= layer_id <= CRITICAL_BAND_END
    }
```

---

#### Gate 2: Activation Delta Magnitude Threshold

**Keep:** Token positions where the L2 norm of the activation delta (with-prefix minus without-prefix) exceeds a minimum threshold.

**Discard:** Token positions where the prefix made negligible difference to the model's internal state at that layer. These positions are not contributing semantic information.

**Basis:** The activation delta analysis showed that failed transfers (Sable form 1, 0/13 details) produced near-zero or anti-correlated deltas. High-transfer examples (Elara 3-form stack, Kael) showed strong delta magnitudes in the critical band.

**Threshold:** Start with the 50th percentile of delta magnitudes across your existing 13-example dataset. Adjust based on what fraction of positions survive — target retaining roughly the top 30–40% of positions by delta magnitude.

**Implementation:**
```python
import numpy as np

def gate2_magnitude_filter(kv_cache_critical_band, baseline_activations, 
                            prefix_activations, threshold_percentile=50):
    """Keep token positions with significant activation delta magnitude."""
    
    # Compute delta for each token position at each critical layer
    deltas = {}
    for layer_id in kv_cache_critical_band:
        prefix_act = prefix_activations[layer_id]    # shape: (n_tokens, hidden_dim)
        baseline_act = baseline_activations[layer_id] # shape: (n_tokens, hidden_dim)
        delta = prefix_act - baseline_act             # shape: (n_tokens, hidden_dim)
        deltas[layer_id] = delta
    
    # Compute L2 norm of delta per token position (averaged across layers)
    all_deltas = np.stack(list(deltas.values()))  # (n_layers, n_tokens, hidden_dim)
    delta_norms = np.linalg.norm(all_deltas, axis=2).mean(axis=0)  # (n_tokens,)
    
    # Apply threshold
    threshold = np.percentile(delta_norms, threshold_percentile)
    significant_positions = np.where(delta_norms > threshold)[0]
    
    # Filter KV cache to significant positions only
    pruned_kv = {}
    for layer_id, kv_pairs in kv_cache_critical_band.items():
        pruned_kv[layer_id] = {
            pos: kv_pairs[pos] 
            for pos in significant_positions 
            if pos in kv_pairs
        }
    
    return pruned_kv, significant_positions, delta_norms
```

---

#### Gate 3: Semantic Alignment Score

**Keep:** Token positions whose activation delta aligns strongly with the semantic residual signal — the 9% of variance remaining after subtracting the universal injection component (first principal component).

**Discard:** Positions whose delta is dominated by the universal "I was primed with context" signal rather than fact-specific semantic content.

**Basis:** PCA analysis showed 91% of variance in a single universal component. The fact-specific content lives in the remaining 9%. Gate 2 filtered by magnitude — this gate filters by *direction*, keeping only positions pointing toward semantic content rather than generic priming signal.

**Threshold:** Cosine similarity to semantic residual > 0.3. This is a soft threshold — adjust based on how many positions survive. You want to retain enough positions to cover the target facts without including noise.

**Critical note:** The universal component (first PC) must be computed once from your existing 13-example dataset and stored as a fixed reference vector. Do not recompute per batch — you need a consistent reference direction across all examples.

**Implementation:**
```python
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Run this ONCE on your existing 13-example dataset and save the result
def compute_universal_component(all_delta_activations):
    """
    Compute the universal injection direction from existing data.
    Save this vector — use it as fixed reference for all future filtering.
    
    all_delta_activations: shape (n_examples, hidden_dim)
    """
    pca = PCA(n_components=1)
    pca.fit(all_delta_activations)
    universal_component = pca.components_[0]  # shape: (hidden_dim,)
    np.save('universal_injection_component.npy', universal_component)
    return universal_component


def subtract_universal_component(delta_vector, universal_component):
    """Project out the universal injection direction from a delta vector."""
    projection = np.dot(delta_vector, universal_component) * universal_component
    return delta_vector - projection


def gate3_semantic_alignment_filter(pruned_kv_gate2, delta_activations, 
                                     universal_component, threshold=0.3):
    """Keep positions whose deltas align with semantic content, not generic priming."""
    
    semantic_positions = []
    
    for layer_id, kv_pairs in pruned_kv_gate2.items():
        for pos in kv_pairs:
            # Get delta for this position at this layer
            delta = delta_activations[layer_id][pos]  # shape: (hidden_dim,)
            
            # Subtract universal component
            semantic_delta = subtract_universal_component(delta, universal_component)
            
            # Compute alignment: how much of the remaining signal is semantic?
            original_norm = np.linalg.norm(delta)
            semantic_norm = np.linalg.norm(semantic_delta)
            semantic_ratio = semantic_norm / (original_norm + 1e-8)
            
            if semantic_ratio > threshold:
                semantic_positions.append((layer_id, pos, semantic_ratio))
    
    # Build final pruned KV cache
    final_pruned_kv = {}
    for layer_id, pos, score in semantic_positions:
        if layer_id not in final_pruned_kv:
            final_pruned_kv[layer_id] = {}
        final_pruned_kv[layer_id][pos] = pruned_kv_gate2[layer_id][pos]
    
    return final_pruned_kv, semantic_positions


def full_pruning_pipeline(kv_cache, baseline_activations, prefix_activations, 
                           universal_component):
    """Run all three gates in sequence."""
    
    # Gate 1: Layer band
    g1 = gate1_layer_filter(kv_cache)
    print(f"Gate 1: {sum(len(v) for v in kv_cache.values())} → "
          f"{sum(len(v) for v in g1.values())} KV pairs")
    
    # Gate 2: Magnitude
    g2, significant_positions, delta_norms = gate2_magnitude_filter(
        g1, baseline_activations, prefix_activations
    )
    print(f"Gate 2: → {sum(len(v) for v in g2.values())} KV pairs "
          f"({len(significant_positions)} significant positions)")
    
    # Gate 3: Semantic alignment
    g3, semantic_positions = gate3_semantic_alignment_filter(
        g2, 
        {l: prefix_activations[l] - baseline_activations[l] 
         for l in range(CRITICAL_BAND_START, CRITICAL_BAND_END + 1)},
        universal_component
    )
    print(f"Gate 3: → {sum(len(v) for v in g3.values())} KV pairs "
          f"(semantic content retained)")
    
    return g3
```

---

### What Survives Pruning

After all three gates, you have a sparse set of KV pairs that are:
- From the right layer depth (semantic processing zone)
- Activation-significant (the prefix actually changed something here)
- Semantically aligned (the change is about the facts, not generic priming)

This is your **ground truth target** for the synthetic token optimization. Everything downstream is trying to reproduce this pruned set.

**Expected compression at pruning stage:** From ~32M numbers (full KV cache) down to roughly 50K–200K numbers (pruned critical band, significant positions only). This is before synthetic token compression — just the data reduction from smart filtering.

---

## Stage 2: Synthetic Token Optimization

### Goal

Find a single vector `v` in the model's embedding space (shape: `hidden_dim = 5120`) such that when `v` is processed through the model's normal forward pass as a token, the resulting K and V projections at each critical band layer closely match the pruned KV target from Stage 1.

### Why a Token Embedding Rather Than Direct KV Injection

Direct KV injection inserts pre-computed pairs around the model's attention mechanism. Synthetic token embedding injection inserts a vector *into* the mechanism — the model computes its own K and V projections from it through its normal learned weight matrices. This means:

- The representation is natively integrated with the model's processing
- Multiple synthetic tokens are attended to selectively based on query relevance, exactly like real tokens
- Positional identity is preserved — no merging ambiguity
- The optimization target is well-defined: minimize distance between produced KV projections and pruned target

### Optimization Loop

**Input:** Pruned KV target (from Stage 1), model with embedding access

**Output:** Synthetic embedding vector `v` (~5120 floats, ~20KB on disk)

```python
import numpy as np
from typing import Dict, Tuple

def optimize_synthetic_token(
    pruned_kv_target: Dict,          # output of full_pruning_pipeline
    model,                            # llama.cpp model instance with layer capture
    hidden_dim: int = 5120,
    n_iterations: int = 10000,
    initial_lr: float = 0.01,
    perturbation_scale: float = 0.001,
    convergence_threshold: float = 0.01
) -> np.ndarray:
    """
    Gradient-free optimization to find synthetic token embedding.
    Uses perturbation hill-climbing — no gradient access required.
    """
    
    # Initialize: start from mean of existing high-transfer prefix embeddings
    # if available, otherwise random normal
    v = np.random.normal(0, 0.1, hidden_dim).astype(np.float32)
    
    current_loss = evaluate_synthetic_token(v, pruned_kv_target, model)
    best_loss = current_loss
    best_v = v.copy()
    
    print(f"Initial loss: {current_loss:.4f}")
    
    for iteration in range(n_iterations):
        
        # Perturb: add random noise scaled to perturbation_scale
        perturbation = np.random.normal(0, perturbation_scale, hidden_dim).astype(np.float32)
        v_candidate = v + perturbation
        
        # Evaluate candidate
        candidate_loss = evaluate_synthetic_token(v_candidate, pruned_kv_target, model)
        
        # Accept if improvement
        if candidate_loss < current_loss:
            v = v_candidate
            current_loss = candidate_loss
            
            if current_loss < best_loss:
                best_loss = current_loss
                best_v = v.copy()
        
        # Logging
        if iteration % 500 == 0:
            print(f"Iteration {iteration}: loss={current_loss:.4f}, "
                  f"best={best_loss:.4f}")
        
        # Convergence check
        if best_loss < convergence_threshold:
            print(f"Converged at iteration {iteration}")
            break
        
        # Adaptive perturbation scale — reduce as we converge
        if iteration % 1000 == 0 and iteration > 0:
            perturbation_scale *= 0.95
    
    print(f"Optimization complete. Final loss: {best_loss:.4f}")
    return best_v


def evaluate_synthetic_token(
    v: np.ndarray,
    pruned_kv_target: Dict,
    model
) -> float:
    """
    Inject synthetic token v, capture resulting KV projections,
    measure distance to pruned target.
    """
    
    # Inject v as a token embedding and run forward pass
    # Capture KV projections at critical band layers
    produced_kv = inject_and_capture(v, model)
    
    # Compute loss: mean squared distance between produced and target KV pairs
    total_loss = 0.0
    n_pairs = 0
    
    for layer_id in pruned_kv_target:
        if layer_id not in produced_kv:
            continue
        for pos in pruned_kv_target[layer_id]:
            target_k, target_v = pruned_kv_target[layer_id][pos]
            produced_k, produced_v = produced_kv[layer_id].get(0, (None, None))
            
            if produced_k is not None:
                # Cosine distance rather than L2 — we care about direction not magnitude
                k_loss = 1.0 - cosine_similarity(
                    target_k.reshape(1, -1), 
                    produced_k.reshape(1, -1)
                )[0, 0]
                v_loss = 1.0 - cosine_similarity(
                    target_v.reshape(1, -1), 
                    produced_v.reshape(1, -1)
                )[0, 0]
                total_loss += k_loss + v_loss
                n_pairs += 2
    
    return total_loss / max(n_pairs, 1)
```

### Gradient Descent Alternative

If gradient access through the embedding layer is available (PyTorch wrapper around the model), gradient descent will converge orders of magnitude faster than perturbation hill-climbing:

```python
import torch

def optimize_synthetic_token_gradient(pruned_kv_target, model_torch, 
                                       hidden_dim=5120, n_steps=1000, lr=0.001):
    """
    Gradient-based optimization — much faster if gradient access is available.
    """
    v = torch.randn(hidden_dim, requires_grad=True, dtype=torch.float32)
    optimizer = torch.optim.Adam([v], lr=lr)
    
    for step in range(n_steps):
        optimizer.zero_grad()
        
        # Forward pass with synthetic token
        produced_kv = model_torch.forward_with_embedding(v)
        
        # Compute loss against pruned target
        loss = compute_kv_distance_loss(produced_kv, pruned_kv_target)
        
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Step {step}: loss={loss.item():.4f}")
    
    return v.detach().numpy()
```

---

## Stage 3: Validation

### Detail Transfer Test

The primary validation is identical to the original KV injection experiment. Take the optimized synthetic token `v`, inject it as a single token embedding, run the bare prompt ("Write a short story about Elara..."), and count detail transfers. 

**Target:** ≥7/13 details transferred from a single synthetic token. This would be comparable to or better than the full 80-token prefix (10/13) on a per-token basis — 1 token vs 80 tokens.

### Compression Ratio Measurement

```
Full prefix:        80 tokens × 5120 dims × 2 bytes = 819,200 bytes (~800KB of KV cache)
Synthetic token:    1 × 5120 dims × 4 bytes = 20,480 bytes (~20KB)
Compression ratio:  ~40x in storage, ~80x in context window tokens
```

If the semantic content is preserved, this is the `.mem` file format. Store `v` as a flat float32 array. Load it, inject it, done.

### Multi-Token Stack Test

Optimize synthetic tokens for two independent memories (e.g., Elara and Kael). Stack both as a 2-token prefix. Run dual-recall prompt. Measure whether both character sheets are recalled with comparable fidelity to the sequential KV cache stacking experiment (which achieved 17/25).

If stacking synthetic tokens preserves addressability — and it should, because positional identity is maintained — this confirms the format is composable.

### Robustness Test

Optimize a synthetic token for Elara using surface forms f0 and f1. Test recall using a prompt constructed from surface form f2 vocabulary (words that didn't appear in training forms). If recall holds, the synthetic token has captured semantic content rather than surface token patterns.

---

## Output Artifacts

After successful validation, this pipeline produces:

**The `.mem` file format:**
```
Header:     model_id (hash), hidden_dim, optimization_loss, detail_transfer_score
Payload:    float32 array of shape (n_synthetic_tokens, hidden_dim)
Metadata:   fact_type, source_description, creation_timestamp, version
```

A complete character sheet compressed to ~20KB per synthetic token. Portable across inference sessions. Stackable. Individually addressable by the model's own attention mechanism. Zero prompt token cost beyond the synthetic tokens themselves.

---

## Relationship to ΔW Translation

Once the compression pipeline is validated, the ΔW translator MLP input changes from raw KV state (32M numbers) to synthetic token embedding (5120 numbers). The translator now learns:

```
f(synthetic_token_embedding) → ΔW (low-rank weight update)
```

This is a dramatically simpler mapping. The compression step has already extracted the semantic content into a clean, low-dimensional representation. The translator's job is just converting between two compact representations rather than finding signal in 32 million noisy numbers.

The complete pipeline becomes:

```
Fact sheet (natural language)
    ↓
Full KV cache capture
    ↓
Three-gate pruning
    ↓
Synthetic token optimization
    ↓ ────────────────────────────────┐
Inject for working memory          Translate for permanent memory
(.mem file, ~20KB, zero tokens)    (ΔW, LoRA-style weight update)
```

One compression step. Two outputs. User chooses working memory, permanent memory, or both.

---

*KV Pruning & Synthetic Token Optimization Spec — February 25, 2026*
*Part of the Semantic Memory Injection research program*
