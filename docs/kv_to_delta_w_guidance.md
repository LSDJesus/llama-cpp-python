# KV→ΔW Translation: Technical Guidance & Next Steps
### Research Continuation Document — February 25, 2026

---

## Context for the Copilot

This document provides technical guidance for the next phase of an ongoing experiment in semantic memory injection for large language models. The researcher has already demonstrated (same day, February 25 2026) that KV cache prefixes can transfer 10/13 fictional character details into model generation without those details appearing anywhere in the prompt — achieving 91% of direct prompt performance. The foundational experiment works. This document covers what to build next.

The goal of the next phase is to translate the temporary semantic influence of a KV cache prefix into permanent weight-level modifications — effectively teaching a model new facts in two forward passes on consumer hardware, in seconds, without conventional fine-tuning.

---

## Phase 1: Bulk Data Collection

### What We're Building

A dataset of tuples. Each tuple captures:

1. A KV cache state encoding a specific piece of knowledge (the prefix)
2. The layer activations produced when that KV cache influences generation of a test prompt
3. The layer activations produced when the same test prompt is run WITHOUT the prefix (baseline)
4. The delta between those two activation sets

This dataset is the raw material for everything that follows. Target size: 100–500 examples before any model training begins.

### Vary Surface Form Deliberately

When generating prefix examples, describe the same underlying fact in multiple different surface forms. For example, the same character's hair color expressed as:

- "Elara has silver-streaked auburn hair"
- "Her hair was a mix of silver and auburn"
- "Silver threads ran through Elara's dark auburn hair"

If the activation deltas for these three descriptions are geometrically similar despite different token sequences, that's the generalization signal we need. It means the model is encoding the *semantic content* not just the token pattern. If they're dissimilar, surface form is contaminating the signal and we need to understand why before proceeding.

### Layer Selection for Capture

Do not capture all layers. Based on the layer sensitivity mapping already completed, the critical band is approximately 30–55% depth. These layers are where semantic understanding is richest and most flexible. Late layers (>85% depth) are formatting and output-committed — their activations are less useful for this purpose. Early layers are syntactic — also less useful.

**Recommended capture targets:** Layers in the 30–55% band only. This reduces data volume and noise.

### What to Store Per Example

For each example, store:

```
{
  "fact_type": "character_physical" | "location" | "relationship" | "event" | etc.,
  "fact_content": "human readable description of the fact",
  "surface_form_id": integer (which surface form variant this is),
  "prefix_tokens": [...],
  "baseline_activations": { layer_id: [float array] },
  "prefix_activations": { layer_id: [float array] },
  "delta_activations": { layer_id: [float array] }  // prefix - baseline
}
```

Store as JSON or numpy .npz files. Keep raw data — don't discard anything yet.

---

## Phase 2: Bulk Data Analysis (Before Any Training)

### Why This Step Matters

Do not skip directly to training a model on this data. The bulk analysis pass is essentially free information about what the data actually looks like. It will tell you whether a pattern exists, how clean it is, and what architecture the translator network actually needs. Skipping this step risks training a model on noise or building a more complex architecture than necessary.

### Tool: UMAP

UMAP (Uniform Manifold Approximation and Projection) takes high-dimensional vectors and projects them to 2D while preserving neighborhood relationships. If your delta activation vectors cluster by fact type in 2D, you have a clean generalizable signal.

**Install:**
```bash
pip install umap-learn matplotlib numpy
```

**Basic usage:**
```python
import numpy as np
import umap
import matplotlib.pyplot as plt

# Load your delta activations — shape: (n_examples, activation_dim)
# Load your labels — fact type for each example

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(delta_activations)  # shape: (n_examples, 2)

# Plot colored by fact type
plt.figure(figsize=(12, 8))
for fact_type in unique_fact_types:
    mask = labels == fact_type
    plt.scatter(embedding[mask, 0], embedding[mask, 1], label=fact_type, alpha=0.7)
plt.legend()
plt.title("Delta Activation Space by Fact Type")
plt.savefig("umap_visualization.png", dpi=150)
plt.show()
```

### What You're Looking For

**Good signal (proceed to MLP):**
- Examples of the same fact type cluster together in the 2D projection
- Different fact types form distinct, separable clusters
- Different surface forms of the same fact land close together

**Weak signal (investigate before proceeding):**
- Clusters exist but overlap significantly
- Same fact described differently lands in different regions

**No signal (something is wrong):**
- Random scatter with no visible structure
- This would indicate capture layer choice problem, surface form contamination, or a more fundamental issue with the approach

### Additional Analysis: PCA

Before UMAP, run PCA and check how many principal components are needed to explain 90% of variance. If 5–10 components explain most of the variance, the signal is concentrated and structured. If you need 100+ components, the signal is diffuse. This informs MLP architecture depth.

```python
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(delta_activations)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
print(f"Components needed for 90% variance: {n_components_90}")
```

### Cosine Similarity Within Fact Types

For each fact type, compute the average cosine similarity between all pairs of delta activations within that type. High within-type similarity (>0.7) and low between-type similarity (<0.3) is the ideal pattern. This is a quantitative signal to complement the visual UMAP.

```python
from sklearn.metrics.pairwise import cosine_similarity

# For a specific fact type
type_mask = labels == "character_physical"
type_deltas = delta_activations[type_mask]
sim_matrix = cosine_similarity(type_deltas)
avg_within_similarity = (sim_matrix.sum() - np.trace(sim_matrix)) / (len(type_deltas) * (len(type_deltas) - 1))
print(f"Average within-type cosine similarity: {avg_within_similarity:.3f}")
```

---

## Phase 3: MLP Translator Architecture

### Only Begin This Phase If Phase 2 Shows Clear Structure

If the UMAP and similarity analysis show clean clusters, proceed. If not, revisit the data collection approach.

### What the MLP Needs to Learn

The MLP maps from delta activation space to weight delta space (ΔW). Specifically it learns:

```
f(delta_activations) → ΔW (expressed as low-rank matrices)
```

### Why Low-Rank (LoRA-Style) Weight Updates

Expressing ΔW as a low-rank update (two small matrices A and B where ΔW = A × B) serves two critical purposes:

1. **Prevents over-adjustment** — low-rank decomposition constrains how much weights can shift, acting as a natural regularizer that protects existing model knowledge from being overwritten
2. **Keeps updates tiny** — a full weight matrix for a large model layer could be billions of parameters. A low-rank approximation might be thousands. This is what makes KB-scale knowledge files possible.

The rank hyperparameter (typically 4, 8, or 16 in LoRA) controls the tradeoff between expressiveness and stability. Start with rank 8.

### Suggested MLP Architecture (Starting Point)

```python
import torch
import torch.nn as nn

class KVtoDeltaW(nn.Module):
    def __init__(self, activation_dim, target_layer_shape, rank=8):
        super().__init__()
        self.rank = rank
        target_rows, target_cols = target_layer_shape
        
        # Encoder: compress activation delta to latent representation
        self.encoder = nn.Sequential(
            nn.Linear(activation_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Two heads: produce the A and B matrices of the low-rank update
        self.head_A = nn.Linear(128, target_rows * rank)
        self.head_B = nn.Linear(128, rank * target_cols)
        
        self.target_rows = target_rows
        self.target_cols = target_cols
    
    def forward(self, delta_activations):
        latent = self.encoder(delta_activations)
        A = self.head_A(latent).view(-1, self.target_rows, self.rank)
        B = self.head_B(latent).view(-1, self.rank, self.target_cols)
        delta_W = torch.bmm(A, B)  # (batch, target_rows, target_cols)
        return delta_W
```

This is a starting point — not gospel. The actual architecture will depend on what the data analysis reveals about signal structure.

### Training Signal

The training objective is to produce a ΔW such that when applied to the model weights, the model generates outputs consistent with the injected facts **without** the KV cache prefix being present.

The loss function compares:
- Model output WITH prefix (target behavior)
- Model output WITHOUT prefix but WITH weight update applied (what we're training toward)

This requires running the model during training, which is expensive. A cheaper proxy: compare the activations at critical layers between the two conditions. If the activations match, the output will match. This is the **activation matching loss** — train ΔW to minimize the difference between prefix-influenced activations and weight-update-influenced activations.

---

## Phase 4: Validation

### The Two-Pass Test

The ultimate validation is clean and binary:

1. Take a fact the base model definitely does not know (fictional, zero-contamination)
2. Run two passes: forward pass with prefix to capture KV state → translator network produces ΔW → apply ΔW to model weights
3. Query the model about that fact with no prefix, no context, nothing
4. Did it answer correctly?

If yes — permanent knowledge formation via two forward passes is real.

### Composability Test

Apply two independent ΔW updates for two independent facts. Query for both. Do both answers hold? Does either interfere with the other? This tests whether the low-rank updates are genuinely composable or whether they interfere in weight space.

### Reversibility Test

Apply a ΔW update. Verify the fact is known. Remove the ΔW update (subtract it). Verify the fact is no longer known and the model returns to baseline behavior. This establishes that the updates are clean and non-destructive.

### Existing Knowledge Preservation Test

After applying a ΔW update for a new fact, run the model's standard benchmark questions. Verify that existing capabilities are not degraded. This is the catastrophic forgetting check — the low-rank constraint should prevent this but it needs to be empirically verified.

---

## Key Risks and Mitigations

**Risk: Activation deltas are noisy and don't generalize across surface forms**
Mitigation: The surface form variation in data collection is specifically designed to detect this. If it fails, investigate whether capturing from a single layer rather than multiple layers produces cleaner signal.

**Risk: MLP overfits to training examples and doesn't generalize to new facts**
Mitigation: Hold out 20% of your data collection as a test set. Never train on it. Evaluate generalization on held-out examples before claiming success.

**Risk: ΔW updates interfere with each other when composed**
Mitigation: The composability test will reveal this. If interference occurs, investigate whether orthogonality constraints on the low-rank matrices reduce it.

**Risk: Weight updates degrade existing model capabilities**
Mitigation: The low-rank constraint limits this. If degradation occurs, reduce rank or add an L2 regularization term penalizing large weight movements.

**Risk: The effect requires too many examples to be practically useful**
Mitigation: This is an empirical question. Even if the first version requires 10 examples of a fact to form a reliable weight update, that's still orders of magnitude cheaper than conventional fine-tuning. Document what you find.

---

## What Success Looks Like

A minimal viable demonstration of this mechanism needs to show four things:

1. A fact the base model does not know, confirmed by baseline queries
2. Two forward passes producing a ΔW update — timed on consumer hardware
3. The model answering correctly about that fact with no context, no prefix, nothing
4. Existing model capabilities unaffected

That four-point demonstration, combined with the already-proven KV injection results, is a complete research finding. It is also, if it works, one of the more significant results in applied LLM research in recent memory.

---

## Suggested Reading (For Deeper Context)

If the copilot wants background on the established techniques this builds on:

- **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2021) — the low-rank weight update approach
- **ROME: Rank-One Model Editing** (Meng et al., 2022) — editing factual associations in transformer weights directly
- **MEMIT** (Meng et al., 2023) — mass editing of model memories, related approach
- **KV Cache compression literature** — background on what the KV cache actually encodes

Note: none of these papers do what this experiment is attempting. They are background context, not prior art for this specific pipeline.

---

*KV→ΔW Translation Guidance — February 25, 2026*
*Part of the Semantic Memory Injection research program*
