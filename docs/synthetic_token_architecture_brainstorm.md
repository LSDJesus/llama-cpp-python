# Synthetic Token Architecture — Brainstorm Notes
### February 26, 2026 — Brian Emmett

---

## 1. KV Cache Mechanics

Each token at each transformer layer produces exactly **one K vector and one V vector**, computed from the token's hidden state $h$ via learned projection matrices:

$$Q = h W_Q, \quad K = h W_K, \quad V = h W_V$$

For Qwen3-14B (40 layers, GQA with 8 KV heads, head_dim=128):
- Per token per layer: one $K$ of shape $[8 \times 128]$ + one $V$ of shape $[8 \times 128]$
- Per token total: $40 \times 2 \times 1024 = 81{,}920$ floats (~320KB at fp32)

**One position = one K,V pair per layer. This is a hard constraint.** Multi-head attention doesn't change this — the heads receive slices of the same $h$.

Attention is causal: prefix tokens do not attend back to the prompt. The prefix populates KV pairs into the cache; prompt tokens' Queries reach back and pull from those cached pairs.

---

## 2. Why CMA-ES Is the Wrong Tool

The current `synthetic_token_cmaes.py` optimizes a 5120-dim embedding $e$ using gradient-free evolution. Loss drops 5x (4.0 → 0.72) but generation recall is 0/13.

Root cause: **cold random initialization in 5120-dimensional space.** The optimizer finds a local attractor that satisfies activation metrics but is geometrically wrong for what attention actually needs. 5120 unconstrained dimensions is too large a search space for a gradient-free method.

The forward pass is **differentiable** — the model weights are the correct "decompression key" for gradient computation. The right algorithm is gradient descent (backprop) on $e$ with frozen model weights. This is called **soft prompt tuning** in the fine-tuning literature and works for exactly this reason.

---

## 3. LoRA as a Conceptual Frame

LoRA compresses a weight delta $\Delta W \in \mathbb{R}^{d \times d}$ into two small matrices:

$$\Delta W = B A, \quad A \in \mathbb{R}^{r \times d},\ B \in \mathbb{R}^{d \times r},\ r \ll d$$

This works because useful $\Delta W$ have **low intrinsic rank** — their information content lives in a low-dimensional subspace of the full space. SVD proves this subspace exists and is optimal; LoRA learns it freely without requiring orthogonality.

The synthetic token problem is LoRA-adjacent: the KV cache matrix $K \in \mathbb{R}^{117 \times d_{kv}}$ for a 117-token prefix has some intrinsic rank $r$. If $r$ is small (the 91% PCA-1 dominance from the activation analysis suggests it is), then $r$ virtual tokens can span the same subspace as 117 real ones.

---

## 4. SVD-Based Approach (Upgraded Architecture)

**Step 1 — Measure actual rank:**

Compute the full KV cache for the 117-token prefix at critical layers. Stack K matrices, run SVD, check how many singular values are needed to explain 90% of variance. This number is $r$ — the minimum virtual token count. Expected: 3–8.

**Step 2 — Warm initialization via pseudoinverse:**

Instead of random noise, initialize each virtual token $e_i$ using:

$$e_i \approx k_i^* W_K^+$$

Where $k_i^*$ is the $i$-th SVD basis vector of the target KV matrix and $W_K^+$ is the Moore-Penrose pseudoinverse of the projection matrix. This gives a deterministic, geometrically meaningful starting point.

The catch: the same $e_i$ must also satisfy $W_V$ simultaneously. $W_K^+$ ignores $V$ entirely. The remaining optimization effort reconciles the $W_K / W_V$ coupling.

**Step 3 — Masked loss function:**

Not all 5120 dimensions respond meaningfully to the Elara prefix. Compute a per-layer, per-dimension mask from the baseline dataset:

$$m_\ell = \left(|\Delta h_\ell^\text{target}| > \tau\right) \in \{0,1\}^{5120}$$

Apply the mask before computing cosine similarity:

$$\mathcal{L}_\ell = 1 - \frac{(\Delta h_\ell^\text{produced} \odot m_\ell) \cdot (\Delta h_\ell^\text{target} \odot m_\ell)}{|\cdots||\cdots|}$$

The ~4700 near-zero dimensions are excluded. The effective optimization dimensionality drops from 5120 to ~200–400 per layer. CMA-ES (or gradient descent) works much better in lower effective dimension.

**Critical:** freeze the mask from baseline data before optimization starts. Do not recompute per step — the optimizer will game a dynamic mask.

**Step 4 — Optimize with backprop (not CMA-ES):**

The model weights are the decompression function. The forward pass through the frozen model is differentiable. Adam on $e$ with the masked multi-layer loss and frozen weights is the correct solver.

Practical path: optimize in PyTorch/HuggingFace using the same model weights (GGUF → safetensors conversion, or load directly via `transformers`). Transfer the optimized $e$ vector back to llama.cpp for inference. The weights are identical — only the serialization format differs.

---

## 5. Per-Layer Decomposition

Instead of finding one $e$ that works across all 40 layers, find the optimal $e_\ell$ for each critical layer independently. Then:

1. Compute the similarity matrix between per-layer optimal embeddings
2. High cosine similarity between $e_{14}$ and $e_{18}$ → they're compatible → merge into one token
3. Low similarity → distinct semantic needs → keep separate tokens
4. Only retain tokens for layers where the KV delta actually matters (Gate 2 magnitude filter applied per-layer)

This is exactly what `per_layer_distillation.py` was designed to probe. The inter-layer similarity matrix is the key output — it determines minimum $r$ empirically from the model's own geometry rather than from assumptions.

**Self-suppression property:** a token $e_{14}$ whose K at layer 22 doesn't match the prompt's Q at layer 22 gets near-zero softmax weight at that layer and self-suppresses. Inter-layer interference may be partially mitigated by the attention mechanism itself.

---

## 6. Two Injection Strategies

### Approach A: Embedding injection (path to portable `.mem`)
Inject $r$ virtual embedding vectors. The model computes K,V via its own forward pass. Compressed representation is small (~$r \times 5120$ floats). The model weights are the decompression key.

- **Pro:** portable, storable, zero extra C API work
- **Con:** $W_K/W_V$ coupling across layers is the core hard problem

### Approach B: Direct KV write (validated by Experiment 2)
Write $k^*, v^*$ directly into the KV cache at specific layers, skipping the projection step entirely. Experiment 2 already proved this works (10/13). SVD-compressed basis vectors reduce storage further.

- **Pro:** no coupling problem, target values written exactly
- **Con:** requires a write API in the C extension (`llama_set_kv_layer`), still larger than Approach A

**Recommended sequencing:** Use Approach B to validate exactly which K,V values produce recall. Then use those validated targets to train the Approach A optimizer with correct targets rather than activation-proxy targets.

---

## 7. Gradient Exposure in llama.cpp

GGML has gradient infrastructure (`ggml_build_backward()`, `.grad` fields on tensors, gradient accumulation ops) but the **inference path never invokes it**. Intermediate activations are freed/overwritten as soon as they're no longer needed.

Exposing gradients for Grinder use (offline optimization only, not inference) would require:

1. At graph construction: call `ggml_build_backward()` on the inference graph
2. Mark the input embedding tensor as a grad-requiring leaf node
3. After forward pass: run `ggml_graph_compute()` on the backward graph
4. Read `e->grad` — that's $\frac{\partial \mathcal{L}}{\partial e}$

Known blockers to investigate:
- Flash attention kernel may not have a backward pass in GGML
- Backward pass requires storing all intermediate activations (memory cost)
- KV cache split-batch execution makes graph boundaries awkward

A `--grad-mode` flag that builds the backward graph and keeps activations — only for Grinder offline runs — would be sufficient. The PyTorch path is lower-risk for a first implementation.

---

## 8. Summary of What's Proven vs. What's Bet

| Claim | Status |
|---|---|
| KV prefix injection transfers semantic facts | **Proven** — 10/13 (77%) |
| Critical band at layers 14–22 | **Proven** — layer sensitivity mapping |
| 91% PCA-1 dominance (universal injection component) | **Proven** — activation delta analysis |
| SVD can identify minimum virtual token count | **Mathematically sound** — standard linear algebra |
| Pseudoinverse warm init is better than random | **Mathematically sound** — trivially true |
| Masked loss reduces effective search dimensionality | **Mathematically sound** — sparse signal recovery |
| Backprop through frozen weights finds correct $e$ | **Plausible** — soft prompt tuning literature supports it |
| Single $e$ can satisfy $W_K$ and $W_V$ simultaneously at multiple layers | **Open empirical question** — per-layer similarity matrix will answer it |

**The 0/13 CMA-ES result is not evidence the hypothesis is wrong.** It's evidence that gradient-free optimization in an uninitialized 5120-dim space doesn't converge to the right solution in 2000 evaluations. The architecture is a well-posed compressed sensing problem with a warm-initializable solver.

---

*Next: implement SVD rank measurement on existing KV cache captures, then prototype the pseudoinverse warm init with masked backprop via PyTorch.*
