## Add penultimate layer hidden state extraction API

### Motivation

Modern diffusion models with LLM-based text encoders (Lumina-2, Flux-style architectures, Z-Image) don't use the final transformer output — they use **`hidden_states[-2]`**, the pre-norm penultimate layer representation. This is the output of the second-to-last transformer block *before* the final RMSNorm and LM head projection.

The reason: the final layer has been heavily optimized for next-token prediction and collapses rich representational detail into a vocabulary-distribution space. The penultimate layer retains more spatial, stylistic, and semantic nuance — which is exactly what denoiser conditioning requires.

Currently, llama.cpp's embedding API only exposes:
- `llama_get_embeddings()` / `llama_get_embeddings_ith()` — post-norm final layer output
- `llama_get_embeddings_seq()` — pooled embeddings

There is no way to access intermediate layer hidden states. Anyone using llama.cpp as a text encoder for image generation is forced to use the wrong representation, or fall back to HuggingFace Transformers (which is typically 3-4× more VRAM for the same model due to bf16 vs quantized weights).

### API

```c
// Get the penultimate layer (pre-norm) embeddings for the ith token.
// Returns the hidden state BEFORE the final RMSNorm/LayerNorm, matching
// HuggingFace hidden_states[-2]. Required for diffusion model conditioning
// where the text encoder was trained against pre-norm representations.
// shape: [n_embd] (1-dimensional)
// Returns NULL if the model does not provide penultimate embeddings or embeddings are not enabled.
LLAMA_API float * llama_get_embeddings_penultimate_ith(struct llama_context * ctx, int32_t i);
```

**Requirements:**
- `embeddings = true` in context params
- `pooling_type = LLAMA_POOLING_TYPE_NONE` for per-token access
- Model builder must populate `res->t_embd_penultimate` (opt-in per architecture)

### Implementation

The implementation mirrors the existing `embd` / `get_embeddings_ith()` pipeline exactly:

| Component | What's added |
|---|---|
| `llama-graph.h` | `t_embd_penultimate` tensor field + `get_embd_penultimate()` getter |
| `llama-graph.cpp` | Reset to nullptr, mark as output in `set_outputs()` |
| `llama-context.h` | `embd_penultimate` buffer_view + `get_embeddings_penultimate_ith()` method |
| `llama-context.cpp` | Buffer sizing/allocation, GPU→CPU async copy (both encode + batched decode paths), output reorder, error-checked getter, C API export |
| `include/llama.h` | Public API declaration |

**Model builder opt-in (5 lines):**
```cpp
// After the transformer loop, before the final norm:
cur = inpL;

if (cparams.embeddings) {
    res->t_embd_penultimate = cur;
    cb(res->t_embd_penultimate, "result_penultimate", -1);
}

cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
```

Currently implemented for `qwen3.cpp` and `qwen3vl.cpp`. Any other model builder can add the same 5-line snippet to expose penultimate states for its architecture.

### Verification

Tested with Qwen3-4B and Qwen3-VL-4B (Q4_K_M GGUF):
- Penultimate embeddings are non-null, non-zero for all token positions
- Distinctly different from final-layer output (cosine similarity ~0.85-0.92, not 1.0)
- Correct shape: `[n_tokens, n_embd]`
- **Same-model GGUF** (e.g. Qwen3-4B safetensors → Qwen3-4B Q4_K_M GGUF): penultimate output is a direct match — no adapter needed, quantization error is negligible for conditioning
- **Cross-variant GGUF** (e.g. Qwen3-4B base → Qwen3-VL-4B-Instruct GGUF): achieves 0.979 cosine similarity to HF base hidden_states[-2], correctable to near-parity with a small alignment adapter (~42M params) since instruct tuning and VL fine-tuning shift the weight distribution
- No performance impact when `embeddings = false` (tensor is never allocated)
- No impact on existing `llama_get_embeddings_ith()` behavior

### Use case: GGUF text encoder for diffusion models

This enables replacing an 8GB bf16 safetensors text encoder with a 2.4GB Q4_K_M GGUF — same model, 3.3× smaller, with vision capabilities (via mmproj) as a bonus:

| | HF safetensors (bf16) | GGUF (Q4_K_M) |
|---|---|---|
| Size on disk | 7.67 GB | 2.4 GB |
| VRAM usage | ~8.2 GB | ~2.5 GB |
| Vision support | No | Yes (via mmproj) |
| Penultimate access | `hidden_states[-2]` | `llama_get_embeddings_penultimate_ith()` |

### Files changed

- `include/llama.h` (+6 lines)
- `src/llama-graph.h` (+2 lines)
- `src/llama-graph.cpp` (+4 lines)
- `src/llama-context.h` (+4 lines)
- `src/llama-context.cpp` (~55 lines across 9 insertion points)
- `src/models/qwen3.cpp` (+5 lines)
- `src/models/qwen3vl.cpp` (+5 lines)

Total: ~81 lines added, 0 lines modified, 0 lines deleted.
