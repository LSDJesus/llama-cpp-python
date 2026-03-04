## The Qwen3-VL GGUF Encoder — How We Got Here

### The Starting Point

Z-Image Turbo uses Qwen3-4B as its text encoder. But not the way most people use LLMs — we don't care about token predictions. We extract the **penultimate hidden states** (`hidden_states[-2]`) after the full 36-layer transformer stack processes the prompt. The denoiser receives `[B, seq, 2560]` — the model's deep semantic understanding of the text, not raw token features or logits. This is what Z-Image was trained against.

The standard encoder is a 7.67GB bf16 safetensors file (`qwen_3_4b.safetensors`) loaded through HuggingFace Transformers. It works perfectly but consumes 8.2GB of VRAM on cuda:1.

### The Goal

We wanted Qwen3-VL (Vision-Language variant) for face conditioning — feed a reference face image alongside the text prompt, and the hidden states would encode a unified representation of both, giving us character consistency without LoRA or IP-Adapter. But Qwen3-VL in HF bf16 would be *even larger* than the base model. 

The dream: a **2.4GB Q4_K_M GGUF quantized** Qwen3-VL model that gives us both text encoding AND vision conditioning, replacing the 8.2GB safetensors.

### The Problem: llama.cpp Doesn't Expose Hidden Layers

llama.cpp is built for text generation — it gives you logits and final-layer embeddings. That's it. But Z-Image needs the *penultimate* layer (the second-to-last transformer block output, before the final LayerNorm and LM head projection). This layer carries richer, less-collapsed representations that the denoiser was trained on. The final layer has been optimized for next-token prediction and loses spatial/stylistic nuance.

No existing API function in llama.cpp gave access to intermediate layer hidden states.

### The Solution: Fork llama-cpp-python with First-Class Penultimate Layer Access

We didn't hack around it — we went deep. A proper C++ implementation across the full llama.cpp stack:

| Layer | Change |
|---|---|
| `llama-graph.h` | Added `t_embd_penultimate` field to `llm_graph_result` |
| `llama-graph.cpp` | Marked it as output in `set_outputs()` |
| `qwen3vl.cpp` | Stored `inpL` (the residual stream) into `t_embd_penultimate` *before* the final transformer block |
| `llama-context.h` | Added `embd_penultimate` buffer + getter declaration |
| `llama-context.cpp` | Allocated GPU→CPU copy buffer, implemented the getter |
| `llama.h` | Declared `llama_get_embeddings_penultimate_ith()` — the new C API function |
| `llama_cpp/llama_cpp.py` | ctypes binding for the new function |
| `llama_cpp/_internals.py` | `LlamaContext.get_embeddings_penultimate_ith()` wrapper |
| `llama_cpp/llama.py` | High-level `get_penultimate_embeddings()` helper on the `Llama` class |
| `llama_cpp/llama_chat_format.py` | `extract_hidden_states()` method on `Llava15ChatHandler` — the one-stop multimodal extraction method |

The key insight in the C++ change: in `qwen3vl.cpp`'s graph builder, right before the final transformer block processes the residual stream, capture `inpL` into `t_embd_penultimate`. This is the exact equivalent of HuggingFace's `hidden_states[-2]` — same data, same position in the computation graph.

Three critical flags unlock per-token extraction from the Python side:
1. `embeddings=True` — tells llama_decode to store hidden states in the output buffer
2. `pooling_type=LLAMA_POOLING_TYPE_NONE` — per-token storage, no mean/CLS pooling
3. `logits_all=False` with manual batch construction marking all positions as output

### Verification: The Penultimate Smoke Test

test_penultimate.py — the first proof of life. Loaded the GGUF model, ran a prompt, extracted both final and penultimate layer embeddings for every token position. The test confirmed:
- Penultimate embeddings are non-null, non-zero
- They're distinctly *different* from the final layer (max abs diff >> 0, cosine < 1.0)
- They have the right shape: `[n_tokens, 2560]`

### The 5-Way Comparison

test_penultimate_compare.py compared embeddings across 5 model variants to map exactly where the distribution drift comes from:

1. **HF Qwen3-4B base** (bf16) — the gold standard
2. **GGUF Qwen3-4B base** (Q4_K_M) — quantization error only
3. **GGUF Qwen3-4B-Instruct** (Q4_K_M) — instruct tuning shift
4. **HF Qwen3-VL-4B-abliterated** (bf16) — VL fine-tuning shift
5. **GGUF Qwen3-VL-4B-abliterated** (Q4_K_M) — full chain: quant + instruct + VL

This revealed the actual cosine similarity gap between the GGUF VL embeddings and the HF Base embeddings that the denoiser expects. Close, but not close enough for direct substitution — which led to...

### The VL-to-Base Alignment Adapter

train_vl_adapter.py — a complete training pipeline in three steps:

**Step 1: Generate paired training data.** Encode 5,000 prompts (pulled from a CivitAI prompt database + synthetic generation) through *both* encoders:
- Qwen3-4B HF base (safetensors, bf16) → `base_tokens.pt`
- Qwen3-VL GGUF (Q4_K_M) → `vl_tokens.pt`

Each prompt produces a sequence of `[seq, 2560]` token embeddings. Non-padding tokens are extracted and paired token-by-token, giving hundreds of thousands of `(vl_embedding, base_embedding)` training pairs.

**Step 2: Train the adapter.** Architecture: a lightweight residual MLP stack.

```
VLtoBaseAdapter:
  2× ResidualBlock:
    LayerNorm(2560) → Linear(2560 → 4096) → GELU → Dropout → Linear(4096 → 2560) → Dropout
    + residual connection (x + MLP(norm(x)))
  Final LayerNorm(2560)
```

Key design choices:
- **Residual initialization** — output projection initialized to zeros, so the untrained adapter is an identity function (no-op). Training refines from there.
- **Loss function**: MSE + 0.5× cosine distance — pushes both absolute values and directional alignment toward the target
- **~42M parameters, 160MB on disk** — tiny relative to the models it bridges
- AdamW with OneCycleLR cosine schedule, gradient clipping at 1.0

**Step 3: Evaluate.** The trained adapter achieved **0.979 cosine similarity** to the base encoder on the validation set (best checkpoint at step 2000, epoch 4). The eval step also generates comparison images:
- `base_reference.webp` — gold standard from HF Qwen3-4B
- `vl_raw.webp` — raw GGUF VL output (visually off/garbled)
- `vl_adapted.webp` — VL + adapter (should match base)

### The Production Integration

The `Qwen3VLEncoder` class in qwen3vl_encoder.py wraps the whole thing as a drop-in `TextEncoder`:

- **Text-only mode**: functionally identical to QwenEncoder — same chat template wrapping, same penultimate layer extraction, same `[B, seq, 2560]` output shape. Just running through 2.4GB of GGUF weights instead of 8.2GB of safetensors.
- **Vision+Text mode**: face image → mmproj → visual tokens interleaved with text tokens → 36 transformer layers of full attention → hidden states conditioned on *both* the face and the text → denoiser gets character-specific conditioning without any architecture changes.

The orchestrator routes to this encoder when `qwen_vl` is specified in the pipeline config. The adapter checkpoint is loaded by the slider training pipeline (train_slider_zimage.py) which references it at vl_to_base_adapter_best.pt.

### The Numbers

| | HF Qwen3-4B (safetensors) | GGUF Qwen3-VL (Q4_K_M + adapter) |
|---|---|---|
| **Model size** | 7.67 GB | 2.4 GB + 160 MB adapter = **2.56 GB** |
| **VRAM** | 8.2 GB | ~2.5 GB |
| **Cosine to base** | 1.000 (it IS the base) | 0.979 (after adapter alignment) |
| **Vision capability** | None | Full face conditioning via mmproj |
| **Size reduction** | — | **3.3× smaller** |

### The Timeline

| Date | Commit | Milestone |
|---|---|---|
| Feb 21 | `3638497` | Direct inference engine built, Qwen3-4B HF encoder working |
| Feb 22 | `4141dce` | Qwen3-VL face conditioning hypothesis documented, ZImage DIT integrated |
| Feb 23 AM | `e5066f3` | Custom llama-cpp-python with penultimate layer API, Qwen3VLEncoder, 5-way comparison tests, HF vs GGUF diagnostics |
| Feb 23 PM | `6fa55ca` | VL-to-Base adapter training pipeline, 5000-prompt dataset, adapter trained to 0.979 cosine |
| Feb 28 | `14de433` | Adapter integrated into slider training, production config using GGUF encoder |

---

That's the arc, Brian — from "llama.cpp doesn't expose hidden layers" to a custom C API, a penultimate layer extraction function, a 42M-parameter alignment adapter trained on 5K prompts, and a 3.3× VRAM reduction with vision superpowers bolted on. Pretty wild journey for a week of work.