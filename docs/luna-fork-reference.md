# Luna Fork — Reference: Extensions to llama.cpp and llama-cpp-python

This document lists every custom addition made to the LSDJesus forks of
`llama.cpp` and `llama-cpp-python` beyond the upstream JamePeng/ggml-org base.

---

## C++ Changes (vendor/llama.cpp)

### `include/llama.h`

**`llama_model_params` — new field:**
```c
bool skip_output_head;  // skip loading output.weight (lm_head) to save VRAM
                        // in embedding-only mode. Qwen3/Qwen3VL only.
```

**New C API functions (`[Luna]` tagged):**
```c
// Hidden state: output of last transformer block, BEFORE output_norm.
// = hidden_states[-2] in HuggingFace convention.
// Requires embeddings=true and POOLING_TYPE_NONE.
float * llama_get_embeddings_penultimate_ith(struct llama_context * ctx, int32_t i);

// Hidden state from a specific transformer layer for token i.
// Requires llama_set_layer_capture() to have been called for that layer.
float * llama_get_embeddings_layer_ith(struct llama_context * ctx, int32_t layer, int32_t i);

// Enable per-layer hidden state capture.
// mask: bool array [n_layers], true = capture this layer's output.
void llama_set_layer_capture(struct llama_context * ctx, const bool * mask, int32_t n_layers);

// Skip layers during inference (pass hidden state through unchanged).
// mask: bool array [n_layers], true = skip this layer.
void llama_set_layer_skip(struct llama_context * ctx, const bool * mask, int32_t n_layers);

// Get total number of transformer layers in the model.
int32_t llama_get_n_layer(struct llama_context * ctx);
```

---

### `src/llama-graph.h` / `src/llama-graph.cpp`

Added to the graph result struct:
- `ggml_tensor * t_embd_penultimate` — the penultimate capture tensor
- `std::vector<ggml_tensor *> t_embd_layers` — per-layer capture tensors
- `get_embd_penultimate()` / `get_embd_layer(il)` accessors
- Both tensors are marked as graph outputs so ggml schedules their extraction

---

### `src/llama-context.h` / `src/llama-context.cpp`

- `buffer_view<float> embd_penultimate` — host-side output buffer
- `std::vector<buffer_view<float>> embd_layers` — per-layer output buffers
- Buffer allocation uses `n_outputs_max * n_embd * sizeof(float)` per capture
- Two extraction paths wired up:
  - **Encode path** (single call, all tokens in one memcpy)
  - **Batched decode path** (multi-sequence, `n_outputs_prev` offset for correct placement)
- `output_reorder()` runs before all `get_embeddings_*` calls to handle sequence reordering
- `set_layer_capture()` / `set_layer_skip()` store boolean masks as `std::vector<bool>` on the context; accessible from graph builders as `layer_capture` / `layer_skip`

---

### `src/llama-model.cpp`

- `skip_output_head` respected in `load_tensors()` for `LLM_ARCH_QWEN3` and `LLM_ARCH_QWEN3VL`:
  - When `true`: passes `TENSOR_NOT_REQUIRED | TENSOR_SKIP` — tensor metadata is tracked but data is never loaded into VRAM
  - When `false` (default): normal load with tied-embedding fallback
- `llama_model_default_params()` default: `/*.skip_output_head =*/ false`

---

### `src/models/qwen3.cpp` + `qwen3vl.cpp` + `qwen35.cpp` + `qwen35moe.cpp`

All four builders have identical hook insertions:

**Inside the layer loop (top):**
```cpp
// [Luna] layer skip — pass hidden state through unchanged
if (layer_skip && il < (int) layer_skip->size() && (*layer_skip)[il]) {
    cb(inpL, "l_out", il);
    continue;
}
```

**Inside the layer loop (bottom, after l_out):**
```cpp
// [Luna] per-layer hidden state capture
if (layer_capture && il < (int) layer_capture->size() && (*layer_capture)[il]) {
    if (res->t_embd_layers.empty()) {
        res->t_embd_layers.resize(n_layer, nullptr);
    }
    res->t_embd_layers[il] = cur;
    cb(cur, "result_layer", il);
}
```

**After the loop, before output_norm:**
```cpp
// [Luna] hidden_states[-2] capture
if (cparams.embeddings) {
    res->t_embd_penultimate = cur;
    cb(res->t_embd_penultimate, "result_penultimate", -1);
}
```

**lm_head guard:**
```cpp
// lm_head — skip if output.weight was not loaded (skip_output_head mode)
if (model.output != nullptr) {
    cur = build_lora_mm(model.output, cur);
    ...
}
```

> **Qwen3.5 note:** Qwen3.5 and Qwen3.5-MoE are hybrid SSM/Transformer models
> (Delta Net base). The penultimate capture point is semantically identical —
> it is the residual stream exiting the last layer (whether recurrent or attention),
> before `output_norm`. The layer skip/capture hooks also work on recurrent layers;
> skipping one passes the residual stream through unchanged.

---

## Python Changes (llama_cpp/)

### `llama_cpp.py` — ctypes bindings

**`llama_model_params` struct** — added field:
```python
("skip_output_head", ctypes.c_bool),
```

**New function bindings:**
```python
llama_get_embeddings_penultimate_ith(ctx, i: c_int32) -> c_float_p
llama_get_embeddings_layer_ith(ctx, layer: c_int32, i: c_int32) -> c_float_p
llama_set_layer_capture(ctx, mask: c_bool_p, n_layers: c_int32) -> None
llama_set_layer_skip(ctx, mask: c_bool_p, n_layers: c_int32) -> None
llama_get_n_layer(ctx) -> c_int32
```

---

### `llama.py` — `Llama` high-level class

**Constructor parameter:**
```python
Llama(
    ...,
    skip_output_head: bool = False,   # saves ~220 MB VRAM on Qwen3-4B Q4
)
```

**Methods:**

```python
def get_penultimate_embeddings(
    token_positions: Optional[List[int]] = None
) -> Optional[np.ndarray]
# Returns: float32 [n_positions, n_embd]
# = hidden_states[-2]: last block output, BEFORE output_norm
# Requires: embeddings=True, POOLING_TYPE_NONE, after llama_decode()

def get_layer_embeddings(
    layer: int,
    token_positions: Optional[List[int]] = None
) -> Optional[np.ndarray]
# Returns: float32 [n_positions, n_embd]
# Requires: set_layer_capture([layer]) called before decode

def set_layer_capture(layers: Optional[List[int]] = None) -> None
# layers: list of 0-based layer indices to capture, or None to disable all

def set_layer_skip(layers: Optional[List[int]] = None) -> None
# layers: list of 0-based layer indices to skip, or None to disable all

def get_n_layer() -> int
# Total number of transformer layers in the loaded model
```

---

## Supported Models

| Model | penultimate | layer_capture | layer_skip | skip_output_head |
|---|---|---|---|---|
| Qwen3 (dense) | ✅ | ✅ | ✅ | ✅ |
| Qwen3-VL | ✅ | ✅ | ✅ | ✅ |
| Qwen3.5 (hybrid SSM) | ✅ | ✅ | ✅ | — |
| Qwen3.5-MoE | ✅ | ✅ | ✅ | — |

> `skip_output_head` is only wired into the model loader for Qwen3/Qwen3VL.
> Qwen3.5 uses a different loader case; add if needed.

---

## Usage Example

```python
from llama_cpp import Llama
import numpy as np

# Load model in embedding mode, skip lm_head to save VRAM
llm = Llama(
    model_path="Qwen3-4B.Q4_K_M.gguf",
    n_gpu_layers=-1,
    embeddings=True,
    pooling_type=0,           # LLAMA_POOLING_TYPE_NONE
    skip_output_head=True,    # saves ~220 MB VRAM
    n_ctx=2048,
    n_batch=512,
)

# Optional: capture hidden states from specific layers
n_layers = llm.get_n_layer()
llm.set_layer_capture([n_layers // 2, n_layers - 4])  # mid + near-final

# Encode a batch of sequences
tokens_a = llm.tokenize(b"Hello world")
tokens_b = llm.tokenize(b"Another sequence")
# ... fill and decode a batch ...

# Extract hidden_states[-2] for all tokens
hidden = llm.get_penultimate_embeddings()  # shape: [n_tokens, n_embd]

# Extract from a specific layer
mid_hidden = llm.get_layer_embeddings(n_layers // 2)  # shape: [n_tokens, n_embd]

# Extract just the last token (e.g. pooling via last-token)
last_token = llm.get_penultimate_embeddings(token_positions=[-1])  # shape: [1, n_embd]
```

---

## Tensor Semantics Reference

```
Token → tok_embd
  ↓
[ Transformer Block 0 ]   ← layer_capture[0] captures here
  ↓
[ Transformer Block 1 ]   ← layer_capture[1] captures here
  ...
[ Transformer Block N-1 ] ← layer_capture[N-1] captures here
  ↓
cur = inpL                ← t_embd_penultimate = cur  (hidden_states[-2])
  ↓
output_norm(cur)          ← t_embd = cur              (hidden_states[-1] / last_hidden_state)
  ↓
lm_head(cur)              ← t_logits                  (logits)
```
