# llama-cpp-python — Copilot Instructions

## Architecture Overview

This is a **Python ctypes binding layer** over the C library [llama.cpp](https://github.com/ggml-org/llama.cpp). The codebase is a JamePeng fork with custom extensions (embeddings, layer capture, semantic memory research). It has three architectural tiers:

1. **Low-level ctypes bindings** — `llama_cpp/llama_cpp.py` (~5k lines) and `llama_cpp/mtmd_cpp.py` mirror the llama.cpp C API 1:1 using a decorator pattern from `_ctypes_extensions.py`. Also `_ggml.py` for ggml tensor primitives.
2. **Intermediate wrappers** — `_internals.py` provides `LlamaModel`, `LlamaContext`, `LlamaBatch`, `LlamaSamplingContext` as Pythonic RAII wrappers over the C structs.
3. **High-level API** — `llama.py` (`Llama` class) is the primary user-facing interface; `llama_embedding.py` (`LlamaEmbedding`) extends it for embedding/reranking workloads; `llama_chat_format.py` handles chat template rendering and multimodal chat handlers.

The OpenAI-compatible server lives in `llama_cpp/server/` (FastAPI + SSE), using `LlamaProxy` in `model.py` for lazy model loading and `settings.py` (Pydantic) for configuration.

## Adding New C Bindings

When wrapping a new llama.cpp C function, follow the established decorator pattern in `llama_cpp.py`:

```python
# 1. Paste the C header comment verbatim above the binding
# LLAMA_API int32_t llama_n_vocab(const struct llama_vocab * vocab);
@ctypes_function(
    "llama_n_vocab",                          # exact C symbol name
    [llama_vocab_p_ctypes],                   # argtypes as ctypes
    ctypes.c_int32,                           # restype
)
def llama_n_vocab(vocab: llama_vocab_p, /) -> int:
    """Docstring describing the function."""
    ...  # body is just `...` — the decorator replaces it
```

- Use `NewType` pointer aliases (e.g. `llama_model_p`) for type hints, but `*_ctypes` variants (e.g. `llama_model_p_ctypes = ctypes.c_void_p`) for argtypes
- For `mtmd_cpp.py`, use `ctypes_function_mtmd` (same pattern, different shared library)
- Structs are `ctypes.Structure` subclasses with `_fields_` — add `TYPE_CHECKING` block for IDE hints

## Build System

**Build backend:** scikit-build-core compiles the vendored `vendor/llama.cpp/` via CMake and installs shared libraries into `llama_cpp/lib/`.

**Package manager:** This workspace uses `uv` (not plain pip) for fast, cached installs. Most packages are already cached locally.

**Environment setup:**
```powershell
uv venv .venv --python 3.13
.venv\Scripts\activate.ps1
uv pip install "scikit-build-core[pyproject]>=0.9.2" numpy typing-extensions diskcache jinja2 Pillow
```

**CUDA build on Windows (this workspace):**
```powershell
$env:CMAKE_BUILD_PARALLEL_LEVEL = "18"
$env:GGML_CUDA = "on"
uv pip install --verbose --no-build-isolation --config-settings=cmake.args="-DGGML_CUDA=ON;-DGGML_NATIVE=ON" -e .
```

**Build parallelism:** The machine has 20 logical cores. `CMAKE_BUILD_PARALLEL_LEVEL=18` uses 18 for compilation (leaves 2 for the OS). Without this, MSBuild defaults to ~4-6 threads and builds take 3-4x longer.

**CPU optimizations:** `-DGGML_NATIVE=ON` enables auto-detection of AVX2/FMA/F16C via `FindSIMD.cmake`. Without it, the CPU backend builds with no SIMD and logs `AVX2 available = false` at runtime despite the hardware supporting it.

**Key CMake options** (passed via `cmake.args` or `$env:CMAKE_ARGS`): `-DGGML_CUDA=ON`, `-DGGML_NATIVE=ON`, `-DGGML_METAL=ON`, `-DGGML_VULKAN=ON`, `-DGGML_BLAS=ON`. See `CMakeLists.txt` for the full target list and `llama_cpp_python_install_target()` helper.

**Cleaning build artifacts:** Remove `llama_cpp/lib/*.dll`, `llama_cpp/lib/*.so`, and `_skbuild/` or `build/` directories.

## Testing

```bash
python -m pytest --full-trace -v
```

- Tests live in `tests/` and require a vocab-only GGUF at `vendor/llama.cpp/models/ggml-vocab-llama-spm.gguf`
- Integration tests download real models via `huggingface_hub.hf_hub_download` (see `llama_cpp_model_path` fixture in `test_llama.py`)
- Test dependencies: `pip install -e ".[test]"` (includes scipy, httpx, huggingface-hub)

## Key Conventions

- **Python 3.9+ compatibility** — no walrus operators in hot paths; use `from __future__ import annotations` everywhere
- **Type hints are mandatory** — use `typing` imports, `Optional`, `Union`, `NewType` for pointer types
- **numpy for tensor data** — embeddings, logits, and token arrays use `npt.NDArray`; never raw Python lists for numerical work
- **Resource management** — wrapper classes implement `close()` + `__del__()` calling `llama_model_free()` / `llama_context_free()` etc. via the C API
- **Suppress C library stdout** — wrap noisy C calls with `suppress_stdout_stderr(disable=verbose)` from `_utils.py`
- **Chat format registration** — use `@register_chat_completion_handler("name")` decorator in `llama_chat_format.py`; multimodal handlers extend `Llava15ChatHandler`

## Server

Run: `python -m llama_cpp.server --model <path.gguf>` or configure via YAML with `--config_file`. The server exposes `/v1/completions`, `/v1/chat/completions`, `/v1/embeddings` endpoints. Double-lock pattern in `app.py` (`llama_outer_lock`/`llama_inner_lock`) handles concurrent request cancellation for streaming.

## Fork-Specific Extensions

This fork (JamePeng) includes experimental work:
- **`LlamaEmbedding`** in `llama_embedding.py` — specialized embedding/reranking class with auto-configuration, streaming batch processing, and normalization modes
- **Semantic Memory Injection research** — documented in `semantic_memory_injection.md`; involves layer-level embedding extraction and K/V injection into transformer attention layers
- **Custom C API extensions** (all `[Luna]`-tagged in the C++ source):
  - `llama_set_layer_capture(ctx, mask, n_layers)` — enable per-layer hidden state capture (bool mask)
  - `llama_set_layer_skip(ctx, mask, n_layers)` — skip layers during inference (layer pruning)
  - `llama_get_embeddings_layer_ith(ctx, layer, i)` — retrieve captured hidden state for layer+token
  - `llama_get_embeddings_penultimate_ith(ctx, i)` — pre-norm penultimate layer output (HF `hidden_states[-2]`)
  - `llama_get_n_layer(ctx)` — get total transformer layer count
  - Currently wired into model builders: `qwen3.cpp`, `qwen3vl.cpp`
  - Python high-level API: `Llama.set_layer_capture()`, `Llama.set_layer_skip()`, `Llama.get_layer_embeddings()`, `Llama.get_penultimate_embeddings()`

## File Reference

| File | Purpose |
|------|---------|
| `llama_cpp/llama_cpp.py` | Low-level ctypes bindings for llama.h |
| `llama_cpp/mtmd_cpp.py` | ctypes bindings for multimodal (mtmd) library |
| `llama_cpp/_internals.py` | RAII wrappers: LlamaModel, LlamaContext, LlamaBatch, LlamaSamplingContext |
| `llama_cpp/llama.py` | High-level `Llama` class — primary API |
| `llama_cpp/llama_chat_format.py` | Chat templates, function calling, multimodal handlers |
| `llama_cpp/llama_embedding.py` | `LlamaEmbedding` for embedding/reranking |
| `llama_cpp/server/app.py` | FastAPI server with OpenAI-compatible endpoints |
| `llama_cpp/server/settings.py` | Pydantic settings for server and model configuration |
| `CMakeLists.txt` | Builds vendor/llama.cpp, installs shared libs to llama_cpp/lib/ |
