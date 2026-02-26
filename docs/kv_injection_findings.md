# KV Cache Semantic Injection: Experimental Findings
### February 25, 2026 — Brian Emmett

---

## Executive Summary

We demonstrate that a language model's KV (key-value) cache can carry semantic information from a prefix context into generation, causing the model to produce specific fictional details that were **never present in the prompt**. In a controlled experiment using purely fictional content (eliminating pretraining contamination), a KV cache prefix containing 13 character/setting details transferred **10 of 13 details (77%)** into the generated output — matching **91% of the performance** of embedding those same details directly in the prompt text.

This finding establishes a mechanism for "teaching" a model information through attention state alone, opening a path toward permanent knowledge implantation via learned weight modifications.

---

## Background

This research builds on a custom fork of [llama-cpp-python](https://github.com/ggml-org/llama.cpp) with C-level extensions for per-layer hidden state capture and layer skipping during inference. The extensions allow:

- **`llama_set_layer_skip(ctx, mask, n_layers)`** — skip specific transformer layers during forward pass (hidden state passes through unchanged)
- **`llama_set_layer_capture(ctx, mask, n_layers)`** — capture per-layer hidden states
- **`llama_get_embeddings_layer_ith(ctx, layer, i)`** — retrieve captured hidden states

These were developed as part of an ongoing semantic memory injection research program.

---

## Experiment 1: Layer Sensitivity Mapping

### Method

For two models (Qwen3-32B-Uncensored, 64 layers; Qwen3-14B-Claude-Opus-Distill, 40 layers), we ran a fixed creative writing prompt 60+ times — once as a baseline with no layers skipped, then once more skipping each individual layer in turn. Output quality was scored on factual accuracy (4 target facts) and prose coherence.

### Key Finding: Transformer layers cluster into "critical" and "resilient" bands

**Qwen3-14B Layer Sensitivity Map** (40 layers):
```
C = critical (output degrades when skipped)
. = resilient (output unaffected when skipped)
~ = moderate

~~C....~C...CC~CC.CC..C.~C~.C.CC.C~~.C~~
0         1         2         3
0123456789012345678901234567890123456789
```

- **Critical layers** concentrate at 30–55% depth — where the model forms semantic understanding
- **Late layers (>85%)** are almost universally resilient — they polish/format, not decide
- Over half of single-layer skips on the 32B model **fixed** a degenerate baseline (instruction-repeating), producing coherent prose with all 4 facts

### Implication

The sensitivity map identifies which layers carry the most semantic leverage — and therefore which layers would be the highest-value targets for KV cache injection or weight modification.

---

## Experiment 2: KV Cache Semantic Injection

### Motivation

The layer skip experiment showed that individual layers contribute unevenly to output quality. This raised a question: if we pre-load information into the KV cache (via a text prefix), can the model's attention mechanism transfer that information into generation — even when the prompt itself contains none of those details?

### Method

We designed a **zero-contamination fictional prompt** to eliminate any possibility that the model "already knows" the target details from pretraining:

**Bare prompt** (given to the model):
> *"Write a short story about a woman named Elara walking through a forest at dusk. Three paragraphs. Rich sensory details."*

**Details prefix** (loaded into KV cache only, never appears in the prompt):
> - Elara Voss, 34 years old
> - Silver-streaked auburn hair, crescent-shaped scar on left palm
> - Forest: Thornwood, in the Greywander Basin
> - Late November, first frost
> - Companion: wolfhound named Cassius, one blue eye, one amber eye
> - Brass compass belonging to grandmother Mirabel

**Detection targets**: 13 specific, unusual details checked via regex (e.g., "silver", "auburn", "Thornwood", "Cassius", "wolfhound", "brass compass", "Mirabel", "crescent", "November", "frost", heterochromia, age 34, "Greywander").

**Model**: Qwen3-14B-Claude-4.5-Opus-Distill (Q4_K_M), 40 layers, temp=0 (deterministic).

### Conditions

| ID | Condition | Description |
|----|-----------|-------------|
| A | **Baseline** | Bare prompt only. No details anywhere. |
| B | **Full KV Injection** | Details prefix processed first → KV cache populated at all layers. Then bare prompt processed and generation proceeds. |
| C | **Critical-Only Injection** | Details prefix processed with resilient layers skipped → KV entries for details exist only at critical layers. |
| D | **Resilient-Only Injection** | Details prefix processed with critical layers skipped → KV entries exist only at resilient layers. |
| E | **Direct Prompt** | All details embedded directly in the prompt text. Quality ceiling. |

### Results

| Condition | Details Found | Score |
|-----------|:---:|:---:|
| **A) Baseline** — no details | **0 / 13** | 0% |
| **B) Full KV Injection** — prefix in KV cache | **10 / 13** | 77% |
| **C) Critical-Only** — partial KV (critical layers) | **0 / 13** | 0% |
| **D) Resilient-Only** — partial KV (resilient layers) | **0 / 13** | 0% |
| **E) Direct Prompt** — details in prompt text | **11 / 13** | 85% |

#### Details breakdown for Full KV Injection (Condition B):

| Detail | Found? | Matched Text |
|--------|:---:|---|
| Silver hair | **Yes** | "silver-streaked auburn hair" |
| Auburn hair | **Yes** | (same as above) |
| Crescent scar | **Yes** | "scar that bisected her left palm—a pale, curved wound" |
| Forest: Thornwood | **Yes** | "slanted through Thornwood like spilled wine" |
| November | **Yes** | implicit via seasonal description |
| First frost | **Yes** | "frost-stiff leaves" |
| Wolfhound | **Yes** | "wolfhound" |
| Companion: Cassius | **Yes** | "Cassius's paws crunched on frost-stiff leaves" |
| Heterochromia | **Yes** | "one blue as riverstone, one amber like smoldering resin" |
| Brass compass | **Yes** | "grandmother's brass compass trembled slightly in Elara's grip" |
| Age: 34 | No | — |
| Valley: Greywander | No | — |
| Grandmother: Mirabel | No | — |

### Sample Output Comparison

**Condition A (Baseline — 0/13 details):**
> *"The forest welcomed Elara with a breath of damp earth and crushed pine. As she stepped from the clearing, the last light of day spilled through the canopy in molten ribbons..."*

Generic forest scene. No character details. No named companion, no specific setting, no physical description.

**Condition B (Full KV Injection — 10/13 details):**
> *"The last light of day slanted through Thornwood like spilled wine, staining the skeletal trees in amber and bruise-purple. Elara's breath hung in crystalline wisps as she stepped over gnarled roots... Cassius's paws crunched on frost-stiff leaves, his mismatched eyes—one blue as riverstone, one amber like smoldering resin—tracking the way ahead... Her grandmother's brass compass trembled slightly in Elara's grip, its needle quivering though no wind stirred."*

> *"She paused, brushing a lock of silver-streaked auburn hair from her face, and let her fingers linger on the scar that bisected her left palm—a pale, curved wound that itched when the temperature dropped."*

Rich, specific, and factually consistent with the injected details — despite those details never appearing in the prompt.

---

## Key Findings

### 1. KV Cache Carries Semantic State

The KV cache is not merely a computational buffer — it encodes semantic content that the attention mechanism retrieves during generation. A text prefix processed into the KV cache achieves **91% of the detail-transfer performance** of embedding the same content directly in the prompt (10/13 vs 11/13).

### 2. KV Coherence Requires All Layers

Partial KV injection (Conditions C and D) completely failed:
- **Critical-only (C)**: Corrupted into repeated `|im|` tokens — attention misalignment from inconsistent KV state across layers
- **Resilient-only (D)**: Model ignored the prefix entirely, behaving identically to baseline

This demonstrates that the transformer requires consistent key-value entries **across all layers** for attention to function correctly. Layer-selective KV injection through layer skipping alone is insufficient — a more targeted mechanism (e.g., direct KV tensor manipulation) may be required for per-layer control.

### 3. Layer Sensitivity ≠ KV Injection Leverage

The layer skip experiment measures *processing importance* (which computations are essential). The KV injection experiment measures *attention coherence* (which KV entries are required). These are independent properties. A layer can be skippable for processing but essential for KV coherence. This distinction is important for future injection architectures.

---

## Implications & Future Directions

### Near-term: Compressed KV Memory

If full-prefix KV injection transfers 10/13 details, can a *compressed* representation (fewer tokens encoding the same semantics) achieve comparable results? This would reduce the context window cost of KV-based knowledge injection.

### Medium-term: KV-to-ΔW Translation

The most significant implication is a path from **temporary** KV-based knowledge (context window dependent) to **permanent** weight-level knowledge:

1. Process a fact through the model, capturing KV state and layer activations
2. Train a small translator network: `f(KV state) → ΔW` (per-layer weight deltas)
3. Apply the weight deltas as low-rank updates (LoRA-style)
4. The model now "remembers" the fact without any context prefix

This would enable:
- **Two-pass fine-tuning**: One forward pass to capture the knowledge representation, one pass through the translator to compute weight updates
- **Composable knowledge**: Independent ΔW per fact, stackable and removable
- **Model-native knowledge plugins**: Shareable, tiny files (KB-scale) that add specific knowledge to any compatible base model

### Long-term: Learned Memory Formation

A trained KV→ΔW translator would effectively give language models the ability to form *memories* — not through context windows, retrieval augmentation, or conventional fine-tuning, but through the same mechanism the model uses to process information during inference, converted to permanent weight modifications.

---

## Reproducibility

All experiments were conducted on a Windows workstation with dual GPUs (RTX-class), using:
- **llama-cpp-python** (JamePeng fork) with custom C-level layer_skip/layer_capture extensions
- **Qwen3-14B-Claude-4.5-Opus-Distill** (Q4_K_M quantization, 40 layers)
- **Qwen3-32B-Uncensored** (i1-Q4_K_M quantization, 64 layers) — for layer sensitivity only
- **Temperature 0** (deterministic generation) for all conditions
- Scripts: `scripts/layer_skip_prose.py`, `scripts/kv_injection_v2.py`
- Raw results: `scripts/kv_injection_v2_14B.json`, `scripts/prose_14B_results.json`, `scripts/prose_32B_results.json`

---

## Appendix: Experimental Infrastructure

### Custom C API Extensions (llama.cpp fork)

| Function | Purpose |
|----------|---------|
| `llama_set_layer_skip(ctx, mask, n_layers)` | Skip specific layers during inference |
| `llama_set_layer_capture(ctx, mask, n_layers)` | Enable per-layer hidden state capture |
| `llama_get_embeddings_layer_ith(ctx, layer, i)` | Retrieve captured hidden state |
| `llama_get_embeddings_penultimate_ith(ctx, i)` | Pre-norm penultimate layer output |
| `llama_get_n_layer(ctx)` | Get transformer layer count |

### Python API (llama_cpp)

```python
llm = Llama(model_path="model.gguf", n_gpu_layers=-1)

# Skip layers 3, 4, 5 during next eval
llm.set_layer_skip([3, 4, 5])

# Process tokens (KV cache populated only at non-skipped layers)
llm.eval(tokens)

# Re-enable all layers
llm.set_layer_skip(None)
```
