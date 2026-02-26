#!/usr/bin/env python3
"""
KV Cache Compression PoC
========================

Can we compress an 80-token KV prefix into 1-5 synthetic entries
that produce the same model behavior?

The idea: take 80 KV entries from a full prefix eval, compute cluster
centroids (k-means), inject those N centroids into a fresh N-token state.
The model then has N KV entries instead of 80, each one a dense
"compressed word" representing multiple original tokens.

Tests:
  1. ROUNDTRIP  — extract KV, re-inject unchanged → verify zero degradation
  2. CENTROID   — average all 80 rows into 1 synthetic KV entry
  3. K-MEANS(N) — cluster 80 rows into N representative entries
  4. TRUNCATED  — first N tokens of prefix (baseline compression curve)
  5. BARE       — no prefix at all
"""
from __future__ import annotations

import argparse
import ctypes
import json
import re
import struct
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from llama_cpp import Llama
from llama_cpp.llama import LlamaState

# ═══════════════════════════════════════════════════════════════════════
#  GGML TYPE MAPPING
# ═══════════════════════════════════════════════════════════════════════

GGML_TYPE_INFO = {
    0: ("F32",  np.float32, 4, 1),   # name, dtype, type_size, block_size
    1: ("F16",  np.float16, 2, 1),
}


# ═══════════════════════════════════════════════════════════════════════
#  STATE BINARY PARSER
# ═══════════════════════════════════════════════════════════════════════

def parse_state(data: bytes) -> dict:
    """Parse llama.cpp binary state format.
    Returns structure with byte offsets for each KV data region."""
    o = 0
    result = {}

    # 1. Arch string: uint32 length + bytes
    str_len = struct.unpack_from('<I', data, o)[0]; o += 4
    arch = data[o:o+str_len].decode('utf-8'); o += str_len
    result['arch'] = arch

    # 2. Output IDs: n_outputs (int32) + output_pos array
    n_outputs = struct.unpack_from('<i', data, o)[0]; o += 4
    if n_outputs > 0:
        o += n_outputs * 4
    result['n_outputs'] = n_outputs

    # 3. Logits: uint64 count + float32 data
    logits_size = struct.unpack_from('<Q', data, o)[0]; o += 8
    if logits_size > 0:
        o += int(logits_size) * 4

    # 4. Embeddings: uint64 count + float32 data
    embd_size = struct.unpack_from('<Q', data, o)[0]; o += 8
    if embd_size > 0:
        o += int(embd_size) * 4

    # 5. KV Cache
    n_stream = struct.unpack_from('<I', data, o)[0]; o += 4
    result['n_stream'] = n_stream
    result['streams'] = []

    for s in range(n_stream):
        stream = {'stream_idx': s}
        cell_count = struct.unpack_from('<I', data, o)[0]; o += 4
        stream['cell_count'] = cell_count

        if cell_count == 0:
            result['streams'].append(stream)
            continue

        # Cell metadata
        stream['cells'] = []
        stream['meta_start'] = o
        for i in range(cell_count):
            pos = struct.unpack_from('<i', data, o)[0]; o += 4
            n_seq = struct.unpack_from('<I', data, o)[0]; o += 4
            sids = []
            for _ in range(n_seq):
                sid = struct.unpack_from('<i', data, o)[0]; o += 4
                sids.append(sid)
            stream['cells'].append({'pos': pos, 'seq_ids': sids})
        stream['meta_end'] = o

        # KV data header
        v_trans = struct.unpack_from('<I', data, o)[0]; o += 4
        n_layer = struct.unpack_from('<I', data, o)[0]; o += 4
        stream['v_trans'] = bool(v_trans)
        stream['n_layer'] = n_layer
        stream['data_header_offset'] = o - 8  # offset of v_trans field

        # Keys (per layer): type(i32) + row_size(u64) + data
        stream['k_layers'] = []
        for l in range(n_layer):
            k_type = struct.unpack_from('<i', data, o)[0]; o += 4
            k_row_sz = struct.unpack_from('<Q', data, o)[0]; o += 8
            k_offset = o
            k_total = cell_count * int(k_row_sz)
            o += k_total
            stream['k_layers'].append({
                'type': k_type, 'row_size': int(k_row_sz),
                'offset': k_offset, 'total_size': k_total,
            })

        # Values (per layer): depends on v_trans
        stream['v_layers'] = []
        if not v_trans:
            for l in range(n_layer):
                v_type = struct.unpack_from('<i', data, o)[0]; o += 4
                v_row_sz = struct.unpack_from('<Q', data, o)[0]; o += 8
                v_offset = o
                v_total = cell_count * int(v_row_sz)
                o += v_total
                stream['v_layers'].append({
                    'type': v_type, 'row_size': int(v_row_sz),
                    'offset': v_offset, 'total_size': v_total,
                    'transposed': False,
                })
        else:
            for l in range(n_layer):
                v_type = struct.unpack_from('<i', data, o)[0]; o += 4
                v_el_sz = struct.unpack_from('<I', data, o)[0]; o += 4
                v_n_embd = struct.unpack_from('<I', data, o)[0]; o += 4
                v_offset = o
                v_total = int(v_n_embd) * cell_count * int(v_el_sz)
                o += v_total
                stream['v_layers'].append({
                    'type': v_type, 'el_size': int(v_el_sz),
                    'n_embd': int(v_n_embd),
                    'offset': v_offset, 'total_size': v_total,
                    'transposed': True,
                })

        result['streams'].append(stream)

    result['parsed_bytes'] = o
    return result


def extract_kv(data: bytes, parsed: dict, stream_idx: int = 0
               ) -> List[Dict[str, np.ndarray]]:
    """Extract per-layer K and V matrices as numpy arrays.
    Returns list of {'k': (cells, k_dim), 'v': (cells, v_dim)} per layer."""
    stream = parsed['streams'][stream_idx]
    cell_count = stream['cell_count']
    layers = []

    for l in range(stream['n_layer']):
        kl = stream['k_layers'][l]
        vl = stream['v_layers'][l]

        # Keys
        k_info = GGML_TYPE_INFO.get(kl['type'])
        if k_info is None:
            raise ValueError(f"Unsupported K type {kl['type']} at layer {l}")
        k_dtype = k_info[1]
        k_raw = data[kl['offset']:kl['offset']+kl['total_size']]
        k_arr = np.frombuffer(k_raw, dtype=k_dtype)
        k_dim = kl['row_size'] // np.dtype(k_dtype).itemsize
        k_mat = k_arr.reshape(cell_count, k_dim).copy()

        # Values
        v_info = GGML_TYPE_INFO.get(vl['type'])
        if v_info is None:
            raise ValueError(f"Unsupported V type {vl['type']} at layer {l}")
        v_dtype = v_info[1]
        v_raw = data[vl['offset']:vl['offset']+vl['total_size']]
        v_arr = np.frombuffer(v_raw, dtype=v_dtype)

        if vl.get('transposed', False):
            v_n_embd = vl['n_embd']
            # Binary layout: (n_embd, cell_count) row-major → transpose
            v_mat = v_arr.reshape(v_n_embd, cell_count).T.copy()
        else:
            v_dim = vl['row_size'] // np.dtype(v_dtype).itemsize
            v_mat = v_arr.reshape(cell_count, v_dim).copy()

        layers.append({'k': k_mat, 'v': v_mat})

    return layers


def inject_kv(template_bytes: bytes, template_parsed: dict,
              compressed_kv: List[Dict[str, np.ndarray]],
              stream_idx: int = 0) -> bytes:
    """Inject compressed KV matrices into template state bytes.
    compressed_kv[l] = {'k': (N, k_dim), 'v': (N, v_dim)}.
    N must equal template's cell_count."""
    buf = bytearray(template_bytes)
    stream = template_parsed['streams'][stream_idx]
    n_new = stream['cell_count']

    for l in range(stream['n_layer']):
        kl = stream['k_layers'][l]
        vl = stream['v_layers'][l]
        ckv = compressed_kv[l]

        # Inject K
        k_info = GGML_TYPE_INFO[kl['type']]
        k_dtype = k_info[1]
        k_data = ckv['k'][:n_new].astype(k_dtype).tobytes()
        assert len(k_data) == kl['total_size'], \
            f"K size mismatch layer {l}: {len(k_data)} vs {kl['total_size']}"
        buf[kl['offset']:kl['offset']+kl['total_size']] = k_data

        # Inject V
        v_info = GGML_TYPE_INFO[vl['type']]
        v_dtype = v_info[1]
        if vl.get('transposed', False):
            # Transpose to (n_embd, N) and write row-major
            v_data = ckv['v'][:n_new].T.astype(v_dtype).tobytes()
        else:
            v_data = ckv['v'][:n_new].astype(v_dtype).tobytes()
        assert len(v_data) == vl['total_size'], \
            f"V size mismatch layer {l}: {len(v_data)} vs {vl['total_size']}"
        buf[vl['offset']:vl['offset']+vl['total_size']] = v_data

    return bytes(buf)


# ═══════════════════════════════════════════════════════════════════════
#  COMPRESSION METHODS
# ═══════════════════════════════════════════════════════════════════════

def compress_centroid(kv_layers: List[Dict[str, np.ndarray]]
                      ) -> List[Dict[str, np.ndarray]]:
    """Average all rows into a single row per layer."""
    result = []
    for layer in kv_layers:
        k_mean = layer['k'].astype(np.float32).mean(axis=0, keepdims=True)
        v_mean = layer['v'].astype(np.float32).mean(axis=0, keepdims=True)
        result.append({
            'k': k_mean.astype(layer['k'].dtype),
            'v': v_mean.astype(layer['v'].dtype),
        })
    return result


def compress_kmeans(kv_layers: List[Dict[str, np.ndarray]], n_clusters: int
                    ) -> List[Dict[str, np.ndarray]]:
    """K-means clustering on concatenated per-cell KV features."""
    from sklearn.cluster import KMeans

    n_layer = len(kv_layers)
    cell_count = kv_layers[0]['k'].shape[0]

    if n_clusters >= cell_count:
        return kv_layers  # No compression needed

    # Build feature matrix: concat all K and V across layers per cell
    features = []
    layer_k_dims = []
    layer_v_dims = []
    for layer in kv_layers:
        features.append(layer['k'].astype(np.float32))
        features.append(layer['v'].astype(np.float32))
        layer_k_dims.append(layer['k'].shape[1])
        layer_v_dims.append(layer['v'].shape[1])
    feature_matrix = np.hstack(features)  # (cell_count, total_dim)

    print(f"      K-means: {cell_count} points × {feature_matrix.shape[1]} dims → {n_clusters} clusters")

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42, max_iter=300)
    km.fit(feature_matrix)
    centroids = km.cluster_centers_  # (n_clusters, total_dim)

    # Split centroids back into per-layer K and V
    result = []
    col = 0
    for l in range(n_layer):
        k_dim = layer_k_dims[l]
        v_dim = layer_v_dims[l]
        k_centroid = centroids[:, col:col+k_dim]
        col += k_dim
        v_centroid = centroids[:, col:col+v_dim]
        col += v_dim
        result.append({
            'k': k_centroid.astype(kv_layers[l]['k'].dtype),
            'v': v_centroid.astype(kv_layers[l]['v'].dtype),
        })
    return result


def compress_kmeans_per_layer(kv_layers: List[Dict[str, np.ndarray]],
                               n_clusters: int
                               ) -> List[Dict[str, np.ndarray]]:
    """K-means per layer independently (loses cross-layer coherence but
    may find better per-layer representatives)."""
    from sklearn.cluster import KMeans

    result = []
    for l, layer in enumerate(kv_layers):
        k_f32 = layer['k'].astype(np.float32)
        v_f32 = layer['v'].astype(np.float32)
        combined = np.hstack([k_f32, v_f32])

        km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
        km.fit(combined)

        k_dim = k_f32.shape[1]
        k_centroid = km.cluster_centers_[:, :k_dim]
        v_centroid = km.cluster_centers_[:, k_dim:]

        result.append({
            'k': k_centroid.astype(layer['k'].dtype),
            'v': v_centroid.astype(layer['v'].dtype),
        })
    return result


# ═══════════════════════════════════════════════════════════════════════
#  MEMORY & TEST SETUP
# ═══════════════════════════════════════════════════════════════════════

ELARA_PREFIX = (
    "Character and setting details for the story:\n"
    "- The woman's name is Elara Voss. She is 34 years old.\n"
    "- She has silver-streaked auburn hair and a crescent-shaped scar on her left palm.\n"
    "- The forest is called Thornwood. It sits in a valley called the Greywander Basin.\n"
    "- It is late November, the first frost has already come.\n"
    "- Her companion is a wolfhound named Cassius with one blue eye and one amber eye.\n"
    "- She carries a brass compass that belonged to her grandmother, Mirabel."
)

ELARA_SHORTENED = (
    "Elara Voss, 34, silver-streaked auburn hair, crescent scar left palm. "
    "Thornwood forest, Greywander Basin, late November, first frost. "
    "Wolfhound Cassius: one blue eye, one amber eye. "
    "Brass compass from grandmother Mirabel."
)

ELARA_PROMPT = (
    "Write a short story about a woman named Elara walking through a forest at dusk. "
    "Three paragraphs. Rich sensory details. /no_think"
)

ELARA_CHECKS = {
    "age_34":        r"\b34\b",
    "silver_hair":   r"silver",
    "auburn_hair":   r"auburn",
    "crescent_scar": r"crescent",
    "thornwood":     r"[Tt]hornwood",
    "greywander":    r"[Gg]reywander",
    "november":      r"[Nn]ovember",
    "frost":         r"frost",
    "wolfhound":     r"wolfhound",
    "cassius":       r"[Cc]assius",
    "heterochromia": r"blue.{1,30}amber|amber.{1,30}blue",
    "brass_compass": r"brass.{1,15}compass|compass.{1,15}brass",
    "mirabel":       r"[Mm]irabel",
}

KV_SEP = "\n---\n"


def wrap_chat(user_msg: str) -> str:
    return (
        "<|im_start|>user\n" + user_msg + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def generate_fresh(llm: Llama, prompt: str, max_tokens: int,
                    prefix: Optional[str] = None) -> str:
    """Generate from scratch with optional prefix."""
    llm._ctx.memory_clear(True)
    llm.reset()
    if prefix:
        prefix_tokens = llm.tokenize(prefix.encode(), add_bos=True)
        llm.eval(prefix_tokens)
        prompt_tokens = llm.tokenize(prompt.encode(), add_bos=False)
    else:
        prompt_tokens = llm.tokenize(prompt.encode(), add_bos=True)
    llm.eval(prompt_tokens)

    output_tokens: list[int] = []
    eos = llm.token_eos()
    for _ in range(max_tokens):
        tok = llm.sample(temp=0.0, top_k=-1, top_p=1.0,
                         repeat_penalty=1.0, penalty_last_n=0)
        if tok == eos:
            break
        output_tokens.append(tok)
        llm.eval([tok])
    return llm.detokenize(output_tokens).decode("utf-8", errors="replace")


def generate_from_state(llm: Llama, state: LlamaState, prompt: str,
                         max_tokens: int) -> str:
    """Load a KV state, eval prompt on top, generate."""
    llm.load_state(state)
    prompt_tokens = llm.tokenize(prompt.encode(), add_bos=False)
    llm.eval(prompt_tokens)

    output_tokens: list[int] = []
    eos = llm.token_eos()
    for _ in range(max_tokens):
        tok = llm.sample(temp=0.0, top_k=-1, top_p=1.0,
                         repeat_penalty=1.0, penalty_last_n=0)
        if tok == eos:
            break
        output_tokens.append(tok)
        llm.eval([tok])
    return llm.detokenize(output_tokens).decode("utf-8", errors="replace")


def score(text: str, checks: Dict[str, str]) -> Tuple[int, Dict[str, bool]]:
    results = {k: bool(re.search(p, text)) for k, p in checks.items()}
    return sum(results.values()), results


def create_template_state(llm: Llama, n_tokens: int) -> LlamaState:
    """Create a state with exactly n_tokens KV entries using dummy tokens."""
    llm._ctx.memory_clear(True)
    llm.reset()
    # Use BOS + dummy tokens to fill N positions
    dummy = llm.tokenize((" test" * max(1, n_tokens)).encode(), add_bos=True)
    dummy = dummy[:n_tokens]
    if len(dummy) < n_tokens:
        # Pad with the last token
        dummy = dummy + [dummy[-1]] * (n_tokens - len(dummy))
    llm.eval(dummy)
    return llm.save_state()


# ═══════════════════════════════════════════════════════════════════════
#  MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="KV Cache Compression PoC")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--main-gpu", type=int, default=0)
    ap.add_argument("--single-gpu", action="store_true")
    ap.add_argument("--output", default="scripts/kv_compression_results.json")
    args = ap.parse_args()

    extra = {}
    if args.single_gpu:
        extra["tensor_split"] = [1.0, 0.0]

    print("Loading model...")
    llm = Llama(
        model_path=args.model,
        n_gpu_layers=-1,
        n_ctx=args.n_ctx,
        main_gpu=args.main_gpu,
        verbose=False,
        **extra,
    )

    chat_prompt = wrap_chat(ELARA_PROMPT)
    full_prefix = ELARA_PREFIX + KV_SEP
    results = {}

    # ═══════════════════════════════════════════════════════════════
    #  STEP 1: Create full prefix state and parse it
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*72}")
    print(f"  STEP 1: Full Prefix State Capture")
    print(f"{'═'*72}")

    full_tokens = llm.tokenize(full_prefix.encode(), add_bos=True)
    n_full = len(full_tokens)
    print(f"  Full prefix: {n_full} tokens")

    llm._ctx.memory_clear(True)
    llm.reset()
    llm.eval(full_tokens)
    full_state = llm.save_state()

    # Parse and extract KV
    full_parsed = parse_state(full_state.llama_state)
    full_stream = full_parsed['streams'][0]
    print(f"  Arch: {full_parsed['arch']}")
    print(f"  State: {len(full_state.llama_state):,} bytes")
    print(f"  Cells: {full_stream['cell_count']}")
    print(f"  Layers: {full_stream['n_layer']}")
    print(f"  V transposed: {full_stream['v_trans']}")
    print(f"  K type: {GGML_TYPE_INFO.get(full_stream['k_layers'][0]['type'], ('?',))[0]}")
    print(f"  V type: {GGML_TYPE_INFO.get(full_stream['v_layers'][0]['type'], ('?',))[0]}")
    print(f"  K row size: {full_stream['k_layers'][0]['row_size']} bytes "
          f"({full_stream['k_layers'][0]['row_size']//2} dims for F16)")
    if full_stream['v_trans']:
        print(f"  V embd: {full_stream['v_layers'][0]['n_embd']} dims")
    else:
        print(f"  V row size: {full_stream['v_layers'][0]['row_size']} bytes")

    full_kv = extract_kv(full_state.llama_state, full_parsed)
    print(f"  Extracted: {len(full_kv)} layers, "
          f"K shape {full_kv[0]['k'].shape}, V shape {full_kv[0]['v'].shape}")

    total_kv_bytes = sum(l['total_size'] for l in full_stream['k_layers']) + \
                     sum(l['total_size'] for l in full_stream['v_layers'])
    print(f"  Total KV data: {total_kv_bytes:,} bytes ({total_kv_bytes/1024/1024:.1f} MB)")

    # ═══════════════════════════════════════════════════════════════
    #  STEP 2: Roundtrip sanity check
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*72}")
    print(f"  STEP 2: Roundtrip Sanity Check")
    print(f"{'═'*72}")

    # Extract and re-inject the same KV values
    roundtrip_bytes = inject_kv(full_state.llama_state, full_parsed, full_kv)
    assert roundtrip_bytes == full_state.llama_state, "Roundtrip failed! Bytes differ."
    print(f"  ✓ Extract → Inject roundtrip: bytes identical")

    # Test recall from original state
    print(f"  Testing recall from original state...")
    orig_text = generate_from_state(llm, full_state, chat_prompt, args.max_tokens)
    orig_n, orig_res = score(orig_text, ELARA_CHECKS)
    print(f"  ✓ Original state recall: {orig_n}/13")
    results['full_state'] = {'score': orig_n, 'tokens': n_full}

    # ═══════════════════════════════════════════════════════════════
    #  STEP 3: Baselines
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*72}")
    print(f"  STEP 3: Baselines")
    print(f"{'═'*72}")

    # Bare
    print(f"  [bare] No prefix...")
    bare_text = generate_fresh(llm, chat_prompt, args.max_tokens)
    bare_n, bare_res = score(bare_text, ELARA_CHECKS)
    print(f"    → {bare_n}/13")
    results['bare'] = {'score': bare_n, 'tokens': 0}

    # Full prefix (fresh eval, not from state)
    print(f"  [full] Full prefix ({n_full} tokens)...")
    full_text = generate_fresh(llm, chat_prompt, args.max_tokens, prefix=full_prefix)
    full_n, full_res = score(full_text, ELARA_CHECKS)
    print(f"    → {full_n}/13")
    results['full_prefix'] = {'score': full_n, 'tokens': n_full}

    # Shortened text
    short_tokens = llm.tokenize((ELARA_SHORTENED + KV_SEP).encode(), add_bos=True)
    print(f"  [shortened] Compressed text ({len(short_tokens)} tokens)...")
    short_text = generate_fresh(llm, chat_prompt, args.max_tokens,
                                 prefix=ELARA_SHORTENED + KV_SEP)
    short_n, short_res = score(short_text, ELARA_CHECKS)
    print(f"    → {short_n}/13")
    results['shortened_text'] = {'score': short_n, 'tokens': len(short_tokens)}

    # ═══════════════════════════════════════════════════════════════
    #  STEP 4: Truncation baseline curve
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*72}")
    print(f"  STEP 4: Truncation Baselines")
    print(f"{'═'*72}")

    for trunc_n in [5, 10, 20, 40]:
        if trunc_n >= n_full:
            continue
        trunc_tokens = full_tokens[:trunc_n]
        trunc_text_prefix = llm.detokenize(trunc_tokens).decode("utf-8", errors="replace")

        llm._ctx.memory_clear(True)
        llm.reset()
        llm.eval(trunc_tokens)
        trunc_state = llm.save_state()

        trunc_text = generate_from_state(llm, trunc_state, chat_prompt, args.max_tokens)
        trunc_score, _ = score(trunc_text, ELARA_CHECKS)
        print(f"  [trunc-{trunc_n}] First {trunc_n} tokens → {trunc_score}/13")
        results[f'truncated_{trunc_n}'] = {'score': trunc_score, 'tokens': trunc_n}

    # ═══════════════════════════════════════════════════════════════
    #  STEP 5: KV Compression — Centroid (N=1)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*72}")
    print(f"  STEP 5: KV Compression")
    print(f"{'═'*72}")

    compression_configs = [
        (1, "centroid"),
        (3, "kmeans"),
        (5, "kmeans"),
        (10, "kmeans"),
        (20, "kmeans"),
    ]

    for n_compressed, method in compression_configs:
        if n_compressed >= n_full:
            continue

        label = f"{method}_{n_compressed}"
        ratio = n_full / n_compressed
        kv_bytes_compressed = total_kv_bytes * n_compressed // full_stream['cell_count']

        print(f"\n  [{label}] {n_full} → {n_compressed} tokens "
              f"({ratio:.0f}× compression, {kv_bytes_compressed/1024:.0f} KB)...")

        # Create template with N KV entries
        template = create_template_state(llm, n_compressed)
        template_parsed = parse_state(template.llama_state)
        t_stream = template_parsed['streams'][0]

        actual_n = t_stream['cell_count']
        if actual_n != n_compressed:
            print(f"    WARNING: template has {actual_n} cells, expected {n_compressed}")

        # Compute compressed KV
        if method == "centroid":
            compressed_kv = compress_centroid(full_kv)
        elif method == "kmeans":
            compressed_kv = compress_kmeans(full_kv, n_compressed)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Verify shapes match template
        assert compressed_kv[0]['k'].shape[0] == actual_n, \
            f"Shape mismatch: compressed has {compressed_kv[0]['k'].shape[0]} rows, " \
            f"template expects {actual_n}"

        # Inject into template
        modified_bytes = inject_kv(template.llama_state, template_parsed, compressed_kv)
        compressed_state = LlamaState(
            input_ids=template.input_ids.copy(),
            scores=template.scores.copy(),
            n_tokens=template.n_tokens,
            llama_state=modified_bytes,
            llama_state_size=template.llama_state_size,
            seed=template.seed,
        )

        # Test
        try:
            comp_text = generate_from_state(llm, compressed_state, chat_prompt,
                                             args.max_tokens)
            comp_n, comp_res = score(comp_text, ELARA_CHECKS)
            print(f"    → {comp_n}/13 details recalled")

            # Show which details
            hits = [k for k, v in comp_res.items() if v]
            if hits:
                print(f"    ✓ {', '.join(hits)}")

            results[label] = {
                'score': comp_n,
                'tokens': n_compressed,
                'ratio': ratio,
                'kv_bytes': kv_bytes_compressed,
                'details': {k: v for k, v in comp_res.items()},
            }
        except Exception as e:
            print(f"    ✗ Generation failed: {e}")
            results[label] = {'score': -1, 'tokens': n_compressed, 'error': str(e)}

    # Also test per-layer k-means for comparison
    for n_compressed in [5, 10]:
        label = f"kmeans_perlayer_{n_compressed}"
        print(f"\n  [{label}] Per-layer k-means, {n_compressed} tokens...")

        template = create_template_state(llm, n_compressed)
        template_parsed = parse_state(template.llama_state)
        actual_n = template_parsed['streams'][0]['cell_count']

        compressed_kv = compress_kmeans_per_layer(full_kv, n_compressed)

        modified_bytes = inject_kv(template.llama_state, template_parsed, compressed_kv)
        compressed_state = LlamaState(
            input_ids=template.input_ids.copy(),
            scores=template.scores.copy(),
            n_tokens=template.n_tokens,
            llama_state=modified_bytes,
            llama_state_size=template.llama_state_size,
            seed=template.seed,
        )

        try:
            comp_text = generate_from_state(llm, compressed_state, chat_prompt,
                                             args.max_tokens)
            comp_n, comp_res = score(comp_text, ELARA_CHECKS)
            print(f"    → {comp_n}/13 details recalled")
            hits = [k for k, v in comp_res.items() if v]
            if hits:
                print(f"    ✓ {', '.join(hits)}")
            results[label] = {'score': comp_n, 'tokens': n_compressed}
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            results[label] = {'score': -1, 'error': str(e)}

    # ═══════════════════════════════════════════════════════════════
    #  SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═'*72}")
    print(f"  COMPRESSION RESULTS SUMMARY")
    print(f"{'═'*72}")

    # Sort by tokens
    sorted_results = sorted(results.items(),
                             key=lambda x: x[1].get('tokens', 0))

    print(f"\n  {'Method':<25} {'Tokens':>8} {'Score':>8} {'Ratio':>8}")
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8}")
    for key, r in sorted_results:
        tokens = r.get('tokens', '?')
        sc = r.get('score', '?')
        ratio = f"{n_full/tokens:.0f}×" if isinstance(tokens, int) and tokens > 0 else '-'
        print(f"  {key:<25} {tokens:>8} {f'{sc}/13':>8} {ratio:>8}")

    # Save
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {args.output}")

    del llm


if __name__ == "__main__":
    main()
