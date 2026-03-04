"""Quick test: does HF Transformers dequantize GGUF or keep it compact?"""
import torch, gc, os, psutil

proc = psutil.Process()

def mem_mb():
    return proc.memory_info().rss / 1024 / 1024

print(f"Baseline RSS: {mem_mb():.0f} MB")
print(f"torch version: {torch.__version__}")

from transformers import AutoModelForCausalLM
print("transformers imported")

gguf_path = r"D:\AI\SD Models\text_encoders\Qwen3-4B.i1-Q4_K_M.gguf"
gguf_size_mb = os.path.getsize(gguf_path) / 1024 / 1024
print(f"GGUF file size: {gguf_size_mb:.1f} MB")

mem_before = mem_mb()
print(f"Before load RSS: {mem_before:.0f} MB")

try:
    model = AutoModelForCausalLM.from_pretrained(
        os.path.dirname(gguf_path),
        gguf_file=os.path.basename(gguf_path),
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    mem_after = mem_mb()
    model_mem = mem_after - mem_before
    print(f"After load RSS: {mem_after:.0f} MB")
    print(f"Memory used by model: {model_mem:.0f} MB")

    # Check parameter dtypes
    dtypes = {}
    total_params = 0
    for name, param in model.named_parameters():
        dt = str(param.dtype)
        if dt not in dtypes:
            dtypes[dt] = {"count": 0, "numel": 0, "example": name}
        dtypes[dt]["count"] += 1
        dtypes[dt]["numel"] += param.numel()
        total_params += param.numel()

    print(f"\nTotal parameters: {total_params:,}")
    print("Parameter dtypes:")
    for dt, info in dtypes.items():
        if "16" in dt:
            bpp = 2
        elif "32" in dt:
            bpp = 4
        else:
            bpp = 1
        bytes_est = info["numel"] * bpp
        print(f"  {dt}: {info['count']} tensors, {info['numel']:,} params, ~{bytes_est/1024/1024:.0f} MB  (e.g. {info['example']})")

    ratio = model_mem / gguf_size_mb
    print(f"\nMemory / GGUF size ratio: {ratio:.2f}x")
    if ratio > 2.0:
        print("VERDICT: DEQUANTIZED to full precision. HF loader unpacks Q4_K_M -> fp16/fp32.")
        print("This defeats the purpose of GGUF for VRAM savings.")
    elif ratio > 1.2:
        print("VERDICT: Partially dequantized. Some overhead but not full expansion.")
    else:
        print("VERDICT: Kept near quantized size. GGUF savings preserved.")

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
