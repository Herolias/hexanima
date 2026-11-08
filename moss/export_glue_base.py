"""Base-model time embedding for mossrun's full-model mode -> glue_base.bin (same "MGL1" format as export_glue.py).
glue.bin carries the rapid-moss adapter's time MLP and modulation biases (4-step sampler); the full mode (adapter off, Euler, CFG) needs the SFT model's own.
usage: python export_glue_base.py [GGUF] [OUT]   (GGUF default $MOSS_GGUF or models/MOSS-Image-SFT-Q8_0.gguf; these tensors are BF16/F32 in every unsloth quant)
Install: files/moss/glue_base.bin"""
import os, struct, sys, numpy as np
from moss_ref import W
g = W(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MOSS_GGUF", "models/MOSS-Image-SFT-Q8_0.gguf")); out = sys.argv[2] if len(sys.argv) > 2 else "glue_base.bin"
T = [(n, 0, g(n).float().contiguous().numpy().astype(np.float32)) for n in (
    "time_embedding.0.weight", "time_embedding.0.bias", "time_embedding.2.weight", "time_embedding.2.bias", "shared_block_modulation.bias", "video_head_modulation.bias")]
hdr = 8 + len(T) * 76; off = (hdr + 127) // 128 * 128; ent = b""; data = []
for name, dt, a in T:
    d = list(a.shape) + [1] * (3 - a.ndim)
    ent += struct.pack("<40sii3iqq", name.encode(), dt, a.ndim, *d, off, a.nbytes); data.append((off, a.tobytes())); off = (off + a.nbytes + 127) // 128 * 128
with open(out, "wb") as f:
    f.write(b"MGL1" + struct.pack("<i", len(T)) + ent)
    for o, b in data: f.seek(o); f.write(b)
print(out, off, "bytes,", len(T), "tensors")
