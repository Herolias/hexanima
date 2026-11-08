"""Qwen3.5-9B GGUF -> te_packs/te00..te31.bin (+ te32.bin: final norm) for mossrun.
usage: python export_te_all.py FIRST LAST     env TE_GGUF (default models/Qwen3.5-9B-UD-Q4_K_XL.gguf), TE_OUT (default te_packs)
Q4_K tensors become q4a tiles, every other type q8 tiles. From the Q8_0 GGUF every matrix is an exact q8 repack: the 4-bit encoder's text embedding
is ~5% off the 8-bit one (moss/te_cmp.py), which CFG amplifies in the full-model mode; the 8-bit layers still fit TE_BUF (largest 246 MB)."""
import os, sys, numpy as np
from export_te import pack_te_layer, Q
q = Q(os.environ.get("TE_GGUF", "models/Qwen3.5-9B-UD-Q4_K_XL.gguf")); out = os.environ.get("TE_OUT", "te_packs"); os.makedirs(out, exist_ok=True)
for i in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
    b, info = pack_te_layer(q, i); open(f"{out}/te{i:02d}.bin", "wb").write(b); print(i, len(b), flush=True)
    assert len(b) <= 262144000, "layer pack larger than mossrun's TE_BUF"
if int(sys.argv[2]) >= 31:
    w = q("output_norm.weight").numpy().astype(np.float32); open(f"{out}/te32.bin", "wb").write(w.tobytes() + bytes(16384 - w.nbytes)); print("final")
    t = q.t["token_embd.weight"]; name = {"Q8_0": "embd.q8", "Q4_K": "embd.q4k"}[t.tensor_type.name]   # raw GGUF rows, mossrun dequantizes the prompt's rows
    with open(f"{out}/{name}", "wb") as f:   # in 8192-row chunks: one tobytes() of the whole 1 GB table doubles the peak RAM
        for r in range(0, t.data.shape[0], 8192): f.write(np.ascontiguousarray(t.data[r:r + 8192]).tobytes())
    print(name)
