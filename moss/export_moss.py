"""Pack MOSS-Image GGUF (UD-Q5_K_XL) + rapid-moss LoRA into HMX q8 tile files, one per block.
blkNN.bin layout (all weight records are q8 tiles: [nb][kb] x 1152 B = 128 B scales + 1024 B offset-binary int8, see hexagon/mega/export_all.py):
  W[q,k,v,o]      [K=3072 + 128 lora rows][N=3072]
  W[up_gate]      [3072+128][16384] with n-blocks interleaved (up_j, gate_j)
  W[down]         [8192+128][3072]
  A[qkv]          [3072][384] (lora_down^T of q,k,v), A[o] [3072][128], A[up_gate] [3072][128], A[down] [8192][128]
  gain            f32: q_gain/sqrt(128) (128), k_gain (128), norm1 w (3072), norm2 w (3072), pad to 8192 floats
usage: python export_moss.py OUTDIR [first last]"""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moss_ref import W, Lora
from export_all_q8 import wt16, q8tiles   # copy of the tile helpers

def mat(w_oi, lora=None, interleave=False):
    """w_oi [out,in] fp32 -> [K(+128), N] tile bytes"""
    w = w_oi.T.contiguous()                                        # [K,N]
    if interleave:
        K, N = w.shape; h = N // 2
        w = torch.stack((w[:, :h].reshape(K, h // 32, 32), w[:, h:].reshape(K, h // 32, 32)), 2).reshape(K, N)   # (up block j, gate block j)
    if lora is not None:
        A, B, s = lora                                             # A [128,in], B [out,128]
        b = (s * B.T).contiguous()                                 # [128,N]
        if interleave:
            h = b.shape[1] // 2
            b = torch.stack((b[:, :h].reshape(128, h // 32, 32), b[:, h:].reshape(128, h // 32, 32)), 2).reshape(128, -1)
        w = torch.cat((w, b), 0)
    return q8tiles(wt16(w.numpy()))

def pack_layer(g, lo, i):
    p = f"layers.{i}."; L = lambda n: lo(i, n)
    out = []; sizes = {}
    def add(name, b): sizes[name] = len(b); out.append(b)
    for n in ("q", "k", "v", "o"): add(n, mat(g(p + f"proj.{n}.weight"), L(n)))
    add("up_gate", mat(g(p + "proj.ffn_up_gate.weight"), L("ffn_up_gate"), True))
    add("down", mat(g(p + "proj.ffn_down.weight"), L("ffn_down")))
    At = lambda n: lo.f.get_tensor(f"diffusion_model.layers.{i}.proj.{n}.lora_down.weight").T      # [in,128]
    add("A_qkv", q8tiles(wt16(torch.cat((At("q"), At("k"), At("v")), 1).numpy())))
    for n in ("o", "ffn_up_gate", "ffn_down"): add("A_" + n, q8tiles(wt16(At(n).contiguous().numpy())))
    gain = np.concatenate([g(p + "q_norm.weight").numpy() / np.sqrt(128.0), g(p + "k_norm.weight").numpy(), g(p + "norm1.weight").numpy(), g(p + "norm2.weight").numpy()]).astype(np.float32)
    add("gain", gain.tobytes() + bytes(8192 * 4 - gain.nbytes))
    return b"".join(out), sizes

def pack_refiner(g, i):
    """text_embedder.blocks.i -> same layout as pack_layer; biases ride in the 4 extra K tiles (row 0), A matrices are zeros (unused in mode 1)"""
    p = f"text_embedder.blocks.{i}."
    def bias_lora(b):   # fake lora tuple (A dummy, B [N,128] with col 0 = bias, scale 1)
        B = torch.zeros(b.shape[0], 128); B[:, 0] = b; return (None, B, 1.0)
    qkv = g(p + "qkv.weight"); qkvb = g(p + "qkv.bias")
    out = []; sizes = {}
    def add(name, b): sizes[name] = len(b); out.append(b)
    for j, n in enumerate(("q", "k", "v")): add(n, mat(qkv[j * 3072:(j + 1) * 3072], bias_lora(qkvb[j * 3072:(j + 1) * 3072])))
    add("o", mat(g(p + "proj.weight"), bias_lora(g(p + "proj.bias"))))
    ug = torch.cat((g(p + "ffn_up.weight"), g(p + "ffn_gate.weight")), 0); ugb = torch.cat((g(p + "ffn_up.bias"), g(p + "ffn_gate.bias")))
    add("up_gate", mat(ug, bias_lora(ugb), True))
    add("down", mat(g(p + "ffn_down.weight"), bias_lora(g(p + "ffn_down.bias"))))
    add("A", bytes(1327104 + 442368 + 442368 + 1179648))
    gain = np.concatenate([g(p + "q_norm.weight").numpy() / np.sqrt(128.0), g(p + "k_norm.weight").numpy(), g(p + "norm1.weight").numpy(), g(p + "norm2.weight").numpy()]).astype(np.float32)
    add("gain", gain.tobytes() + bytes(8192 * 4 - gain.nbytes))
    return b"".join(out), sizes

def pack_proj(g):
    """text_embedder.proj (4096 -> 3072) + bias as [96 nb][132 kb] q8 tiles, padded to the block pack size"""
    w = g("text_embedder.proj.weight"); b = g("text_embedder.proj.bias"); B = torch.zeros(3072, 128); B[:, 0] = b
    d = mat(w, (None, B, 1.0)); assert len(d) == 96 * 132 * 1152, len(d)
    return d + bytes(135397376 - len(d))

if __name__ == "__main__":   # env MOSS_GGUF: source GGUF (Q8_0 repacks exactly: per-32 scales = the tile scales); MOSS_REFDIR: also write the text refiner (proj, ref0, ref1)
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    g = W(os.environ.get("MOSS_GGUF", "models/MOSS-Image-SFT-UD-Q5_K_XL.gguf")); lo = Lora("models/lora.safetensors")
    rd = os.environ.get("MOSS_REFDIR")
    if rd:
        os.makedirs(rd, exist_ok=True); open(f"{rd}/proj.bin", "wb").write(pack_proj(g))
        for i in range(2): open(f"{rd}/ref{i}.bin", "wb").write(pack_refiner(g, i)[0])
        print("refiner", flush=True)
    first = int(sys.argv[2]) if len(sys.argv) > 2 else 0; last = int(sys.argv[3]) if len(sys.argv) > 3 else 41
    for i in range(first, last + 1):
        b, sz = pack_layer(g, lo, i); open(f"{out}/blk{i:02d}.bin", "wb").write(b); print(i, len(b), flush=True)
    print(sz)
