"""Check mossrun's full-model mode against the fp32 torch reference (moss_ref.py, which mirrors pipeline_moss_image.py).
On the phone: MOSS_DUMP=dir in files/moss/tune.env, one picture in Full mode; mossrun then writes z.bin (noise), tx.bin / tn.bin (refined prompt / negative text),
sig.bin (schedule), vc0.bin / vu0.bin (first-step velocities of both branches) and out.bin (final latent, before BN denorm).
usage: python check_full.py DUMPDIR W H CFG [GGUF]      (all blocks are dequantized to fp32 once: ~19 GB RAM; run it under systemd-run -p MemoryMax=...)
Prints the relative L2 error of both first-step velocities and of the final latent after the whole schedule."""
import os, sys, time, numpy as np, torch
from moss_ref import W, block, rope_tables, time_vecs, rms, lin, NL
D, Wd, H, cfg = sys.argv[1], int(sys.argv[2]) // 16, int(sys.argv[3]) // 16, float(sys.argv[4])
src = W(sys.argv[5] if len(sys.argv) > 5 else os.environ.get("MOSS_GGUF", "models/MOSS-Image-SFT-Q8_0.gguf")); cache = {}
def g(n):
    if n not in cache: cache[n] = src(n)
    return cache[n]
ld = lambda n, *shape: torch.from_numpy(np.fromfile(os.path.join(D, n), np.float32)).reshape(*shape)
z = ld("z.bin", 128, H, Wd); tx = ld("tx.bin", -1, 3072); tn = ld("tn.bin", -1, 3072) if os.path.exists(os.path.join(D, "tn.bin")) else None
sig = np.fromfile(os.path.join(D, "sig.bin"), np.float32)
def forward(lat, t, txt, skip=()):   # = MOSSImageTransformer.forward on refined text rows (the dump holds the text refiner's output)
    T = txt.shape[0]; sin, cos = rope_tables(H, Wd, T, g("rope.bands"))
    x = torch.cat((lin(lat.flatten(1).T, g("video_embedder.weight"), g("video_embedder.bias")), txt), 0)
    mod, head = time_vecs(t, g)   # tp=None: the SFT model's own time MLP and modulation biases
    for i in range(NL):
        if i not in skip: x = block(x, mod, sin, cos, g, i)
    x = rms(x[:H * Wd], g("final_norm_video.weight")) * (1 + head[1]) + head[0]
    return lin(x, g("final_linear_video.weight")).T.reshape(128, H, Wd)
rel = lambda a, b: float((a - b).norm() / b.norm())
cfgon = cfg != 1 and tn is not None
torch.set_grad_enabled(False); x = z.clone(); t0 = time.time()
for s, sg in enumerate(sig):
    t = float(np.float32(sg) * np.float32(1000)); sn = float(sig[s + 1]) if s + 1 < len(sig) else 0.0
    vc = forward(x, t, tx); vu = forward(x, t, tn, skip={22}) if cfgon else None
    if s == 0:
        print(f"step 0 (t {t:.0f}): cond velocity rel err {rel(ld('vc0.bin', 128, H, Wd), vc):.4%}" + (f", negative branch {rel(ld('vu0.bin', 128, H, Wd), vu):.4%}" if cfgon else ""), flush=True)
    v = vu + cfg * (vc - vu) if cfgon else vc
    x = x + (np.float32(sn) - np.float32(sg)) * v
    print(f"  step {s + 1}/{len(sig)} done ({time.time() - t0:.0f} s)", flush=True)
out = ld("out.bin", 128, H, Wd)
print(f"final latent after {len(sig)} steps: rel err {rel(out, x):.4%}  (phone std {out.std():.3f}, reference std {x.std():.3f})")
