"""MOSS-Image full model on a desktop GPU (bf16 weights, like the desktop pipelines): sampler-settings study against the phone.
usage: python sampler_gpu.py OUTDIR TX TXN W H SEED CFG RUN [RUN ...]      env MOSS_GGUF (default models/MOSS-Image-SFT-Q8_0.gguf), Z (noise file, e.g. a MOSS_DUMP z.bin)
TX / TXN: refined text rows, mossrun's tx.cache format or a MOSS_DUMP tx.bin / tn.bin.
RUN = KIND:STEPS[:SHIFT][:sgm][:noskip]   KIND euler | 2m (DPM-Solver++ 2M); sgm = Unsloth Studio's sgm_uniform schedule (default: the reference training-grid
      schedule); noskip = no block skipped in the negative branch (Studio on SFT; default: block 22 as pipeline_moss_image.py). Studio = euler:20:sgm:noskip.
Writes OUTDIR/<run>.lat (final latent [128,H/16,W/16] fp32, before BN denorm, for decode_cmp-style decoding).  Needs a CUDA torch and ~10 GB of VRAM."""
import os, sys, time, struct, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moss_ref as M
from moss_ref import rms, lin, block, rope_tables, time_vecs, NL
OUT, txf, tnf, Wp, Hp, seed, cfg = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), float(sys.argv[7]); runs = sys.argv[8:]
os.makedirs(OUT, exist_ok=True); H, Wd = Hp // 16, Wp // 16; dev, dt = "cuda", torch.bfloat16
torch.set_grad_enabled(False)
src = M.W(os.environ.get("MOSS_GGUF", "models/MOSS-Image-SFT-Q8_0.gguf")); cpu = {}; gpu = {}
def gc(n):
    if n not in cpu: cpu[n] = src(n)
    return cpu[n]
t0 = time.time()
for n in src.t:   # every DiT weight to the GPU once, bf16
    if n.startswith(("layers.", "video_embedder", "final_")): gpu[n] = src(n).to(dev, dt)
print(f"weights on GPU in {time.time() - t0:.0f} s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
g = lambda n: gpu[n]
def load_cache(p):
    if p.endswith(".bin"): return torch.from_numpy(np.fromfile(p, np.float32).reshape(-1, 3072))   # mossrun MOSS_DUMP tx.bin / tn.bin
    b = open(p, "rb").read(); h, T = struct.unpack_from("<Qi", b, 0); return torch.from_numpy(np.frombuffer(b, np.float32, T * 3072, 12).reshape(T, 3072).copy())
tx, tn = load_cache(txf).to(dev, dt), load_cache(tnf).to(dev, dt)
ropes = {}
def forward(lat, t, txt, skip=()):
    T = txt.shape[0]
    if T not in ropes: ropes[T] = [a.to(dev, dt) for a in rope_tables(H, Wd, T, gc("rope.bands"))]
    sin, cos = ropes[T]
    x = torch.cat((lin(lat.flatten(1).T.to(dt), g("video_embedder.weight"), g("video_embedder.bias")), txt), 0)
    mod, head = time_vecs(t, gc); mod, head = mod.to(dev, dt), head.to(dev, dt)
    for i in range(NL):
        if i not in skip: x = block(x, mod, sin, cos, g, i)
    x = rms(x[:H * Wd], g("final_norm_video.weight")) * (1 + head[1]) + head[0]
    return lin(x, g("final_linear_video.weight")).T.reshape(128, H, Wd).float()
def sigmas(steps, shift):   # MOSSImageScheduler (same as mossrun moss_sigmas)
    total = 1000; grid = lambda c, sh: (lambda v: sh * v / (1 + (sh - 1) * v))(np.linspace(np.float32(1), np.float32(sh / (total + sh - 1)), c, dtype=np.float32))
    sg, tr = grid(steps, np.float32(shift)), grid(total, np.float32(1.0))[::-1]; out = []
    for s in sg:
        up = min(int(np.searchsorted(tr, s)), total - 1); lo = max(0, up - 1); v = tr[lo] if abs(s - tr[lo]) <= abs(tr[up] - s) else tr[up]
        if not out or out[-1] != v: out.append(float(v))
    return out
def sgm_uniform(steps, shift):   # Unsloth Studio default (ComfyUI sgm_uniform over ModelSamplingDiscreteFlow), terminal zero appended by the loop
    sh = lambda t: shift * t / (1 + (shift - 1) * t); grid = sh(torch.arange(1, 1001, 1) / 1000); ts = torch.linspace(grid[-1], grid[0], steps + 1)[:-1]
    return [float(sh(t)) for t in ts]
SKIP = {22}
def vel(x, s):
    t = float(np.float32(s) * np.float32(1000)); vc = forward(x, t, tx)
    if cfg == 1: return vc
    vu = forward(x, t, tn, skip=SKIP); return vu + cfg * (vc - vu)
noise = (torch.from_numpy(np.fromfile(os.environ["Z"], np.float32).reshape(128, H, Wd)) if os.environ.get("Z") else torch.randn(128, H, Wd, generator=torch.Generator().manual_seed(seed))).to(dev)   # Z: the phone's noise (MOSS_DUMP z.bin)
for run in runs:
    kind, steps, *opt = run.split(":"); shift = next((float(o) for o in opt if o[0].isdigit()), 5.0)
    sg = (sgm_uniform(int(steps), shift) if "sgm" in opt else sigmas(int(steps), shift)) + [0.0]; SKIP = set() if "noskip" in opt else {22}
    x = noise.clone(); prev = None; t0 = time.time()
    for i in range(len(sg) - 1):
        s, sn = sg[i], sg[i + 1]; v = vel(x, s)
        if kind == "euler" or sn == 0.0:
            x = x + (sn - s) * v; continue
        D = x - s * v   # x0 estimate; DPM-Solver++(2M) in data prediction for rectified flow (alpha = 1 - sigma)
        lam = lambda q: np.log((1 - q) / q); h = lam(sn) - lam(s)
        if prev is not None:
            r = (lam(s) - lam(prev[0])) / h; Dp = (1 + 1 / (2 * r)) * D - (1 / (2 * r)) * prev[1]
        else: Dp = D
        x = (sn / s) * x - (1 - sn) * np.expm1(-h) * Dp; prev = (s, D)
    torch.cuda.synchronize(); print(f"{run}: {len(sg) - 1} steps in {time.time() - t0:.0f} s", flush=True)
    x.cpu().numpy().astype(np.float32).tofile(f"{OUT}/{run.replace(':', '_')}.lat")
