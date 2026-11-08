"""moss_ref.py vs the official MOSSImageTransformer (pipeline_moss_image.py) on the same GGUF weights and the same random inputs.
usage: python official_cmp.py [GGUF]"""
import os, sys, math, numpy as np, torch
from torch import nn
from torch.nn import functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_moss_image.py")).read()
exec(src[src.index("class _RMSNorm"):src.index("class MOSSImageScheduler")])   # the official model classes, unchanged
import moss_ref
from moss_ref import W
g0 = W(sys.argv[1] if len(sys.argv) > 1 else "models/MOSS-Image-SFT-Q8_0.gguf"); cache = {}
def g(n):
    if n not in cache: cache[n] = g0(n).contiguous()
    return cache[n]
torch.set_grad_enabled(False)
with torch.device("meta"): m = MOSSImageTransformer()
sd = {k: g(k) for k in m.state_dict().keys()}
m.load_state_dict(sd, strict=True, assign=True); m.rope.bands = m.rope._default_bands(torch.device("cpu")); m.eval()
torch.manual_seed(0); H = W_ = 16; T = 24
lat = torch.randn(128, H, W_); txt = torch.randn(T, 4096) * 3
for t in (1000.0, 871.0, 463.0, 24.0):
    ref = moss_ref.dit_forward(lat, t, txt, g)
    off = m(lat[None], t, txt[None], return_dict=False)[0][0]
    offs = m(lat[None], t, txt[None], return_dict=False, skip_layers=[22])[0][0]
    print(f"t {t:6.1f}: |moss_ref - official| / |official| = {float((ref - off).norm() / off.norm()):.2e}   (skip-22 vs full: {float((offs - off).norm() / off.norm()):.2%})", flush=True)
