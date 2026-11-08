"""Decode the phone's final latent (out.bin) and the reference's (ref_out.bin, from check_full.py) with the FLUX.2 small decoder on the CPU, side by side.
usage: python decode_cmp.py DUMPDIR W H OUT.jpg [VAEDIR]   (VAEDIR: black-forest-labs/FLUX.2-small-decoder, default vae/)"""
import os, sys, numpy as np, torch
from PIL import Image
from diffusers import AutoencoderKLFlux2
D, Wp, Hp, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]; H, W = Hp // 16, Wp // 16
vae = AutoencoderKLFlux2.from_pretrained(sys.argv[5] if len(sys.argv) > 5 else "vae", torch_dtype=torch.float32).eval()
def dec(name):
    z = torch.from_numpy(np.fromfile(os.path.join(D, name), np.float32)).reshape(1, 128, H, W)
    bn = vae.bn; z = z * (bn.running_var + vae.config.batch_norm_eps).sqrt().view(1, -1, 1, 1) + bn.running_mean.view(1, -1, 1, 1)   # as pipeline_moss_image.py
    z = z.reshape(1, 32, 2, 2, H, W).permute(0, 1, 4, 2, 5, 3).reshape(1, 32, 2 * H, 2 * W)
    with torch.no_grad(): img = vae.decode(z, return_dict=False)[0].clamp(-1, 1)
    return ((img[0].permute(1, 2, 0).numpy() + 1) * 127.5).round().astype(np.uint8)
a, b = dec("out.bin"), dec("ref_out.bin")
print("mean abs pixel difference phone vs reference:", round(float(np.abs(a.astype(np.float32) - b).mean()), 2), "of 255")
c = np.full((Hp, 2 * Wp + 16, 3), 255, np.uint8); c[:, :Wp] = a; c[:, Wp + 16:] = b; Image.fromarray(c).save(out, quality=92)
