"""Static single-frame TAEHV decoder for QNN: z [1,16,96,64] -> [1,3,768,512]. Frame 0/1 of the last two layers are kept as separate tensors (no batch dim).
usage: python export_vae.py out.onnx   (then qairt-converter --float_bitwidth 16; see build_vae.sh)"""
import sys, torch, torch.nn as nn, torch.nn.functional as F
from taehv_torch import load, conv

class Dec(nn.Module):
    def __init__(self):
        super().__init__()
        self.P = {k: nn.Parameter(v, requires_grad=False) for k, v in load().items()}
        for k, v in self.P.items(): self.register_parameter(k.replace(".", "_"), v)

    def mb(self, i, x, past=None):  # MemBlock; past None -> zeros (first frame), only the x half of conv.0 is needed
        P = self.P
        w0 = P[f"decoder.{i}.conv.0.weight"]; C = x.shape[1]
        h = conv(x, w0[:, :C], P[f"decoder.{i}.conv.0.bias"]) if past is None else conv(torch.cat([x, past], 1), w0, P[f"decoder.{i}.conv.0.bias"])
        h = F.relu(h)
        h = F.relu(conv(h, P[f"decoder.{i}.conv.2.weight"], P[f"decoder.{i}.conv.2.bias"]))
        h = conv(h, P[f"decoder.{i}.conv.4.weight"], P[f"decoder.{i}.conv.4.bias"])
        return F.relu(h + x)

    def forward(self, z):
        P = self.P
        up = lambda t: F.interpolate(t, scale_factor=2, mode="nearest")
        h = 3 * torch.tanh(z / 3)
        h = F.relu(conv(h, P["decoder.1.weight"], P["decoder.1.bias"]))
        for i in (3, 4, 5): h = self.mb(i, h)
        h = conv(conv(up(h), P["decoder.7.conv.weight"], None, 0), P["decoder.8.weight"])  # TGrow stride 1 + conv
        for i in (9, 10, 11): h = self.mb(i, h)
        h = up(h)
        t = P["decoder.13.conv.weight"]  # 1x1 128 -> 256: first 128 outputs = frame 0, last 128 = frame 1
        x0 = conv(conv(h, t[:128], None, 0), P["decoder.14.weight"]); x1 = conv(conv(h, t[128:], None, 0), P["decoder.14.weight"])
        for i in (15, 16, 17):
            x1n = self.mb(i, x1, x0); x0 = self.mb(i, x0); x1 = x1n
        h = up(x1)
        h = conv(conv(h, P["decoder.19.conv.weight"][64:], None, 0), P["decoder.20.weight"])  # frame 1 only
        return conv(F.relu(h), P["decoder.22.weight"], P["decoder.22.bias"])

if __name__ == "__main__":
    import numpy as np
    from taehv_torch import decode
    torch.set_grad_enabled(False)
    m = Dec().eval()
    import os   # check latent: z.bin from sd.cpp if present, else a seeded normal one; both are saved next to the ONNX (OUT.z.bin, OUT.ref.bin) to check a compiled context
    z = torch.from_numpy(np.fromfile("z.bin", np.float32)).reshape(1, 16, 96, 64) if os.path.exists("z.bin") else torch.randn(1, 16, 96, 64, generator=torch.Generator().manual_seed(1))
    ref = decode(load(), z); out = m(z)
    print("static vs generic rel err", ((out - ref).norm() / ref.norm()).item())
    z.numpy().astype(np.float32).tofile(sys.argv[1] + ".z.bin"); ref.numpy().astype(np.float32).tofile(sys.argv[1] + ".ref.bin")
    torch.onnx.export(m, z, sys.argv[1], input_names=["z"], output_names=["img"], opset_version=13)
