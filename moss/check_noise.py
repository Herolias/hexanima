"""mossrun's initial noise (philox_randn in mossrun.cpp) vs stable-diffusion.cpp's default RNG, which Unsloth Studio uses for MOSS-Image GGUFs:
PhiloxRNG from sd.cpp src/core/rng_philox.hpp (AUTOMATIC1111's imitation of torch's CUDA randn; it only agrees with real CUDA randn for small tensors,
so sd.cpp's own algorithm is the reference), ported to numpy below.
usage: python check_noise.py      (needs a host C++ compiler; compiles only philox_randn out of mossrun.cpp)"""
import os, subprocess, tempfile, numpy as np
def sdcpp_philox_randn(seed, n, offset=0):   # rng_philox.hpp PhiloxRNG::randn, vectorised
    M0, M1, W0, W1 = np.uint64(0xD2511F53), np.uint64(0xCD9E8D57), 0x9E3779B9, 0xBB67AE85
    c0 = np.full(n, offset, np.uint64); c1 = np.zeros(n, np.uint64); c2 = np.arange(n, dtype=np.uint64); c3 = np.zeros(n, np.uint64)
    k0, k1, m = seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF, np.uint64(0xFFFFFFFF)
    for _ in range(10):
        v1, v2 = c0 * M0, c2 * M1
        c0, c1, c2, c3 = (v2 >> np.uint64(32)) ^ c1 ^ np.uint64(k0), v2 & m, (v1 >> np.uint64(32)) ^ c3 ^ np.uint64(k1), v1 & m
        k0, k1 = (k0 + W0) & 0xFFFFFFFF, (k1 + W1) & 0xFFFFFFFF
    inv = np.float32(2.3283064e-10); inv2pi = np.float32(inv * np.float32(6.2831855))
    u = c0.astype(np.float32) * inv + inv / np.float32(2); v = c1.astype(np.float32) * inv2pi + inv2pi / np.float32(2)
    return np.sqrt(np.float32(-2) * np.log(u)) * np.sin(v)
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mossrun", "mossrun.cpp")).read()
fn = src[src.index("static void philox_randn"):src.index("static bool generate(")]
prog = "#include <cmath>\n#include <cstdio>\n#include <cstdlib>\n#include <cstdint>\n#include <vector>\ntypedef float f32;\n" + fn + r"""
int main(int argc, char** argv) { unsigned long long seed = strtoull(argv[1], 0, 10); size_t n = strtoull(argv[2], 0, 10);
  std::vector<f32> o(n); philox_randn(seed, o.data(), n); fwrite(o.data(), 4, n, stdout); return 0; }"""
with tempfile.TemporaryDirectory() as d:
    open(f"{d}/t.cpp", "w").write(prog); subprocess.run(["c++", "-O2", "-std=c++17", f"{d}/t.cpp", "-o", f"{d}/t"], check=True)
    worst = 0.0
    for seed, W, H in ((42, 1024, 1024), (94195422, 960, 1280), (3836630325, 832, 1216), (2**40 + 7, 512, 512)):
        n = 128 * (H // 16) * (W // 16)
        ours = np.frombuffer(subprocess.run([f"{d}/t", str(seed), str(n)], check=True, capture_output=True).stdout, np.float32)
        ref = sdcpp_philox_randn(seed, n); diff = np.abs(ours - ref); worst = max(worst, float(diff.max()))
        line = f"seed {seed} {W}x{H}: vs sd.cpp Philox max abs diff {diff.max():.2e} ({np.mean(ours == ref):.1%} bit-identical), mean {ours.mean():+.4f} std {ours.std():.4f}"
        print(line)
print("OK" if worst < 1e-4 else "MISMATCH")
