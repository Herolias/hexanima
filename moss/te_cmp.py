"""Text side check: the phone's refined text rows (tx.cache / txn.cache: NPU Qwen3.5-9B + text refiner) vs qwen35_ref.py (torch fp32) on a GGUF, through the same refiner.
usage: python te_cmp.py DIR TE.gguf TAG [DIT.gguf]   DIR holds ids.i32 neg.i32 tx.cache txn.cache from files/moss; writes DIR/TAG.tx.cache + DIR/TAG.txn.cache (same format,
       mossrun reuses them when copied over files/moss/tx.cache / txn.cache) and DIR/TAG.{tx,txn}.f32 (raw text encoder output)."""
import os, sys, struct, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen35_ref, moss_ref
D, te, tag = sys.argv[1], sys.argv[2], sys.argv[3]
g = moss_ref.W(sys.argv[4] if len(sys.argv) > 4 else "models/MOSS-Image-SFT-Q8_0.gguf"); q = qwen35_ref.Q(te)
def ids_hash(ids, q8=os.environ.get("TE_Q8") == "1"):   # mossrun keys the 8-bit encoder's caches apart (TE_Q8=1 for those)
    h = (1469598103934665603 ^ 0x4d4f5353 ^ ((0x5138 << 32) if q8 else 0)) & (2**64 - 1)
    for v in ids: h ^= v & 0xffffffff; h = (h * 1099511628211) & (2**64 - 1)
    return h
def load_cache(p):
    b = open(p, "rb").read(); h, T = struct.unpack_from("<Qi", b, 0); return h, np.frombuffer(b, np.float32, T * 3072, 12).reshape(T, 3072)
torch.set_grad_enabled(False)
for ids_f, cache_f in (("ids.i32", "tx"), ("neg.i32", "txn")):
    ids = np.fromfile(os.path.join(D, ids_f), np.int32).tolist(); T = len(ids)
    h, phone = load_cache(os.path.join(D, cache_f + ".cache"))
    assert h == ids_hash(ids) and phone.shape[0] == T, f"{cache_f}.cache is for another prompt"
    raw = qwen35_ref.encode(q, ids)
    sin, cos = moss_ref.rope_tables(0, 0, T, g("rope.bands"))
    ref = moss_ref.text_refine(raw, sin, cos, g).numpy()
    rel = np.linalg.norm(phone - ref) / np.linalg.norm(ref); per_tok = np.linalg.norm(phone - ref, axis=1) / np.linalg.norm(ref, axis=1)
    print(f"{cache_f}: {T} tokens; phone vs {tag}: rel err {rel:.2%}, per token median {np.median(per_tok):.2%} max {per_tok.max():.2%} (token {per_tok.argmax()})", flush=True)
    raw.numpy().astype(np.float32).tofile(os.path.join(D, f"{tag}.{cache_f}.f32"))
    with open(os.path.join(D, f"{tag}.{cache_f}.cache"), "wb") as f: f.write(struct.pack("<Qi", ids_hash(ids), T)); f.write(ref.astype(np.float32).tobytes())
