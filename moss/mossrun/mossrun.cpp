// mossrun: MOSS-Image on the Hexagon NPU in one process.
//   tokens -> Qwen3.5-9B text encoder (32 layers, HMX/HVX) -> text refiner -> DiT (42 blocks, 4-step rapid-moss sampler) -> VAE latent z
// The VAE (FLUX.2 decoder, QNN context) is run by vaerun on the z file written here.
// stdin:  "load MODELDIR"                         read glue + DiT packs (resident), prints "loaded <s>"
//         "gen IDS.i32 H W SEED STEPS OUT.z"      prints "step i <ms>" lines and "done <ms>"; OUT = fp32 [32][H/8][W/8] (BN-denormalised, unpatchified)
//         "quit"
// MODELDIR: glue.bin embd.q4k te/te00..te32.bin ref/{proj,ref0,ref1}.bin dit/blk00..blk41.bin
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
extern "C" {
#include "rpcmem.h"
#include "remote.h"
#include "mega.h"
}
#include "blk_meta.h"

typedef float f32;
struct dma_buf_sync_ { unsigned long long flags; };
#define DMA_BUF_IOCTL_SYNC_ _IOW('b', 0, dma_buf_sync_)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static size_t up4k(size_t x) { return (x + 4095) & ~(size_t)4095; }
static const int NLAYER_DIT = 42, NLAYER_TE = 32, DIT_PACK = 135397376, TE_BUF = 262144000, MAXT = 512;

template <class F> static void par_for(int n, F f) {
  int nt = n < 4 ? n : 4; if (nt < 2) { for (int i = 0; i < n; i++) f(i); return; }
  std::vector<std::thread> th;
  for (int t = 0; t < nt; t++) th.emplace_back([&, t] { for (int i = t; i < n; i += nt) f(i); });
  for (auto& x : th) x.join();
}

// ------------------------------------------------------------------ rpcmem buffers
// A DSP process can hold only ~1.5 GB of fastrpc_mmap'd buffers (measured), but mapping costs ~0.6 ms per 128 MB: weights live in rpcmem and are mapped on demand (LRU window).
struct Buf { uint8_t* p = nullptr; int fd = -1; size_t sz = 0; bool mapped = false, pinned = false; unsigned long long use = 0; };
static std::vector<Buf*> g_all; static size_t g_mapped = 0; static unsigned long long g_tick = 0; static const size_t MAP_BUDGET = 1400u << 20;
static bool map_buf(Buf& b) {
  b.use = ++g_tick; if (b.mapped) return true;
  while (g_mapped + b.sz > MAP_BUDGET) {
    Buf* v = nullptr; for (Buf* x : g_all) if (x->mapped && !x->pinned && (!v || x->use < v->use)) v = x;
    if (!v) break; fastrpc_munmap(CDSP_DOMAIN_ID, v->fd, v->p, 0); v->mapped = false; g_mapped -= v->sz;
  }
  if (fastrpc_mmap(CDSP_DOMAIN_ID, b.fd, b.p, 0, b.sz, FASTRPC_MAP_FD)) { fprintf(stderr, "fastrpc_mmap %zu failed (mapped %zu MB)\n", b.sz, g_mapped >> 20); return false; }
  b.mapped = true; g_mapped += b.sz; return true;
}
static void unmap_buf(Buf& b) { if (b.mapped) { fastrpc_munmap(CDSP_DOMAIN_ID, b.fd, b.p, 0); b.mapped = false; g_mapped -= b.sz; } }
static Buf make_buf(size_t sz, bool pin = false) {   // pinned buffers are mapped now and never evicted
  Buf b; b.sz = sz; b.p = (uint8_t*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, sz);
  if (!b.p) { fprintf(stderr, "rpcmem_alloc %zu failed\n", sz); return b; }
  b.fd = rpcmem_to_fd(b.p); b.pinned = pin;
  if (pin && !map_buf(b)) { b.p = nullptr; }
  return b;
}
static void sync_clean(const Buf& b) { dma_buf_sync_ s = {(1ull << 2) | 3}; ioctl(b.fd, DMA_BUF_IOCTL_SYNC_, &s); }   // END|RW: CPU writes become visible to the DSP
static void sync_inval(const Buf& b) { dma_buf_sync_ s = {(0ull << 2) | 3}; ioctl(b.fd, DMA_BUF_IOCTL_SYNC_, &s); }  // START|RW before the CPU reads DSP results
static size_t read_into(const std::string& path, uint8_t* dst, size_t maxb, bool keep = false) {   // 4 parallel preads: one stream reads ~0.9 GB/s from the phone's UFS, four reach ~1.5 GB/s
  int fd = open(path.c_str(), O_RDONLY); if (fd < 0) return 0;
  struct stat st; fstat(fd, &st); size_t n = (size_t)st.st_size < maxb ? (size_t)st.st_size : maxb;
  const size_t chunk = 8u << 20; std::atomic<size_t> next(0), got(0);
  auto work = [&] { for (;;) { size_t o = next.fetch_add(chunk); if (o >= n) break; size_t len = std::min(chunk, n - o), d = 0; while (d < len) { ssize_t r = pread(fd, dst + o + d, len - d, o + d); if (r <= 0) return; d += r; } got += d; } };
  std::vector<std::thread> th; for (int i = 0; i < 3; i++) th.emplace_back(work); work(); for (auto& t : th) t.join();
  if (!keep) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);   // keep = leave the pages in the kernel's page cache (it drops them by itself under memory pressure)
  close(fd); return got.load();
}
// stream a list of pack files through the two g_te buffers: a loader thread reads file i+1 while use(i) runs on the DSP
#include <functional>
static int g_keep_blocks = 0;
static int g_nw = 3;   // DSP worker threads per call (MOSS_NW, 1..6): 3 suits V69; on V81 6 is ~13% faster with identical output, 8 hangs the DSP call
static std::string g_ovl;   // MOSS_LORA=dir: user-LoRA patches (blkNN.ovl) laid over the DiT packs' rank-128 LoRA slots after each read
static void apply_ovl(const std::string& pack, uint8_t* dst, size_t maxb) {
  if (g_ovl.empty() || pack.find("/dit/") == std::string::npos) return;
  std::string nm = pack.substr(pack.rfind('/') + 1); nm = g_ovl + "/" + nm.substr(0, nm.size() - 4) + ".ovl";
  FILE* f = fopen(nm.c_str(), "rb"); if (!f) { fprintf(stderr, "no overlay %s\n", nm.c_str()); return; }
  uint32_t n = 0; if (fread(&n, 4, 1, f) != 1) { fclose(f); return; }
  for (uint32_t i = 0; i < n; i++) { uint64_t o; uint32_t l; if (fread(&o, 8, 1, f) != 1 || fread(&l, 4, 1, f) != 1 || o + l > maxb || fread(dst + o, 1, l, f) != l) break; }
  fclose(f);
}   // the first g_keep_blocks DiT packs stay in the page cache between steps (their reads become memcpy)
static bool stream_files(const std::vector<std::string>& files, Buf* bufs, const std::function<bool(int, Buf&)>& use, int keep_first = 0) {
  std::mutex mu; std::condition_variable cv; int ready = -1, freed = -2; std::atomic<bool> fail(false); const int n = (int)files.size();
  std::thread loader([&] {
    for (int i = 0; i < n; i++) {
      { std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return freed >= i - 2; }); }
      Buf& b = bufs[i & 1]; size_t got = read_into(files[i], b.p, b.sz, i < keep_first); if (!got) { fprintf(stderr, "cannot read %s\n", files[i].c_str()); fail = true; }
      else { apply_ovl(files[i], b.p, b.sz); sync_clean(b); }
      { std::lock_guard<std::mutex> lk(mu); ready = i; } cv.notify_all();
      if (fail) break;
    }
  });
  bool ok = true;
  for (int i = 0; i < n && ok; i++) {
    { std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return ready >= i || fail.load(); }); }
    if (fail) { ok = false; break; }
    if (!use(i, bufs[i & 1])) ok = false;
    { std::lock_guard<std::mutex> lk(mu); freed = i; } cv.notify_all();
  }
  { std::lock_guard<std::mutex> lk(mu); freed = 1 << 30; } cv.notify_all(); loader.join();
  return ok;
}

// ------------------------------------------------------------------ activation blob layout (same order as blk_main.c)
struct Layout { int meta[M_NMETA]; size_t total; };
static Layout make_layout(int ntok, int hid, int midcols, bool rows) {
  Layout L; memset(&L, 0, sizeof L); const size_t ns = ntok / 32; size_t off = 0;
  struct E { int id; size_t b; } ent[] = {
    {M_X, (size_t)ntok * hid * 4}, {M_COS, ns * 8192}, {M_SIN, ns * 8192}, {M_H, (size_t)ntok * (hid > 4096 ? hid : 4096) * 2},
    {M_Q, (size_t)ntok * hid * 2}, {M_K, (size_t)ntok * hid * 2}, {M_V, (size_t)ntok * hid * 2}, {M_O, (size_t)ntok * hid * 2}, {M_T, (size_t)ntok * hid * 2},
    {M_MID, (size_t)ntok * midcols * 2}, {M_ROWS, rows ? (size_t)ntok * 25000 * 4 : 4096}, {M_CS, (size_t)ntok * 64 * 4},
    {M_LQ, ns * 12 * 2048}, {M_LO, ns * 4 * 2048}, {M_LU, ns * 4 * 2048}, {M_LD, ns * 4 * 2048}};
  for (auto& e : ent) { L.meta[e.id] = (int)off; off += up4k(e.b); }
  L.total = off; return L;
}

// ------------------------------------------------------------------ glue.bin
struct Glue {
  uint8_t* base = nullptr; size_t len = 0; std::map<std::string, std::pair<const void*, int>> t;   // name -> (ptr, dtype)
  bool load(const std::string& p) {
    int fd = open(p.c_str(), O_RDONLY); if (fd < 0) return false; struct stat st; fstat(fd, &st); len = st.st_size;
    base = (uint8_t*)mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0); close(fd); if (base == MAP_FAILED) return false;
    if (memcmp(base, "MGL1", 4)) return false; int n; memcpy(&n, base + 4, 4); const uint8_t* e = base + 8;
    for (int i = 0; i < n; i++, e += 76) { char name[41] = {0}; memcpy(name, e, 40); int dt; memcpy(&dt, e + 40, 4); long long off; memcpy(&off, e + 60, 8); t[name] = {base + off, dt}; }
    return true;
  }
  const f32* f(const char* n) const { return (const f32*)t.at(n).first; }
  const __fp16* h(const char* n) const { return (const __fp16*)t.at(n).first; }
};

// ------------------------------------------------------------------ Q4_K embedding rows
static void q4k_row(const uint8_t* row, f32* out) {   // 4096 values = 16 super-blocks of 144 bytes
  for (int b = 0; b < 16; b++) {
    const uint8_t* p = row + b * 144; __fp16 dh, mh; memcpy(&dh, p, 2); memcpy(&mh, p + 2, 2); const float d = (float)dh, dm = (float)mh; const uint8_t* s = p + 4; const uint8_t* qs = p + 16;
    float sc[8], mn[8];
    for (int j = 0; j < 4; j++) { sc[j] = s[j] & 63; mn[j] = s[j + 4] & 63; }
    for (int j = 4; j < 8; j++) { sc[j] = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4); mn[j] = (s[j + 4] >> 4) | ((s[j] >> 6) << 4); }
    for (int c = 0; c < 4; c++) for (int i = 0; i < 32; i++) {
      const uint8_t q = qs[c * 32 + i];
      out[b * 256 + c * 64 + i] = d * sc[2 * c] * (q & 15) - dm * mn[2 * c];
      out[b * 256 + c * 64 + 32 + i] = d * sc[2 * c + 1] * (q >> 4) - dm * mn[2 * c + 1];
    }
  }
}

// ------------------------------------------------------------------ tile helpers
// rows [nrows][cols] fp32 -> x32 tiles [slab][cb][32][32]
static void rows_to_x32(const f32* rows, int nrows, int cols, f32* x32, int ns) {
  const int ncb = cols / 32;
  par_for(ns, [&](int s) {
    for (int cb = 0; cb < ncb; cb++) { f32* tile = x32 + ((size_t)s * ncb + cb) * 1024;
      for (int r = 0; r < 32; r++) { const int row = s * 32 + r; if (row < nrows) memcpy(tile + r * 32, rows + (size_t)row * cols + cb * 32, 128); else memset(tile + r * 32, 0, 128); } }
  });
}
static void x32_to_rows(const f32* x32, int nrows, int cols, f32* rows) {
  const int ncb = cols / 32;
  par_for((nrows + 31) / 32, [&](int s) {
    for (int cb = 0; cb < ncb; cb++) { const f32* tile = x32 + ((size_t)s * ncb + cb) * 1024;
      for (int r = 0; r < 32; r++) { const int row = s * 32 + r; if (row < nrows) memcpy(rows + (size_t)row * cols + cb * 32, tile + r * 32, 128); } }
  });
}

// ------------------------------------------------------------------ state
static remote_handle64 g_h = 0;
static Glue g_glue; static uint8_t* g_embd = nullptr; static size_t g_embd_len = 0; static std::string g_dir;
static std::vector<Buf*> g_dit;         // 42 block packs, resident in rpcmem, mapped on demand
static Buf g_te[2], *g_tep[2];                     // streaming pack buffers (TE layers, refiner packs)
static Buf g_blobTE; static Layout g_LT;  // TE + refiner blob (capacity MAXT tokens, hid 4096)
static Buf g_blobD; static Layout g_LD; static int g_blobD_ntok = 0;

static int call(const Layout& L, int* meta, Buf& w, size_t woff, const Buf& blob, const f32* mod, int nmod) {
  int times[32]; int rc = 0; meta[M_WOFF] = (int)woff;
  if (!map_buf(w)) return -1;
  int r = mega_block(g_h, meta, M_NMETA, w.fd, blob.fd, (const uint8_t*)mod, nmod * 4, times, 32, &rc);
  if (r || rc) { fprintf(stderr, "mega_block r=0x%x rc=%d mode=%d\n", r, rc, meta[M_MODE]); return -1; }
  return 0;
}

static void rope_tables(const f32* bands, int H, int W, int T, int ntok, f32* cs, f32* sn, double scale = 1.0) {   // DiT 4D rope, tiles [slab][2][32][32]
  const int Nimg = H * W; const int N = Nimg + T;
  memset(cs, 0, (size_t)ntok / 32 * 8192); memset(sn, 0, (size_t)ntok / 32 * 8192);
  par_for(N, [&](int tk) {
    double c[4] = {0, 0, 0, 0};
    if (tk < Nimg) { c[1] = T; c[2] = (tk / W - (H - 1) / 2.0) * scale; c[3] = (tk % W - (W - 1) / 2.0) * scale; } else c[1] = tk - Nimg;
    for (int d = 0; d < 64; d++) {
      const double a = c[d / 16] * bands[d % 16]; const size_t idx = (((size_t)(tk / 32) * 2 + d / 32) * 1024) + (tk % 32) * 32 + d % 32;
      cs[idx] = (f32)cos(a); sn[idx] = (f32)sin(a);
    }
  });
}

// ------------------------------------------------------------------ text encoder + refiner on the NPU
static const std::vector<float> consts_tile() { std::vector<float> v; return v; }
static void fill_consts(uint8_t* blob, const Layout& L, int ns) {   // L buffers: fp16 1.0 in column 0 of every 4th tile (bias row selector)
  auto fill = [&](int id, size_t ntiles) { __fp16* t = (__fp16*)(blob + L.meta[id]); memset(t, 0, ntiles * 2048);
    for (size_t i = 0; i < ntiles; i += 4) for (int r = 0; r < 32; r++) t[i * 1024 + (r / 2) * 64 + (r & 1)] = (__fp16)1.0f; };
  fill(M_LQ, (size_t)ns * 12); fill(M_LO, (size_t)ns * 4); fill(M_LU, (size_t)ns * 4); fill(M_LD, (size_t)ns * 4);
}
// returns tx rows [T][3072]
static bool encode_text(const std::vector<int>& ids, std::vector<f32>& tx) {
  const int T = (int)ids.size(); const int ntokT = (T + 31) / 32 * 32, ns = ntokT / 32; uint8_t* blob = g_blobTE.p; const Layout& L = g_LT;
  double t0 = now();
  // embeddings -> x32 (4096 wide)
  std::vector<f32> emb((size_t)T * 4096);
  par_for(T, [&](int i) { q4k_row(g_embd + (size_t)ids[i] * 2304, &emb[(size_t)i * 4096]); });
  rows_to_x32(emb.data(), T, 4096, (f32*)(blob + L.meta[M_X]), ns);
  { f32* cs = (f32*)(blob + L.meta[M_CS]); memset(cs, 0, (size_t)ntokT * 64 * 4);
    for (int t = 0; t < T; t++) for (int i = 0; i < 32; i++) { const double a = t * pow(1e7, -(double)i / 32); cs[t * 64 + i] = (f32)cos(a); cs[t * 64 + 32 + i] = (f32)sin(a); } }
  std::vector<f32> mod(18432, 0.f); for (int i = 0; i < 4096; i++) mod[8192 + i] = 1.f;   // gate = 1 (rows 2 of 4096-wide vectors)
  memset(blob + L.meta[M_H], 0, (size_t)ntokT * 4096 * 2);
  sync_clean(g_blobTE);
  int meta[M_NMETA]; memset(meta, 0, sizeof meta); for (int i = 0; i < M_NMETA; i++) meta[i] = L.meta[i];
  meta[M_NTOK] = ntokT; meta[M_NVALID] = T; meta[M_NW] = g_nw; meta[M_NCB] = 128;
  std::vector<std::string> tef; for (int i = 0; i <= NLAYER_TE; i++) { char p[512]; snprintf(p, sizeof p, "%s/te/te%02d.bin", g_dir.c_str(), i); tef.push_back(p); }
  bool ok = stream_files(tef, g_te, [&](int i, Buf& b) {
    meta[M_MODE] = i == NLAYER_TE ? 30 : (i % 4 == 3 ? 21 : 20); meta[M_FIRST] = i == 0; meta[M_REL] = i == NLAYER_TE;
    return call(L, meta, b, 0, g_blobTE, mod.data(), 18432) == 0; });
  if (!ok) return false;
  fprintf(stderr, "te: %.2fs\n", now() - t0); t0 = now();
  // ---- refiner: h16 (TE output tiles, already in M_H) -> proj -> 2 blocks; NCB 96 ----
  { f32* x = (f32*)(blob + L.meta[M_X]); memset(x, 0, (size_t)ntokT * 3072 * 4); }
  { f32* cs = (f32*)(blob + L.meta[M_COS]); f32* sn = (f32*)(blob + L.meta[M_SIN]); std::vector<f32> bands(16); for (int i = 0; i < 16; i++) bands[i] = g_glue.f("rope.bands")[i];
    rope_tables(bands.data(), 0, 1, T, ntokT, cs, sn); }   // text rows only: coords (0, idx, 0, 0)
  fill_consts(blob, L, ns);
  std::vector<f32> rmod(18432, 0.f); for (int i = 0; i < 3072; i++) { rmod[2 * 3072 + i] = 1.f; rmod[5 * 3072 + i] = 1.f; }
  sync_clean(g_blobTE);
  meta[M_NCB] = 96; meta[M_NW] = g_nw;
  { std::vector<std::string> rf; for (const char* n : {"proj", "ref0", "ref1"}) rf.push_back(g_dir + "/ref/" + n + ".bin");
    ok = stream_files(rf, g_te, [&](int i, Buf& b) { meta[M_MODE] = i == 0 ? 2 : 1; meta[M_FIRST] = i == 0; meta[M_REL] = i == 2; return call(L, meta, b, 0, g_blobTE, rmod.data(), 18432) == 0; }); }
  if (!ok) return false;
  sync_inval(g_blobTE);
  tx.assign((size_t)T * 3072, 0.f); x32_to_rows((const f32*)(blob + L.meta[M_X]), T, 3072, tx.data());
  fprintf(stderr, "refiner: %.2fs\n", now() - t0);
  return true;
}

// ------------------------------------------------------------------ CPU glue
static void matvec_f32(const f32* W, const f32* b, const f32* x, int out, int in, f32* y) {
  par_for(out, [&](int o) { double a = b ? b[o] : 0; const f32* w = W + (size_t)o * in; float s = 0; for (int i = 0; i < in; i++) s += w[i] * x[i]; y[o] = a + s; });
}
static void matvec_f16(const __fp16* W, const f32* b, const f32* x, int out, int in, f32* y) {
  par_for(out, [&](int o) { const __fp16* w = W + (size_t)o * in; float s = 0; for (int i = 0; i < in; i++) s += (float)w[i] * x[i]; y[o] = (b ? b[o] : 0.f) + s; });
}
static void time_vecs(float code, std::vector<f32>& mod, std::vector<f32>& head) {
  f32 e[256]; for (int i = 0; i < 128; i++) { const double a = (double)code * pow(10000.0, -(double)i / 128); e[i] = (f32)cos(a); e[128 + i] = (f32)sin(a); }
  std::vector<f32> a(3072), b(3072);
  matvec_f32(g_glue.f("time_embedding.0.weight"), g_glue.f("time_embedding.0.bias"), e, 3072, 256, a.data());
  for (auto& v : a) v = v / (1.f + expf(-v));
  matvec_f32(g_glue.f("time_embedding.2.weight"), g_glue.f("time_embedding.2.bias"), a.data(), 3072, 3072, b.data());
  mod.resize(18432); head.resize(6144);
  matvec_f16(g_glue.h("shared_block_modulation.weight"), g_glue.f("shared_block_modulation.bias"), b.data(), 18432, 3072, mod.data());
  matvec_f16(g_glue.h("video_head_modulation.weight"), g_glue.f("video_head_modulation.bias"), b.data(), 6144, 3072, head.data());
}

// ------------------------------------------------------------------ DiT sampler
static uint64_t ids_hash(const std::vector<int>& ids) { uint64_t h = 1469598103934665603ull ^ 0x4d4f5353ull; for (int v : ids) { h ^= (uint32_t)v; h *= 1099511628211ull; } return h; }
static bool load_tx(const std::vector<int>& ids, std::vector<f32>& tx) {   // the text side depends on the prompt only: reuse it for the next seed / picture
  FILE* f = fopen((g_dir + "/tx.cache").c_str(), "rb"); if (!f) return false;
  uint64_t h = 0; int T = 0; bool ok = fread(&h, 8, 1, f) == 1 && fread(&T, 4, 1, f) == 1 && h == ids_hash(ids) && T == (int)ids.size();
  if (ok) { tx.resize((size_t)T * 3072); ok = fread(tx.data(), 4, tx.size(), f) == tx.size(); }
  fclose(f); return ok;
}
static void save_tx(const std::vector<int>& ids, const std::vector<f32>& tx) {
  FILE* f = fopen((g_dir + "/tx.cache").c_str(), "wb"); if (!f) return; uint64_t h = ids_hash(ids); int T = (int)ids.size();
  fwrite(&h, 8, 1, f); fwrite(&T, 4, 1, f); fwrite(tx.data(), 4, tx.size(), f); fclose(f);
}
static void upsample2(const f32* src, int h, int w, f32* dst) {   // [128][h][w] -> [128][2h][2w], bilinear with half-pixel centres
  par_for(128, [&](int c) { for (int y = 0; y < 2 * h; y++) { float fy = (y + 0.5f) / 2 - 0.5f; int y0 = (int)floorf(fy); float ty = fy - y0; int ya = std::max(0, std::min(h - 1, y0)), yb = std::max(0, std::min(h - 1, y0 + 1));
    for (int x = 0; x < 2 * w; x++) { float fx = (x + 0.5f) / 2 - 0.5f; int x0_ = (int)floorf(fx); float tx_ = fx - x0_; int xa = std::max(0, std::min(w - 1, x0_)), xb = std::max(0, std::min(w - 1, x0_ + 1));
      const f32* p = src + (size_t)c * h * w; dst[((size_t)c * 2 * h + y) * 2 * w + x] = (1 - ty) * ((1 - tx_) * p[ya * w + xa] + tx_ * p[ya * w + xb]) + ty * ((1 - tx_) * p[yb * w + xa] + tx_ * p[yb * w + xb]); } } });
}

static bool generate(const std::vector<int>& ids, int Hpx, int Wpx, unsigned seed, int steps, const std::string& outz) {
  double tall = now(); std::vector<f32> tx;
  if (load_tx(ids, tx)) fprintf(stderr, "te: cached\n"); else { if (!encode_text(ids, tx)) return false; save_tx(ids, tx); }
  const int T = (int)ids.size(), H = Hpx / 16, W = Wpx / 16, Nimg = H * W, N = Nimg + T, ntok = (N + 31) / 32 * 32;
  const int LR = getenv("MOSS_LOWRES") ? atoi(getenv("MOSS_LOWRES")) : 0; const double LS = getenv("MOSS_LOWSCALE") ? atof(getenv("MOSS_LOWSCALE")) : 2.0;
  const bool low = LR > 0 && LR < steps && H % 2 == 0 && W % 2 == 0; const int Hl = H / 2, Wl = W / 2, Nl = Hl * Wl;   // first LR steps on a half-size latent (a quarter of the tokens)
  if (g_blobD_ntok < ntok) {
    if (g_blobD.p) { unmap_buf(g_blobD); rpcmem_free(g_blobD.p); }
    g_LD = make_layout(ntok, 3072, 8192, false); g_blobD = make_buf(g_LD.total, true); if (!g_blobD.p) return false; memset(g_blobD.p, 0, g_LD.total); g_blobD_ntok = ntok;
  }
  uint8_t* blob = g_blobD.p; const Layout& L = g_LD;
  std::vector<f32> bands(16); for (int i = 0; i < 16; i++) bands[i] = g_glue.f("rope.bands")[i];
  int ropeFor = -1;   // 0 = low grid, 1 = full grid
  std::vector<f32> z((size_t)128 * Nimg), x0((size_t)128 * Nimg, 0.f), xin((size_t)128 * Nimg), v((size_t)128 * Nimg), zl, x0l;
  { std::mt19937 gen(seed); std::normal_distribution<float> nd(0.f, 1.f); for (auto& a : z) a = nd(gen); }
  if (low) { zl.resize((size_t)128 * Nl); x0l.assign((size_t)128 * Nl, 0.f);
    for (int c = 0; c < 128; c++) for (int i = 0; i < Hl; i++) for (int j = 0; j < Wl; j++) { const f32* p = &z[(size_t)c * Nimg]; zl[((size_t)c * Hl + i) * Wl + j] = 0.5f * (p[(2 * i) * W + 2 * j] + p[(2 * i) * W + 2 * j + 1] + p[(2 * i + 1) * W + 2 * j] + p[(2 * i + 1) * W + 2 * j + 1]); } }   // mean of 4 N(0,1) times 2: unit variance
  const f32* Q = g_glue.f("latent_mix.Q"); const f32* codes = g_glue.f("step_codes");
  const f32* Wemb = g_glue.f("video_embedder.weight"); const f32* bemb = g_glue.f("video_embedder.bias");
  const f32* Wfin = g_glue.f("final_linear_video.weight"); const f32* nfin = g_glue.f("final_norm_video.weight");
  std::vector<f32> rows((size_t)N * 3072), mod, head, outrows((size_t)Nimg * 3072);
  int meta[M_NMETA]; for (int i = 0; i < M_NMETA; i++) meta[i] = L.meta[i];
  meta[M_NW] = g_nw; meta[M_NCB] = 96; meta[M_MODE] = 0;
  for (int s = 0; s < steps; s++) {
    double ts = now(), tA = ts, tB, tC, tD, tE;
    const bool lowstep = low && s < LR;
    const int Hs = lowstep ? Hl : H, Ws = lowstep ? Wl : W, Ni = Hs * Ws, Ns = Ni + T, ntokS = (Ns + 31) / 32 * 32, nsS = ntokS / 32;
    f32* zs = lowstep ? zl.data() : z.data(); f32* x0s = lowstep ? x0l.data() : x0.data();
    if (low && s == LR) upsample2(x0l.data(), Hl, Wl, x0.data());   // the clean estimate of the low-res steps seeds the full-res ones
    const int rf = lowstep ? 0 : 1; if (ropeFor != rf) { rope_tables(bands.data(), Hs, Ws, T, ntokS, (f32*)(blob + L.meta[M_COS]), (f32*)(blob + L.meta[M_SIN]), lowstep ? LS : 1.0); ropeFor = rf; }
    meta[M_NTOK] = ntokS; meta[M_NVALID] = Ns;
    memcpy(&rows[(size_t)Ni * 3072], tx.data(), (size_t)T * 3072 * 4);
    par_for(Ni / 16 + 1, [&](int blk) { for (int t = blk * 16; t < std::min(Ni, blk * 16 + 16); t++) for (int o = 0; o < 128; o++) {
      float a = zs[(size_t)o * Ni + t]; for (int c = 0; c < 128; c++) a += Q[o * 128 + c] * x0s[(size_t)c * Ni + t]; xin[(size_t)o * Ni + t] = a; } });
    par_for(Ni / 16 + 1, [&](int blk) { for (int t = blk * 16; t < std::min(Ni, blk * 16 + 16); t++) {
      f32 col[128]; for (int c = 0; c < 128; c++) col[c] = xin[(size_t)c * Ni + t];
      f32* r = &rows[(size_t)t * 3072]; for (int o = 0; o < 3072; o++) { const f32* w = Wemb + (size_t)o * 128; float a = bemb[o]; for (int c = 0; c < 128; c++) a += w[c] * col[c]; r[o] = a; } } });
    tB = now();
    std::vector<f32> modv; time_vecs(codes[s], modv, head);
    tC = now();
    rows_to_x32(rows.data(), Ns, 3072, (f32*)(blob + L.meta[M_X]), nsS);
    sync_clean(g_blobD); tD = now();
    std::vector<std::string> df; for (int k = 0; k < NLAYER_DIT; k++) { char p[512]; snprintf(p, sizeof p, "%s/dit/blk%02d.bin", g_dir.c_str(), k); df.push_back(p); }
    if (!stream_files(df, g_te, [&](int k, Buf& b) { meta[M_FIRST] = k == 0; meta[M_REL] = k == NLAYER_DIT - 1; return call(L, meta, b, 0, g_blobD, modv.data(), 18432) == 0; }, g_keep_blocks)) return false;
    sync_inval(g_blobD); tE = now();
    x32_to_rows((const f32*)(blob + L.meta[M_X]), Ni, 3072, outrows.data());
    par_for(Ni / 16 + 1, [&](int blk) { for (int t = blk * 16; t < std::min(Ni, blk * 16 + 16); t++) {
      f32* r = &outrows[(size_t)t * 3072]; double ss = 0; for (int i = 0; i < 3072; i++) ss += (double)r[i] * r[i]; const float rs = 1.f / sqrtf((float)(ss / 3072) + 1e-6f);
      for (int i = 0; i < 3072; i++) r[i] = r[i] * rs * nfin[i] * (1.f + head[3072 + i]) + head[i];
      for (int o = 0; o < 128; o++) { const f32* w = Wfin + (size_t)o * 3072; float a = 0; for (int i = 0; i < 3072; i++) a += w[i] * r[i]; v[(size_t)o * Ni + t] = a; } } });
    for (size_t i = 0; i < (size_t)128 * Ni; i++) x0s[i] = zs[i] - v[i];
    fprintf(stderr, "  step %d%s: mix+embed %.0f ms, time/mod %.0f ms, to-x32 %.0f ms, DSP blocks %.0f ms, head+sampler %.0f ms\n", s, lowstep ? " (low-res)" : "", (tB - tA) * 1e3, (tC - tB) * 1e3, (tD - tC) * 1e3, (tE - tD) * 1e3, (now() - tE) * 1e3);
    printf("step %d %.0f\n", s, (now() - ts) * 1000); fflush(stdout);
  }
  // BN denorm + unpatchify -> [32][2H][2W]
  const f32* bm = g_glue.f("bn.mean"); const f32* bv = g_glue.f("bn.var"); const int H2 = 2 * H, W2 = 2 * W;
  std::vector<f32> zo((size_t)32 * H2 * W2);
  for (int c = 0; c < 128; c++) { const float sc = sqrtf(bv[c] + 1e-4f), mu = bm[c]; const int cc = c / 4, pi = (c / 2) % 2, pj = c % 2;
    for (int h = 0; h < H; h++) for (int w = 0; w < W; w++) zo[((size_t)cc * H2 + 2 * h + pi) * W2 + 2 * w + pj] = x0[(size_t)c * Nimg + h * W + w] * sc + mu; }
  FILE* f = fopen(outz.c_str(), "wb"); fwrite(zo.data(), 4, zo.size(), f); fclose(f);
  printf("done %.0f\n", (now() - tall) * 1000); fflush(stdout);
  return true;
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  rpcmem_init();
  struct remote_rpc_control_unsigned_module ud = {CDSP_DOMAIN_ID, 1}; remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &ud, sizeof ud);
  if (mega_open(mega_URI CDSP_DOMAIN, &g_h)) { fprintf(stderr, "mega_open failed\n"); return 1; }
  printf("ready\n");
  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream ss(line); std::string cmd; ss >> cmd;
    if (cmd == "quit") break;
    if (cmd == "load") {
      ss >> g_dir; double t0 = now();
      if (!g_glue.load(g_dir + "/glue.bin")) { printf("error glue\n"); continue; }
      { int fd = open((g_dir + "/embd.q4k").c_str(), O_RDONLY); if (fd < 0) { printf("error embd\n"); continue; } struct stat st; fstat(fd, &st); g_embd_len = st.st_size; g_embd = (uint8_t*)mmap(nullptr, g_embd_len, PROT_READ, MAP_PRIVATE, fd, 0); close(fd); }
      g_te[0] = make_buf(TE_BUF); g_te[1] = make_buf(TE_BUF); g_all.push_back(&g_te[0]); g_all.push_back(&g_te[1]); if (!g_te[0].p || !g_te[1].p) { printf("error tebuf\n"); continue; }
      g_LT = make_layout(MAXT, 4096, 12288, true); g_blobTE = make_buf(g_LT.total, true); if (!g_blobTE.p) { printf("error teblob\n"); continue; } memset(g_blobTE.p, 0, g_LT.total);
      { long long avail = 0; FILE* mi = fopen("/proc/meminfo", "r"); char ln[256]; while (mi && fgets(ln, sizeof ln, mi)) if (!strncmp(ln, "MemAvailable:", 13)) avail = atoll(ln + 13) * 1024; if (mi) fclose(mi);
        if (getenv("MOSS_LORA")) g_ovl = getenv("MOSS_LORA");
        if (getenv("MOSS_NW")) { int n = atoi(getenv("MOSS_NW")); if (n >= 1 && n <= 6) g_nw = n; }
        g_keep_blocks = getenv("MOSS_KEEP") ? atoi(getenv("MOSS_KEEP")) : 0;   // measured: keeping blocks in the page cache made steps slower on a 12 GB phone (reclaim stalls), so it is opt-in
        fprintf(stderr, "page cache keeps the first %d DiT blocks (MemAvailable %.1f GB)\n", g_keep_blocks, avail / 1e9); }
      printf("loaded %.1f\n", now() - t0); continue;
    }
    if (cmd == "gen") {
      std::string idsf, outz; int H, W, steps; unsigned seed; ss >> idsf >> H >> W >> seed >> steps >> outz;
      std::vector<int> ids; { FILE* f = fopen(idsf.c_str(), "rb"); if (!f) { printf("error ids\n"); continue; } fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); ids.resize(n / 4); if (fread(ids.data(), 4, ids.size(), f) != ids.size()) {} fclose(f); }
      if (ids.empty() || (int)ids.size() > MAXT || H % 16 || W % 16 || (H / 16) * (W / 16) > 6144 || steps < 1 || steps > 4) { printf("error args\n"); continue; }
      if (!generate(ids, H, W, seed, steps, outz)) printf("error gen\n");
    }
  }
  mega_close(g_h);
  return 0;
}
