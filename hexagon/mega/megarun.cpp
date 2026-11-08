#include <sys/stat.h>
// HMX block runner (replaces chainrun): holds the 28 fp16 crouton block weights in rpcmem and runs them on the DSP with the mega kernels.
// stdin: "run X_IN EMB LORA CTX X_OUT [GH GW]" -> "done <ms>"; "quit".   X_IN/X_OUT fp32 [D][T], EMB fp32[2048], LORA fp32[6144], CTX fp32 [1024][512]
// usage: megarun HMXDIR [FIRST COUNT]
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <csignal>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
extern "C" {
#include "rpcmem.h"
#include "remote.h"
#include "mega.h"
}
#include "blk_meta.h"
#include "blk_wlayout.h"

struct dma_buf_sync_ { unsigned long long flags; };
#define DMA_BUF_IOCTL_SYNC_ _IOW('b', 0, dma_buf_sync_)
static const int D = 2048, NCTX = 512, TMAX = 6144;
typedef float f32;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static std::vector<char> readfile(const std::string& p) {
  FILE* f = fopen(p.c_str(), "rb"); std::vector<char> b; if (!f) return b;
  fseek(f, 0, SEEK_END); size_t n = ftell(f); fseek(f, 0, SEEK_SET); b.resize(n);
  if (fread(b.data(), 1, n, f) != n) b.clear();
  fclose(f); return b;
}
static size_t up(size_t x) { return (x + 4095) & ~(size_t)4095; }

struct Layout { int meta[M_NMETA]; size_t total; };
static bool g_q8 = false;
static Layout layout(int T) {
  Layout L; memset(&L, 0, sizeof L);
  size_t sz[M_NMETA] = {0};
  sz[M_X] = (size_t)T * D * 4; sz[M_CTX] = (size_t)NCTX * 1024 * 2; sz[M_MOD] = 9 * D * 4;
  sz[M_COS] = sz[M_SIN] = (size_t)(T / 32) * 2 * 1024 * 4;
  sz[M_H] = sz[M_Q] = sz[M_K] = sz[M_V] = sz[M_O] = sz[M_T] = (size_t)T * D * 2;
  sz[M_MID] = (size_t)T * 8192 * 2; sz[M_CK] = sz[M_CV] = (size_t)NCTX * D * 2;
  static const int order[] = {M_X, M_CTX, M_MOD, M_COS, M_SIN, M_H, M_Q, M_K, M_V, M_O, M_T, M_MID, M_CK, M_CV};
  size_t off = 0;
  for (int id : order) { L.meta[id] = (int)off; off += up(sz[id]); }
#define WO(x) (g_q8 ? WQ(x) : (x))
  L.meta[M_WSQ] = WO(W_SQ); L.meta[M_WSK] = WO(W_SK); L.meta[M_WSV] = WO(W_SV); L.meta[M_WSO] = WO(W_SO); L.meta[M_WCQ] = WO(W_CQ); L.meta[M_WCK] = WO(W_CK); L.meta[M_WCV] = WO(W_CV);
  L.meta[M_WCO] = WO(W_CO); L.meta[M_WM1] = WO(W_M1); L.meta[M_WM2] = WO(W_M2); L.meta[M_GAIN] = WO(W_GAIN); L.meta[M_Q8] = g_q8;
  L.meta[M_NTOK] = T; L.meta[M_NCTX] = NCTX; L.meta[M_NW] = 3; L.meta[M_REL] = 1; L.meta[M_FIRST] = 1;
  L.total = off; return L;
}

// [D][T] fp32 -> x32 tiles [slab][cb] of 32x32 row-major (row = token, col = channel)
static void to_x32(const f32* src, f32* dst, int T) {
  for (int cb = 0; cb < D / 32; cb++) for (int c = 0; c < 32; c++) {
    const f32* row = src + (size_t)(cb * 32 + c) * T;
    for (int s = 0; s < T / 32; s++) { f32* tile = dst + ((size_t)s * (D / 32) + cb) * 1024; for (int r = 0; r < 32; r++) tile[r * 32 + c] = row[s * 32 + r]; }
  }
}
static void from_x32(const f32* src, f32* dst, int T) {
  for (int cb = 0; cb < D / 32; cb++) for (int c = 0; c < 32; c++) {
    f32* row = dst + (size_t)(cb * 32 + c) * T;
    for (int s = 0; s < T / 32; s++) { const f32* tile = src + ((size_t)s * (D / 32) + cb) * 1024; for (int r = 0; r < 32; r++) row[s * 32 + r] = tile[r * 32 + c]; }
  }
}
// ctx fp32 [1024 ch][512 tok] -> fp16 act tiles of matrix [512 rows][1024 cols]
static void to_ctx16(const f32* ct, __fp16* dst) {
  for (int c = 0; c < 1024; c++) for (int i = 0; i < NCTX; i++)
    dst[(((size_t)(i >> 5) * 32 + (c >> 5)) * 1024) + ((i & 31) >> 1) * 64 + (c & 31) * 2 + (i & 1)] = (__fp16)ct[(size_t)c * NCTX + i];
}
static void make_rope(int gh, int gw, f32* cs, f32* sn) {   // tiles [slab][j][32 rows][32 cols] (angle index d = 32 j + col)
  const int dim_h = 128 / 6 * 2, dim_t = 128 - 2 * dim_h, nt = dim_t / 2, nh = dim_h / 2;
  std::vector<double> fh(nh);
  for (int i = 0; i < nh; i++) fh[i] = 1.0 / pow(10000.0, (2.0 * i) / dim_h);
  int T = gh * gw;
  for (int tk = 0; tk < T; tk++) {
    int hh = tk / gw, ww = tk % gw;
    for (int d = 0; d < 64; d++) {
      double ang = d < nt ? 0.0 : d < nt + nh ? hh * fh[d - nt] : ww * fh[d - nt - nh];
      size_t idx = (((size_t)(tk / 32) * 2 + (d / 32)) * 1024) + (tk % 32) * 32 + (d % 32);
      cs[idx] = (f32)cos(ang); sn[idx] = (f32)sin(ang);
    }
  }
}

// SIGTERM (anima.sh stop): finish the DSP call in flight, then release VTCM/HMX/pool through mega_close and exit. Being SIGKILLed in the middle of a
// block left the HMX unusable for the next engine start on V79 (upstream PITFALLS / 1.2.2); a clean exit avoids that.
static volatile sig_atomic_t g_stop = 0;
static void on_term(int) { g_stop = 1; }

// HMX self-test: one fp16 GEMM 32x256x64 through the SDK path (mega_probe stage 2) against a scalar reference on the DSP.
// A silently unavailable HMX returns zeros (error ~ the size of the values, >> 0.5); VTCM/HMX busy (another megarun starting at the same moment) is retried.
static bool hmx_selftest(remote_handle64 h, int first) {
  for (int a = 0, bad = 0; a < 40; a++) {
    long long cyc = 0; float err = -1; int lockrc = -999, vtcmrc = -999;
    int r = mega_probe(h, 256, 1, 2, &cyc, &err, &lockrc, &vtcmrc);
    if (r == 0 && lockrc == 0 && err >= 0 && err < 0.5f) { if (a) fprintf(stderr, "megarun %d: HMX self-test ok after %d retries\n", first, a); return true; }
    bool busy = r == 0 && (vtcmrc < 0 || (lockrc != 0 && lockrc != -999));
    fprintf(stderr, "megarun %d: HMX self-test %s (r=0x%x lock=%d vtcm=%d err=%g)\n", first, busy ? "busy" : "FAILED", r, lockrc, vtcmrc, err);
    if (!busy && ++bad >= 4) return false;
    usleep(busy ? 150000 : 500000);
  }
  return false;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: megarun HMXDIR [FIRST COUNT]\n"); return 1; }
  { struct sigaction sa; memset(&sa, 0, sizeof sa); sa.sa_handler = on_term; sigaction(SIGTERM, &sa, nullptr); signal(SIGPIPE, SIG_IGN); }   // no SA_RESTART: a blocking stdin read returns
  std::string dir = argv[1]; int first = argc > 2 ? atoi(argv[2]) : 0, count = argc > 3 ? atoi(argv[3]) : 28;
  rpcmem_init();
  std::vector<uint8_t*> wts(count);
  { char p[512]; snprintf(p, sizeof p, "%s/blk%02d.bin", dir.c_str(), first); struct stat st; if (!stat(p, &st) && (size_t)st.st_size == WQ_TOTAL) g_q8 = true; }   // q8 weight files are recognised by size
  const size_t wtot = g_q8 ? WQ_TOTAL : W_TOTAL, wmodw = g_q8 ? WQ_MODW : W_MODW;
  double t0 = now();
  for (int i = 0; i < count; i++) {
    char p[512]; snprintf(p, sizeof p, "%s/blk%02d.bin", dir.c_str(), first + i);
    wts[i] = (uint8_t*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, wtot);
    int fd = open(p, O_RDONLY); if (fd < 0 || !wts[i]) { fprintf(stderr, "cannot load %s\n", p); return 1; }
    size_t got = 0; while (got < wtot) { ssize_t r = read(fd, wts[i] + got, wtot - got); if (r <= 0) break; got += r; }
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); close(fd);
    if (got != wtot) { fprintf(stderr, "short read %s\n", p); return 1; }
  }
  fprintf(stderr, "megarun: weights loaded %.1fs\n", now() - t0);
  struct remote_rpc_control_unsigned_module ud = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &ud, sizeof ud);
  remote_handle64 h = 0; int r = mega_open(mega_URI CDSP_DOMAIN, &h);
  if (r) { fprintf(stderr, "mega_open 0x%x\n", r); return 1; }
  if (!getenv("MEGA_NOSELFTEST") && !hmx_selftest(h, first)) { fprintf(stderr, "megarun %d: HMX unavailable\n", first); printf("error hmx\n"); fflush(stdout); mega_close(h); return 3; }
  const int nw_env = getenv("MEGA_NW") ? atoi(getenv("MEGA_NW")) : 0;   // DSP worker threads per block (skel default 3, at most 6: MOSS hung with 8 on V81); V81 has more HVX contexts than V69
  for (int i = 0; i < count; i++) {   // map once (a DSP process takes at most ~13 of these buffers: run several megarun processes for 28 blocks)
    int fd = rpcmem_to_fd(wts[i]);
    dma_buf_sync_ sy = {(1ull << 2) | 3};   // DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW: clean CPU caches so the DSP sees the weights
    ioctl(fd, DMA_BUF_IOCTL_SYNC_, &sy);
    int mr = fastrpc_mmap(CDSP_DOMAIN_ID, fd, wts[i], 0, wmodw, FASTRPC_MAP_FD);
    if (mr) { fprintf(stderr, "fastrpc_mmap block %d rc=%d\n", first + i, mr); return 1; }
  }
  uint8_t* blob = nullptr; int blobT = 0, blobfd = -1;   // activations: one rpcmem buffer mapped into the DSP; the CPU/DSP cache sync is done by hand (dma-buf sync here, dccleaninva on the DSP)

  // make the CPU-written weights visible to the DSP (clean CPU caches), then map every buffer into the DSP once (extended address space: 4 GB do not fit in the normal one)
  for (int i = 0; i < count; i++) {
    int fd = rpcmem_to_fd(wts[i]);
    dma_buf_sync_ sy = {(1ull << 2) | 3};   // DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW
    ioctl(fd, DMA_BUF_IOCTL_SYNC_, &sy);
  }
  printf("ready\n"); fflush(stdout);
  int cur_gh = -1, cur_gw = -1;
  std::vector<f32> xin((size_t)D * TMAX), emb(D), lora(6144), ctxf((size_t)1024 * NCTX), mod((size_t)count * 9 * D);
  std::string line;
  while (!g_stop && std::getline(std::cin, line)) {
    std::istringstream ss(line); std::string cmd; ss >> cmd;
    if (cmd == "quit") break;
    if (cmd != "run") continue;
    std::string xi, em, lo, cx, xo; int gh = 48, gw = 32; unsigned skipmask = 0; ss >> xi >> em >> lo >> cx >> xo >> gh >> gw >> skipmask;
    int T = gh * gw;
    auto xb = readfile(xi), eb = readfile(em), lb = readfile(lo), cb = readfile(cx);
    if (T % 32 || T > TMAX || xb.size() != (size_t)D * T * 4 || eb.size() != D * 4 || lb.size() != 6144 * 4 || cb.size() != (size_t)1024 * NCTX * 4) { fprintf(stderr, "bad input T=%d\n", T); printf("error\n"); fflush(stdout); continue; }
    double ts = now(), t_in = ts;
    if (T > blobT) {
      if (blob) { fastrpc_munmap(CDSP_DOMAIN_ID, blobfd, blob, 0); rpcmem_free(blob); }
      Layout Lb = layout(T); blobT = T;
      blob = (uint8_t*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, Lb.total);
      if (!blob) { fprintf(stderr, "blob alloc failed\n"); printf("error\n"); fflush(stdout); continue; }
      blobfd = rpcmem_to_fd(blob); memset(blob, 0, Lb.total); cur_gh = -1;
      if (fastrpc_mmap(CDSP_DOMAIN_ID, blobfd, blob, 0, Lb.total, FASTRPC_MAP_FD)) { fprintf(stderr, "blob mmap failed\n"); printf("error\n"); fflush(stdout); continue; }
    }
    Layout L = layout(T);
    if (nw_env > 0 && nw_env <= 6) L.meta[M_NW] = nw_env;
    if (gh != cur_gh || gw != cur_gw) { make_rope(gh, gw, (f32*)(blob + L.meta[M_COS]), (f32*)(blob + L.meta[M_SIN])); cur_gh = gh; cur_gw = gw; }
    to_ctx16((const f32*)cb.data(), (__fp16*)(blob + L.meta[M_CTX]));
    bool xin32 = xi.size() > 4 && xi.compare(xi.size() - 4, 4, ".x32") == 0;
    if (xin32) memcpy(blob + L.meta[M_X], xb.data(), (size_t)D * T * 4); else to_x32((const f32*)xb.data(), (f32*)(blob + L.meta[M_X]), T);
    { dma_buf_sync_ sy = {(1ull << 2) | 3}; ioctl(blobfd, DMA_BUF_IOCTL_SYNC_, &sy); }   // END|RW: flush CPU writes
    // modulation vectors of all blocks on a helper thread; block k is waited for before its DSP call
    const f32* e = (const f32*)eb.data(); const f32* lr = (const f32*)lb.data();
    std::atomic<int> mdone(0);
    std::thread mt([&] {
      std::vector<f32> s(D), h1(256);
      for (int i = 0; i < D; i++) s[i] = e[i] / (1.f + expf(-e[i]));
      for (int k = 0; k < count; k++) {
        const __fp16* mw = (const __fp16*)(wts[k] + wmodw);
        for (int site = 0; site < 3; site++) {
          const __fp16* A = mw + (size_t)site * (256 * 2048 + 6144 * 256); const __fp16* B = A + 256 * 2048;
          for (int j = 0; j < 256; j++) { float a = 0; const __fp16* row = A + (size_t)j * 2048; for (int i = 0; i < D; i++) a += (float)row[i] * s[i]; h1[j] = a; }
          f32* out = &mod[((size_t)k * 9 + site * 3) * D];
          for (int o = 0; o < 6144; o++) { float a = lr[o]; const __fp16* row = B + (size_t)o * 256; for (int j = 0; j < 256; j++) a += (float)row[j] * h1[j]; out[o] = a; }
        }
        mdone.store(k + 1, std::memory_order_release);
      }
    });
    double t_conv = now();
    bool ok = true; int times[32];
    int firstRun = -1, lastRun = -1;   // blocks in skipmask are skipped (x passes through unchanged)
    for (int k = 0; k < count; k++) if (!((skipmask >> (first + k)) & 1)) { if (firstRun < 0) firstRun = k; lastRun = k; }
    for (int k = 0; k < count && ok; k++) {
      if ((skipmask >> (first + k)) & 1) continue;
      if (g_stop) { ok = false; break; }   // stop between DSP calls, never in the middle of one
      while (mdone.load(std::memory_order_acquire) <= k) {}
      L.meta[M_FIRST] = k == firstRun; L.meta[M_REL] = k == lastRun;   // VTCM, pool and power votes are held across the blocks of one run, released after the last
      double tm1 = now();
      int rc = 0; int rr = mega_block(h, L.meta, M_NMETA, rpcmem_to_fd(wts[k]), blobfd, (const uint8_t*)&mod[(size_t)k * 9 * D], 9 * D * 4, times, 32, &rc);
      if (getenv("MEGA_DBG") && k == 5) { char lp[256]; snprintf(lp, sizeof lp, "megarun_%d.log", first); FILE* lf = fopen(lp, "a"); if (lf) { fprintf(lf, "blk5 stages(us):"); for (int q = 0; q < 25; q++) fprintf(lf, " %d", times[q]); fprintf(lf, " | call %.1f ms\n", (now() - tm1) * 1e3); fclose(lf); } }
      if (rr || rc) { fprintf(stderr, "block %d failed r=0x%x rc=%d\n", first + k, rr, rc); ok = false; }
    }
    double t_dsp = now();
    mt.join();
    if (!ok) { printf("error\n"); fflush(stdout); continue; }
    { dma_buf_sync_ sy = {(0ull << 2) | 3}; ioctl(blobfd, DMA_BUF_IOCTL_SYNC_, &sy); }   // START|RW: invalidate CPU caches before reading the result
    bool xout32 = xo.size() > 4 && xo.compare(xo.size() - 4, 4, ".x32") == 0;
    if (!xout32) from_x32((const f32*)(blob + L.meta[M_X]), xin.data(), T);
    FILE* f = fopen(xo.c_str(), "wb"); fwrite(xout32 ? (const void*)(blob + L.meta[M_X]) : (const void*)xin.data(), 4, (size_t)D * T, f); fclose(f);
    double t_end = now();
    if (getenv("MEGA_DBG")) { char lp[256]; snprintf(lp, sizeof lp, "megarun_%d.log", first); FILE* lf = fopen(lp, "a"); if (lf) { fprintf(lf, "run: in+conv %.1f ms, blocks %.1f ms, out %.1f ms, total %.1f ms\n", (t_conv - ts) * 1e3, (t_dsp - t_conv) * 1e3, (t_end - t_dsp) * 1e3, (t_end - ts) * 1e3); fclose(lf); } }
    printf("done %.0f\n", (t_end - ts) * 1000); fflush(stdout);
  }
  mega_close(h);
  return 0;
}
