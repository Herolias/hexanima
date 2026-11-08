#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include "gemm2.h"
#include "pool.h"
#include "kern.h"
#define KCH 32
#include "qurt.h"

void hmx_store_v69(uint8_t *, uint32_t);
void hmx_store_v73(uint8_t *, uint32_t, const uint8_t *);
static uint8_t *g_hmx_scales; static int g_hmx_cvt = -1;
int hmx_cvt(void) {   /* 1 = v73+ DSP (QuRT arch version low byte 0x73, 0x75, 0x79, 0x81...) */
#if __HEXAGON_ARCH__ >= 73
  g_hmx_cvt = 1;
#else
  if (g_hmx_cvt < 0) { qurt_arch_version_t v; g_hmx_cvt = !qurt_sysenv_get_arch_version(&v) && (v.arch_version & 0xff) >= 0x73; }
#endif
  return g_hmx_cvt;
}
void hmx_scales_init(uint8_t *p) { HVX_Vector *v = (HVX_Vector *)p; v[0] = Q6_V_vsplat_R(0x3c00); v[1] = Q6_V_vzero(); g_hmx_scales = p; hmx_cvt(); }
void hmx_store(uint8_t *out, uint32_t ctrl) {
#if __HEXAGON_ARCH__ >= 73
  hmx_store_v73(out, ctrl, g_hmx_scales);
#else
  if (g_hmx_cvt > 0) hmx_store_v73(out, ctrl, g_hmx_scales); else hmx_store_v69(out, ctrl);
#endif
}

static void vcopy_range(uint8_t *dst, const uint8_t *src, size_t bytes) {  /* 128B aligned, bytes multiple of 128 */
  HVX_Vector *d = (HVX_Vector *)dst; const HVX_Vector *s = (const HVX_Vector *)src; size_t n = bytes / 128;
  for (size_t i = 0; i < n; i++) {
    if ((i & 31) == 0 && i + 64 < n) Q6_l2fetch_AP((void *)(s + i + 64), (uint64_t)128 | (32ull << 16) | (128ull << 32));
    d[i] = s[i];
  }
}
static int g_q8;   /* 0 fp16 tiles, 1 q8 records (1152 B), 2 q4a records (768 B: scale vec, min vec, 512 B nibbles) */
static volatile int g_dbg;   /* 1: no overlap of workers and HMX, 2: barriers around the HMX loop, 4: read back one vector of each HMX output tile */
void gemm_set_dbg(int f) { g_dbg = f; }
void gemm_set_q8(int on) { g_q8 = on; }
static inline size_t tile_bytes(int f) { return f == 2 ? 768 : f == 1 ? 1152 : 2048; }
typedef struct { uint8_t *dst; const uint8_t *src; size_t bytes; int q8; } cpy_t;   /* q8 field = format 0/1/2 */   /* bytes = destination bytes; q8: src tiles are 1152 B q8 records */
/* q8 tile record -> 2 KB fp16 tile. Bytes are offset-binary (u = q + 128): hf bits 0x6400|u = 1024 + u exactly, minus 1152 (exact in qf16) = q, times the per-column scale vector. (No int16 -> hf convert before v73.) */
static void dq_tile(uint8_t *dst, const uint8_t *src) {   /* qf32 route, see dq4_tile: the qf16 subtract is wrong for some integers on V69 */
  const HVX_Vector S = *(const HVX_Vector *)src, M = Q6_Vh_vsplat_R(0x6400), C = Q6_Vh_vsplat_R(0x6480);
  const HVX_VectorPair Cw = Q6_Wqf32_vmpy_VhfVhf(S, C);   /* 1152 * S */
  const HVX_Vector Clo = Q6_V_lo_W(Cw), Chi = Q6_V_hi_W(Cw);
  const HVX_Vector *q = (const HVX_Vector *)(src + 128); HVX_Vector *d = (HVX_Vector *)dst;
  for (int j = 0; j < 8; j++) {
    HVX_VectorPair P = Q6_Wuh_vunpack_Vub(q[j]);
    HVX_Vector u2[2] = {Q6_V_vor_VV(Q6_V_lo_W(P), M), Q6_V_vor_VV(Q6_V_hi_W(P), M)};
    for (int t = 0; t < 2; t++) {
      const HVX_VectorPair pr = Q6_Wqf32_vmpy_VhfVhf(u2[t], S);
      d[2 * j + t] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(Q6_Vqf32_vsub_Vqf32Vqf32(Q6_V_hi_W(pr), Chi), Q6_Vqf32_vsub_Vqf32Vqf32(Q6_V_lo_W(pr), Clo)));
    }
  }
}
static volatile int g_dq4var;
/* q4a record -> fp16 tile. Do NOT use qf16 for (1024+u)-1024: on V69 the qf16 subtract returns u+1 for u = 4, 8, 12. Use the widening qf32 product (1024+u)*S and add Mn - 1024*S in qf32. */
static void dq4_tile(uint8_t *dst, const uint8_t *src) {
  const HVX_Vector S = *(const HVX_Vector *)src, Mn = *(const HVX_Vector *)(src + 128), M = Q6_Vh_vsplat_R(0x6400), K = Q6_Vh_vsplat_R(0x0F0F), ONE = Q6_Vh_vsplat_R(0x3C00);
  const HVX_VectorPair Sw = Q6_Wqf32_vmpy_VhfVhf(S, M), Mw = Q6_Wqf32_vmpy_VhfVhf(Mn, ONE);
  const HVX_Vector Clo = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_V_lo_W(Mw), Q6_V_lo_W(Sw)), Chi = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_V_hi_W(Mw), Q6_V_hi_W(Sw));
  const HVX_Vector *q = (const HVX_Vector *)(src + 256); HVX_Vector *d = (HVX_Vector *)dst;
  for (int m = 0; m < 4; m++) {
    const HVX_Vector v = q[m];
    const HVX_Vector nib[2] = {Q6_V_vand_VV(v, K), Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(v, 4), K)};
    for (int s = 0; s < 2; s++) {
      HVX_VectorPair P = Q6_Wuh_vunpack_Vub(nib[s]);
      HVX_Vector u2[2] = {Q6_V_vor_VV(Q6_V_lo_W(P), M), Q6_V_vor_VV(Q6_V_hi_W(P), M)};
      for (int t = 0; t < 2; t++) {
        const HVX_VectorPair pr = Q6_Wqf32_vmpy_VhfVhf(u2[t], S);
        d[4 * m + 2 * s + t] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(pr), Chi), Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(pr), Clo)));
      }
    }
  }
}
static void dq4_range(uint8_t *dst, const uint8_t *src, size_t ntiles) {
  for (size_t t = 0; t < ntiles; t++) {
    if ((t & 7) == 0 && t + 16 < ntiles) Q6_l2fetch_AP((void *)(src + (t + 16) * 768), (uint64_t)128 | (48ull << 16) | (128ull << 32));
    dq4_tile(dst + t * 2048, src + t * 768);
  }
}
static void dq_range(uint8_t *dst, const uint8_t *src, size_t ntiles) {
  for (size_t t = 0; t < ntiles; t++) {
    if ((t & 7) == 0 && t + 16 < ntiles) Q6_l2fetch_AP((void *)(src + (t + 16) * 1152), (uint64_t)128 | (72ull << 16) | (128ull << 32));
    dq_tile(dst + t * 2048, src + t * 1152);
  }
}
static void split_copy(const cpy_t *c, int idx, int n) {
  if (c->q8 == 2) {
    size_t nt = c->bytes / 2048, per = (nt + n - 1) / n, t0 = per * idx; if (t0 >= nt) return;
    dq4_range(c->dst + t0 * 2048, c->src + t0 * 768, nt - t0 < per ? nt - t0 : per); return;
  }
  if (c->q8) {
    size_t nt = c->bytes / 2048, per = (nt + n - 1) / n, t0 = per * idx; if (t0 >= nt) return;
    dq_range(c->dst + t0 * 2048, c->src + t0 * 1152, nt - t0 < per ? nt - t0 : per); return;
  }
  size_t per = ((c->bytes / 128 + n - 1) / n) * 128, o = per * idx;
  if (o >= c->bytes) return;
  size_t len = c->bytes - o < per ? c->bytes - o : per;
  vcopy_range(c->dst + o, c->src + o, len);
}
typedef struct {
  cpy_t in, in2;                  /* next weight panel: two n-block segments (in2.bytes==0: single segment in) */
  uint8_t *flush_src; uint8_t *flush_dst_base; int nflush, dst_stride_tiles;  /* flush RG*[2 tiles] -> O */
  size_t dst_tile0; int rg;
  const gemm_epi_t *epi; int flush_cb0; size_t flush_slab0;   /* epilogue: first tile column (2p) and first slab of the flushed panel */
} job_t;
static void job_fn(void *a, int idx, int n) {
  job_t *j = (job_t *)a;
  if (j->in.bytes) split_copy(&j->in, idx, n);
  if (j->in2.bytes) split_copy(&j->in2, idx, n);
  const gemm_epi_t *e = j->epi;
  for (int r = idx; r < j->nflush; r += n) {   /* each slab: 2 adjacent tiles = 4 KB */
    uint8_t *tiles = j->flush_src + (size_t)r * 4096;
    if (e && e->mode == 1) {
      for (int t = 0; t < 2; t++) k_resadd_tile(e->x32 + ((j->flush_slab0 + r) * NCB + j->flush_cb0 + t) * 1024, (const uint16_t *)(tiles + t * 2048), e->gate + (j->flush_cb0 + t) * 32);
      continue;
    }
    if (e && e->mode == 4) {
      const uint16_t *pv = (const uint16_t *)j->flush_dst_base + (j->dst_tile0 + (size_t)r * j->dst_stride_tiles) * 1024;
      for (int t = 0; t < 2; t++) k_resadd2_tile(e->x32 + ((j->flush_slab0 + r) * NCB + j->flush_cb0 + t) * 1024, (const uint16_t *)(tiles + t * 2048), pv + t * 1024, e->gate + (j->flush_cb0 + t) * 32);
      continue;
    }
    if (e && e->mode == 5) {
      k_situ_tile((const uint16_t *)tiles, (const uint16_t *)(tiles + 2048), (uint16_t *)tiles, e->tabs, e->scr + 12 * idx);
      vcopy_range(j->flush_dst_base + (j->dst_tile0 + (size_t)r * j->dst_stride_tiles) * 2048, tiles, 2048);
      continue;
    }
    if (e && e->mode == 6) {
      k_swiglu_tile((const uint16_t *)tiles, (const uint16_t *)(tiles + 2048), (uint16_t *)tiles, e->tabs, e->scr + 8 * idx);
      vcopy_range(j->flush_dst_base + (j->dst_tile0 + (size_t)r * j->dst_stride_tiles) * 2048, tiles, 2048);
      continue;
    }
    if (e && e->mode == 2) k_gelu2((uint16_t *)tiles, 32, e->t2, e->scr + 8 * idx);
    vcopy_range(j->flush_dst_base + (j->dst_tile0 + (size_t)r * j->dst_stride_tiles) * 2048, tiles, 4096);
    if (e && e->mode == 3 && (j->flush_cb0 & 2)) {   /* second panel of a head: all 4 tiles are in O now */
      const size_t slab = j->flush_slab0 + r;
      uint16_t *x4 = (uint16_t *)j->flush_dst_base + ((slab * (size_t)j->dst_stride_tiles) + j->flush_cb0 - 2) * 1024;
      k_rmsrope_head(x4, e->gain, e->rope ? e->cosT + slab * 2048 : 0, e->rope ? e->sinT + slab * 2048 : 0, e->rope);
    }
  }
}

typedef struct { uint8_t *dst; const uint8_t *src; size_t row_bytes, src_stride; int nrows; const uint8_t *src2; size_t bytes2, stride2; } actjob_t;
static void act_fn(void *a, int idx, int n) {
  actjob_t *j = (actjob_t *)a;
  for (int r = idx; r < j->nrows; r += n) {
    uint8_t *d = j->dst + (size_t)r * (j->row_bytes + j->bytes2);
    vcopy_range(d, j->src + (size_t)r * j->src_stride, j->row_bytes);
    if (j->bytes2) vcopy_range(d + j->row_bytes, j->src2 + (size_t)r * j->stride2, j->bytes2);
  }
}
/* weight panel p (n blocks 2p, 2p+1) of K range [kofs, kofs+kb) into dst: contiguous when the range is the whole K, else two segments */
static void set_panel(job_t *j, uint8_t *dst, const uint16_t *W, int p, int kb, int ldw, int kofs, size_t panelB) {
  const size_t tb = tile_bytes(g_q8);
  if (ldw == kb && kofs == 0) { j->in = (cpy_t){dst, (const uint8_t *)W + (size_t)p * kb * 2 * tb, panelB, g_q8}; return; }
  for (int jn = 0; jn < 2; jn++) {
    cpy_t c = {dst + (size_t)jn * kb * 2048, (const uint8_t *)W + (((size_t)(2 * p + jn) * ldw) + kofs) * tb, (size_t)kb * 2048, g_q8};
    if (jn == 0) j->in = c; else j->in2 = c;
  }
}

void hmx_clr(void);
void gemm_run_geo(const uint16_t *A, int M, int K, const uint16_t *W, int N, uint16_t *O, uint8_t *vtcm, const gemm_epi_t *epi, const gemm_geo_t *geo) {
  hmx_clr();
  const int lda = geo ? geo->lda : K / 32, ldw = geo ? geo->ldw : K / 32, kofs = geo ? geo->kofs : 0;
  const int kb = K / 32, nb = N / 32, np = nb / 2, rs = M / 32;
  const int half = epi && (epi->mode == 5 || epi->mode == 6), onb = half ? nb / 2 : nb;   /* mode 5: one output tile per panel */
  const size_t slabB = (size_t)kb * 2048, panelB = (size_t)kb * 4096;
  /* row group = as many slabs as VTCM allows (weights are re-read once per group): budget 7 MB minus 2 weight panels and the out buffers, equalised over the groups */
  int budget = (7 << 20) - 2 * (int)panelB - 65536; int RGmax = budget / (int)(slabB + 8192); if (RGmax < 1) RGmax = 1;
  int ngr = (rs + RGmax - 1) / RGmax; int RG = (rs + ngr - 1) / ngr;
  uint8_t *act = vtcm, *wb[2] = {vtcm + (size_t)RG * slabB, vtcm + (size_t)RG * slabB + panelB};
  uint8_t *ob[2] = {wb[1] + panelB, wb[1] + panelB + (size_t)RG * 4096};
  for (int g0 = 0; g0 < rs; g0 += RG) {
    const int rg = rs - g0 < RG ? rs - g0 : RG;
    if (lda == kb && kofs == 0 && !(geo && geo->A2)) {
      cpy_t ca = {act, (const uint8_t *)A + (size_t)g0 * slabB, (size_t)rg * slabB};
      pool_run((pool_fn)split_copy, &ca);   /* NOTE: split_copy has the pool_fn shape with a cpy_t* arg */
    } else {
      const int nk2 = geo && geo->A2 ? geo->nk2 : 0;
      actjob_t aj = {act, (const uint8_t *)A + ((size_t)g0 * lda + kofs) * 2048, (size_t)(kb - nk2) * 2048, (size_t)lda * 2048, rg, nk2 ? (const uint8_t *)geo->A2 + (size_t)g0 * geo->lda2 * 2048 : 0, (size_t)nk2 * 2048, nk2 ? (size_t)geo->lda2 * 2048 : 0};
      pool_run(act_fn, &aj);
    }
    job_t j; memset(&j, 0, sizeof j);
    set_panel(&j, wb[0], W, 0, kb, ldw, kofs, panelB);
    pool_run(job_fn, &j);
    for (int p = 0; p < np; p++) {
      job_t nj; memset(&nj, 0, sizeof nj);
      if (p + 1 < np) set_panel(&nj, wb[(p + 1) & 1], W, p + 1, kb, ldw, kofs, panelB);
      if (p > 0) { nj.flush_src = ob[(p - 1) & 1]; nj.flush_dst_base = (uint8_t *)O; nj.nflush = rg; nj.dst_stride_tiles = onb; nj.dst_tile0 = (size_t)g0 * onb + (half ? (p - 1) : 2 * (p - 1)); nj.epi = epi; nj.flush_cb0 = 2 * (p - 1); nj.flush_slab0 = g0; }
      pool_start(job_fn, &nj);
      if (g_dbg & 1) pool_wait();
      uint8_t *w = wb[p & 1], *o = ob[p & 1];
      for (int r = 0; r < rg; r++)
        for (int jn = 0; jn < 2; jn++) {
          for (int k0 = 0; k0 < kb; k0 += KCH) {
            int c = kb - k0 < KCH ? kb - k0 : KCH;
            hmx_acc(act + r * slabB + (size_t)k0 * 2048, (c << 11) | 0x7C, w + ((size_t)jn * kb + k0) * 2048, (c << 11) - 1);
          }
          hmx_store(o + ((size_t)r * 2 + jn) * 2048, 0);
          if (g_dbg & 4) { volatile HVX_Vector *rb = (volatile HVX_Vector *)(o + ((size_t)r * 2 + jn) * 2048); HVX_Vector t = *rb; (void)t; }
        }
      if (g_dbg & 2) { asm volatile("barrier" ::: "memory"); asm volatile("syncht" ::: "memory"); }
      if (!(g_dbg & 1)) pool_wait(); else pool_wait();
      if (g_dbg & 2) { asm volatile("barrier" ::: "memory"); }
    }
    job_t fj; memset(&fj, 0, sizeof fj);
    fj.flush_src = ob[(np - 1) & 1]; fj.flush_dst_base = (uint8_t *)O; fj.nflush = rg; fj.dst_stride_tiles = onb; fj.dst_tile0 = (size_t)g0 * onb + (half ? (np - 1) : 2 * (np - 1)); fj.epi = epi; fj.flush_cb0 = 2 * (np - 1); fj.flush_slab0 = g0;
    pool_run(job_fn, &fj);
  }
}

void gemm_run(const uint16_t *A, int M, int K, const uint16_t *W, int N, uint16_t *O, uint8_t *vtcm) { gemm_run_ex(A, M, K, W, N, O, vtcm, 0); }

void gemm_run_ex(const uint16_t *A, int M, int K, const uint16_t *W, int N, uint16_t *O, uint8_t *vtcm, const gemm_epi_t *epi) { gemm_run_geo(A, M, K, W, N, O, vtcm, epi, 0); }

void gemm_dq_var(int v) { g_dq4var = v; }
void gemm_dq_tiles(uint8_t *dst, const uint8_t *src, size_t ntiles, int fmt) {
  cpy_t c = {dst, src, ntiles * 2048, fmt};
  pool_run((pool_fn)split_copy, &c);
}
