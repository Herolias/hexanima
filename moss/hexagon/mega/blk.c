/* One Anima DiT block (self-attn, cross-attn, MLP) on HMX, activations in crouton tiles, shared blob in DDR. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include "HAP_compute_res.h"
#include "HAP_perf.h"
#include "HAP_mem.h"
#include "HAP_power.h"
#include "mega.h"
void power_max(void *);
#include "gemm2.h"
void hmx_clr(void);
#include "pool.h"
#include "kern.h"
#include "blk_meta.h"

typedef uint16_t u16;
#define T16 1024          /* halfwords per h16 tile */
#define VT_TAB (7u << 20) /* tables at the top 1 MB of VTCM */

static u16 *g_et, *g_tabs, *g_tabs2; static HVX_Vector *g_scr;   /* tables (exp, 3 SiTU) + per-worker gather scratch */

static void vcp(void *d, const void *s, size_t bytes) {
  HVX_Vector *dv = (HVX_Vector *)d; const HVX_Vector *sv = (const HVX_Vector *)s; size_t n = bytes / 128;
  for (size_t i = 0; i < n; i++) { if ((i & 31) == 0 && i + 64 < n) Q6_l2fetch_AP((void *)(sv + i + 64), (uint64_t)128 | (32ull << 16) | (128ull << 32)); dv[i] = sv[i]; }
}


/* ---------------- simple parallel kernels ---------------- */
typedef struct { const float *x; u16 *h; const HVX_Vector *M, *N; int ns; } par_t;
static void p_ln(void *a, int i, int n) { par_t *p = (par_t *)a; for (int s = i; s < p->ns; s += n) k_rmsmod_s(p->x + (size_t)s * NCB * 1024, p->h + (size_t)s * NCB * T16, 1, p->M, p->N, 1e-6f); }

/* ---------------- attention (joint, NHEAD heads, last key tile masked) ---------------- */
typedef struct { const u16 *k, *v, *q; u16 *o; int nkb, h, nq; u16 *KT, *Vw, *Qh; } ah_t;
static void p_attn_setup(void *a, int idx, int n) {   /* build K^T and V(+ones) weight tiles for head h, copy Q tiles */
  ah_t *p = (ah_t *)a;
  for (int kb = idx; kb < p->nkb; kb += n)
    for (int j = 0; j < 4; j++) {
      const u16 *ks = p->k + ((size_t)kb * NCB + p->h * 4 + j) * T16; u16 *kd = p->KT + ((size_t)kb * 4 + j) * T16;
      k_transpose_tile(ks, kd);
      vcp(p->Vw + ((size_t)j * p->nkb + kb) * T16, p->v + ((size_t)kb * NCB + p->h * 4 + j) * T16, 2048);
    }
  for (int s = idx; s < p->nq; s += n) vcp(p->Qh + (size_t)s * 4 * T16, p->q + ((size_t)s * NCB + p->h * 4) * T16, 8192);
}
static void p_attn_flush(void *a, int idx, int n) { ah_t *p = (ah_t *)a; for (int s = idx; s < p->nq; s += n) vcp(p->o + ((size_t)s * NCB + p->h * 4) * T16, p->Qh + (size_t)s * 4 * T16, 8192); }
typedef struct { const u16 *S; u16 *P; int nkb; const HVX_Vector *km; } sm_t;
static void p_softmax(void *a, int i, int n) { sm_t *p = (sm_t *)a; k_softmax2p(p->S, p->P, p->nkb, g_et, g_scr + 8 * i, i * 16 / n, (i + 1) * 16 / n, p->km); }

static uint64_t g_acc[8];
#define PC() HAP_perf_get_time_us()
static u16 g_km[64] __attribute__((aligned(128)));
static void attn_run(const u16 *q, const u16 *k, const u16 *v, u16 *o, int nq, int nkv, int nvalid, uint8_t *vt) {
  const int nkb = nkv / 32;
  const int nv = nvalid - (nkb - 1) * 32;   /* valid keys in the last tile */
  const HVX_Vector *km = 0;
  if (nv < 32) { for (int l = 0; l < 64; l++) g_km[l] = (l >> 1) >= nv ? 0xFFFF : 0; km = (const HVX_Vector *)g_km; }
  u16 *KT = (u16 *)vt, *Vw = KT + (size_t)4 * nkb * T16, *Qh = Vw + (size_t)5 * nkb * T16;           /* Qh also receives O */
  u16 *Sb[2], *Pb[2]; Sb[0] = Qh + (size_t)nq * 4 * T16; Sb[1] = Sb[0] + (size_t)nkb * T16; Pb[0] = Sb[1] + (size_t)nkb * T16; Pb[1] = Pb[0] + (size_t)nkb * T16;
  u16 *Ob = Pb[1] + (size_t)nkb * T16;                                                                  /* 5 tiles: 4 + sum */
  for (size_t i = 0; i < (size_t)nkb * T16; i++) Vw[(size_t)4 * nkb * T16 + i] = 0x3C00;               /* ones n-block */
  for (int h = 0; h < NHEAD; h++) {
    ah_t a = {k, v, q, o, nkb, h, nq, KT, Vw, Qh};
    uint64_t c0 = PC(); pool_run(p_attn_setup, &a); g_acc[0] += PC() - c0;
    for (int s = 0; s <= nq; s++) {
      const int b = s & 1;
      uint64_t c1 = PC();
      if (s < nq) {
        for (int nb = 0; nb < nkb; nb++) {
          hmx_acc((uint8_t *)(Qh + (size_t)s * 4 * T16), (4 << 11) | 0x7C, (uint8_t *)(KT + (size_t)nb * 4 * T16), (4 << 11) - 1);
          hmx_store((uint8_t *)(Sb[b] + (size_t)nb * T16), 0);
        }
      }
      g_acc[1] += PC() - c1;
      sm_t sm = {Sb[b], Pb[b], nkb, km};
      uint64_t c2 = PC();
      if (s < nq) pool_start(p_softmax, &sm);
      if (s > 0) {   /* PV for slab s-1 while the workers run softmax(s) */
        const u16 *P = Pb[(s - 1) & 1];
        for (int j = 0; j < 5; j++) {
          for (int k0 = 0; k0 < nkb; k0 += 32) { int c = nkb - k0 < 32 ? nkb - k0 : 32; hmx_acc((uint8_t *)(P + (size_t)k0 * T16), (c << 11) | 0x7C, (uint8_t *)(Vw + ((size_t)j * nkb + k0) * T16), (c << 11) - 1); }
          hmx_store((uint8_t *)(Ob + (size_t)j * T16), 0);
        }
        uint64_t c3 = PC(); g_acc[2] += c3 - c2;
        k_rowscale_sum(Ob, 4, Ob + (size_t)4 * T16);
        g_acc[3] += PC() - c3;
        vcp(Qh + (size_t)(s - 1) * 4 * T16, Ob, 4 * 2048);   /* O replaces the (already consumed) Q slab */
      }
      uint64_t c4 = PC(); if (s < nq) pool_wait(); g_acc[4] += PC() - c4;
    }
    uint64_t c5 = PC(); pool_run(p_attn_flush, &a); g_acc[5] += PC() - c5;
  }
}

/* ---------------- block ---------------- */
static uint32_t us_now(void) { return (uint32_t)HAP_perf_get_time_us(); }

static uint32_t g_ctx; static uint8_t *g_vt; static int g_init; static u16 *g_et_h, *g_tabs_h, *g_tabs2_h;
void mega_blk_release(void) { if (!g_init) return; pool_quit(); if (g_vt) HAP_compute_res_release(g_ctx); g_vt = 0; g_init = 0; }
static int vtcm_get(void) {
  compute_res_attr_t attr; HAP_compute_res_attr_init(&attr);
  HAP_compute_res_attr_set_vtcm_param(&attr, 8u << 20, 1); HAP_compute_res_attr_set_hmx_param(&attr, 1);
  g_ctx = HAP_compute_res_acquire(&attr, 5000000); if (!g_ctx) return -1;
  g_vt = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&attr);
  g_et = (u16 *)(g_vt + VT_TAB); g_tabs = g_et + 32768; g_tabs2 = g_tabs + 3 * 32768; g_scr = (HVX_Vector *)(g_vt + VT_TAB + 393216);
  memcpy(g_et, g_et_h, 65536); memcpy(g_tabs, g_tabs_h, 3 * 65536); memcpy(g_tabs2, g_tabs2_h, 2 * 65536);
  hmx_scales_init(g_vt + (8u << 20) - 32768);   /* v73+ HMX output scale block: below the self-test tiles (last 16 KB), above the tables and scratch */
  return 0;
}

/* HMX self-test: one 32x32x32 fp16 tile of ones x ones must give 32.0 */
static int hmx_selftest(uint8_t *vt) {
  uint8_t *a = vt + (8u << 20) - 16384, *w = a + 2048, *o = w + 2048;
  HVX_Vector one = Q6_Vh_vsplat_R(0x3C003C00); HVX_Vector *pa = (HVX_Vector *)a, *pw = (HVX_Vector *)w, *po = (HVX_Vector *)o;
  for (int i = 0; i < 16; i++) { pa[i] = one; pw[i] = one; po[i] = Q6_V_vzero(); }
  asm volatile("barrier" ::: "memory");
  hmx_clr(); hmx_acc(a, (1 << 11) | 0x7C, w, (1 << 11) - 1); hmx_store(o, 0);
  volatile uint16_t *r = (volatile uint16_t *)o; return r[0] == 0x5000 ? 0 : (int)r[0] + 1;
}
typedef struct { const u16 *t; u16 *t2; float *r; const float *a, *b, *c, *d, *e, *f; float *o; int ns, nct, T; } tej_t;
static void j_t2r(void *a, int i, int n) { tej_t *p = (tej_t *)a; te_t2r(p->t, p->r, p->ns, p->nct, i, n); }
static void j_r2t(void *a, int i, int n) { tej_t *p = (tej_t *)a; te_r2t(p->r, p->t2, p->ns, p->nct, i, n); }
static void j_conv(void *a, int i, int n) { tej_t *p = (tej_t *)a; te_conv_silu(p->a, p->o, p->b, p->T, 8192, i, n); }
static void j_delta(void *a, int i, int n) { tej_t *p = (tej_t *)a; te_delta(p->a, p->b, p->c, p->o, p->d, p->e, p->f, p->T, i, n); }
static void j_aprep(void *a, int i, int n) { tej_t *p = (tej_t *)a; te_attn_prep(p->r, p->o, p->a, p->b, p->c, p->T, i, n); }
static void j_attn(void *a, int i, int n) { tej_t *p = (tej_t *)a; te_attn(p->a, p->b, p->o, p->T, i, n); }
static uint8_t *g_blob;
static int mega_block_impl(remote_handle64 hd, const int *meta, int nmeta, int wfd, const uint8_t *modp, int modlen, int *times, int ntimes, int *rc) {
  uint8_t *blob = g_blob; if (!blob) { *rc = -6; return 0; }
  uint32_t te0 = us_now();
  *rc = 0; for (int i = 0; i < ntimes; i++) times[i] = 0;
  void *wp = 0; uint64 wpa = 0;
  if (HAP_mmap_get(wfd, &wp, &wpa) || !wp) { *rc = -5; return 0; }
  if (meta[M_STOP] == 99) return 0;
  const uint8_t *wts = (const uint8_t *)wp + meta[M_WOFF];
  if (!g_init) {
    g_et_h = (u16 *)memalign(128, 65536); g_tabs_h = (u16 *)memalign(128, 3 * 65536);
    { u16 *t2 = (u16 *)memalign(128, 65536); tab_init(g_et_h, t2); free(t2); }
    tab_init_situ(g_tabs_h);
    g_tabs2_h = (u16 *)memalign(128, 2 * 65536); tab_init_swiglu(g_tabs2_h);
    g_init = 1;
  }
  gemm_set_q8(1);
  if (!pool_n()) pool_init(meta[M_NW] ? meta[M_NW] : 3);
  if (!g_vt && vtcm_get()) { *rc = -1; return 0; }
  uint32_t ctx = g_ctx; uint8_t *vt = g_vt;
  if (meta[M_FIRST]) { HAP_power_request_t rq; memset(&rq, 0, sizeof rq); rq.type = HAP_power_set_HMX; rq.hmx.power_up = TRUE; HAP_power_set((void *)hd, &rq); power_max((void *)hd); }
  if (HAP_compute_res_hmx_lock(ctx)) { *rc = -2; return 0; }
  { int tries = 0, tr;
    while ((tr = hmx_selftest(vt)) != 0 && tries < 3) {
      tries++; HAP_compute_res_hmx_unlock(ctx);
      { uint64_t t0 = HAP_perf_get_time_us(); while (HAP_perf_get_time_us() - t0 < 30000) {} }
      if (HAP_compute_res_hmx_lock(ctx)) { *rc = -2; return 0; }
    }
    if (tr) { *rc = -77; HAP_compute_res_hmx_unlock(ctx); return 0; } }
  hmx_clr();
  g_ncb = meta[M_NCB] ? meta[M_NCB] : 96;
  const int ntok = meta[M_NTOK], ns = ntok / 32;
#define BUF(m) (blob + meta[m])
#define WB(o) ((const u16 *)(wts + (o)))
  float *x = (float *)BUF(M_X); const float *mod = (const float *)modp, *gain = (const float *)(wts + W_GAIN), *cosT = (const float *)BUF(M_COS), *sinT = (const float *)BUF(M_SIN);
  u16 *h16 = (u16 *)BUF(M_H), *q16 = (u16 *)BUF(M_Q), *k16 = (u16 *)BUF(M_K), *v16 = (u16 *)BUF(M_V), *o16 = (u16 *)BUF(M_O), *t16 = (u16 *)BUF(M_T), *mid = (u16 *)BUF(M_MID);
  u16 *Lq = (u16 *)BUF(M_LQ), *Lo = (u16 *)BUF(M_LO), *Lu = (u16 *)BUF(M_LU), *Ld = (u16 *)BUF(M_LD);
  static HVX_Vector LM[MAXNCB] __attribute__((aligned(128))), LN[MAXNCB] __attribute__((aligned(128)));
  static float g_zero[4096] __attribute__((aligned(128)));
  if (ntimes > 24) times[24] = (int)(us_now() - te0);
  int ti = 0; uint32_t t0 = us_now();
#define TICK() do { uint32_t t1 = us_now(); if (ti < ntimes) times[ti++] = (int)(t1 - t0); t0 = t1; if (meta[M_STOP] && ti >= meta[M_STOP]) goto done; } while (0)
  par_t P; memset(&P, 0, sizeof P); P.ns = ns; P.x = x; P.h = h16; P.M = LM; P.N = LN;
  const float *sh_a = mod, *sc_a = mod + DIM, *gt_a = mod + 2 * DIM, *sh_f = mod + 3 * DIM, *sc_f = mod + 4 * DIM, *gt_f = mod + 5 * DIM;
  const gemm_geo_t gq = {NCB, NCB + 4, 0, Lq, 12, 4}, gk = {NCB, NCB + 4, 0, Lq + 4 * T16, 12, 4}, gv = {NCB, NCB + 4, 0, Lq + 8 * T16, 12, 4};

  if (meta[M_MODE] == 20 || meta[M_MODE] == 21) {   /* Qwen3.5 layer: x += attn(rms(x)); x += ffn(rms(x)); everything on 4096-wide tiles */
    const te_hdr_t *H = (const te_hdr_t *)wts; const float *vec = (const float *)(wts + H->voff);
    const int T = meta[M_NVALID]; float *R = (float *)BUF(M_ROWS); const int full = meta[M_MODE] == 21;
    const float *ones = mod + 2 * 4096;
    const int igu = full ? 3 : 4, idn = full ? 4 : 5;
    k_rms_prep(LM, LN, vec, g_zero, g_zero); pool_run(p_ln, &P); TICK();
    tej_t J; memset(&J, 0, sizeof J); J.ns = ns; J.T = T;
    gemm_epi_t eo; memset(&eo, 0, sizeof eo); eo.mode = 1; eo.x32 = x; eo.gate = ones;
    if (!full) {
      float *R0 = R, *R1 = R0 + (size_t)ntok * 8192, *R2 = R1 + (size_t)ntok * 8192, *R3 = R2 + (size_t)ntok * 4096, *R4 = R3 + (size_t)ntok * 64;
      gemm_set_q8(H->fmt[0]); gemm_run_geo(h16, ntok, 4096, WB(H->off[0]), 8192, mid, vt, 0, 0);
      gemm_set_q8(H->fmt[1]); gemm_run_geo(h16, ntok, 4096, WB(H->off[1]), 4096, q16, vt, 0, 0);
      gemm_set_q8(H->fmt[2]); gemm_run_geo(h16, ntok, 4096, WB(H->off[2]), 64, t16, vt, 0, 0); TICK();
      J.t = mid; J.r = R0; J.nct = 256; pool_run(j_t2r, &J); J.t = q16; J.r = R2; J.nct = 128; pool_run(j_t2r, &J); J.t = t16; J.r = R3; J.nct = 2; pool_run(j_t2r, &J); TICK();
      memset(R4, 0, (size_t)ntok * 4096 * 4);
      J.a = R0; J.o = R1; J.b = vec + 8192; pool_run(j_conv, &J);
      J.a = R1; J.b = R3; J.c = R2; J.o = R4; J.d = vec + 8192 + 32768; J.e = J.d + 32; J.f = J.d + 64; pool_run(j_delta, &J); TICK();
      J.r = R4; J.t2 = o16; J.nct = 128; pool_run(j_r2t, &J); TICK();
      gemm_set_q8(H->fmt[3]); gemm_run_geo(o16, ntok, 4096, WB(H->off[3]), 4096, t16, vt, &eo, 0); TICK();
    } else {
      float *Rq = R, *Rkv = Rq + (size_t)ntok * 8192, *Ro = Rkv + (size_t)ntok * 2048;
      gemm_set_q8(H->fmt[0]); gemm_run_geo(h16, ntok, 4096, WB(H->off[0]), 8192, mid, vt, 0, 0);
      gemm_set_q8(H->fmt[1]); gemm_run_geo(h16, ntok, 4096, WB(H->off[1]), 2048, q16, vt, 0, 0); TICK();
      J.t = mid; J.r = Rq; J.nct = 256; pool_run(j_t2r, &J); J.t = q16; J.r = Rkv; J.nct = 64; pool_run(j_t2r, &J); TICK();
      memset(Ro, 0, (size_t)ntok * 4096 * 4);
      J.r = Rq; J.o = Rkv; J.a = vec + 8192; J.b = vec + 8192 + 256; J.c = (const float *)BUF(M_CS); pool_run(j_aprep, &J);
      J.a = Rq; J.b = Rkv; J.o = Ro; pool_run(j_attn, &J); TICK();
      J.r = Ro; J.t2 = o16; J.nct = 128; pool_run(j_r2t, &J); TICK();
      gemm_set_q8(H->fmt[2]); gemm_run_geo(o16, ntok, 4096, WB(H->off[2]), 4096, t16, vt, &eo, 0); TICK();
    }
    /* ffn */
    k_rms_prep(LM, LN, vec + 4096, g_zero, g_zero); pool_run(p_ln, &P);
    { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 6; e.tabs = g_tabs2; e.scr = g_scr; gemm_set_q8(H->fmt[igu]);
      gemm_run_geo(h16, ntok, 4096, WB(H->off[igu]), 24576, mid, vt, &e, 0); }
    { gemm_epi_t e1; memset(&e1, 0, sizeof e1); e1.mode = 4; e1.x32 = x; e1.gate = ones; e1.prev = t16; gemm_set_q8(H->fmt[idn]);
      const gemm_geo_t d0 = {384, 384, 0, 0, 0, 0}, d1 = {384, 384, 192, 0, 0, 0};
      gemm_run_geo(mid, ntok, 6144, WB(H->off[idn]), 4096, t16, vt, 0, &d0);
      gemm_run_geo(mid, ntok, 6144, WB(H->off[idn]), 4096, t16, vt, &e1, &d1); } TICK();
    gemm_set_q8(1); goto done;
  }
  if (meta[M_MODE] == 30) {   /* final norm: h16 = rms(x) * w (pack = f32 weight vector) */
    k_rms_prep(LM, LN, (const float *)wts, g_zero, g_zero); pool_run(p_ln, &P); TICK(); goto done;
  }
  if (meta[M_MODE] == 11) { *rc = 111; gemm_dq_var(meta[M_STOP]); gemm_dq_tiles((uint8_t *)mid, wts, 16, meta[M_FMT]); goto done; }   /* dequant test: first 16 tiles -> mid */
  if (meta[M_MODE] == 10) {   /* Qwen3.5 FFN: x += down(silu(gate(rms(x)))*up(rms(x))); pack: [gate/up interleaved q4a nb 768 kb 128][down q8 nb 128 kb 384][post norm w f32 4096] */
    const size_t O_DN = 75497472u, O_G = O_DN + 56623104u;
    k_rms_prep(LM, LN, (const float *)(wts + O_G), g_zero, g_zero);
    pool_run(p_ln, &P); TICK();
    { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = meta[M_FMT] == 1 ? 0 : 6; e.tabs = g_tabs2; e.scr = g_scr; gemm_set_q8(2);
      gemm_run_geo(h16, ntok, 4096, WB(0), 24576, mid, vt, &e, 0); gemm_set_q8(1); } TICK();
    { gemm_epi_t e1; memset(&e1, 0, sizeof e1); e1.mode = 4; e1.x32 = x; e1.gate = (const float *)(mod + 2 * 4096); e1.prev = t16;
      const gemm_geo_t d0 = {384, 384, 0, 0, 0, 0}, d1 = {384, 384, 192, 0, 0, 0};
      gemm_run_geo(mid, ntok, 6144, WB(O_DN), 4096, t16, vt, 0, &d0);
      gemm_run_geo(mid, ntok, 6144, WB(O_DN), 4096, t16, vt, &e1, &d1); } TICK();
    goto done;
  }
  if (meta[M_MODE] == 2) {   /* text projection: h16 = text tiles [slab][128 cb]; weights = pack at offset 0 [96 nb][132 kb] */
    gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 1; e.x32 = x; e.gate = gt_a;
    const gemm_geo_t gp = {4096 / 32, 4096 / 32 + 4, 0, Lq, 4, 4};
    gemm_run_geo(h16, ntok, 4096 + 128, WB(0), DIM, t16, vt, &e, &gp); TICK(); goto done;
  }
  /* ---- attention half ---- */
  k_rms_prep(LM, LN, gain + 256, sh_a, sc_a);
  pool_run(p_ln, &P); TICK();
  const int nolora = meta[M_MODE] == 1;
  if (!nolora) gemm_run(h16, ntok, DIM, WB(W_AQKV), 384, Lq, vt); TICK();
  { gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 3; e.rope = 1; e.gain = gain; e.cosT = cosT; e.sinT = sinT;
    gemm_run_geo(h16, ntok, DIM + 128, WB(W_Q), DIM, q16, vt, &e, &gq);
    e.gain = gain + 128; gemm_run_geo(h16, ntok, DIM + 128, WB(W_K), DIM, k16, vt, &e, &gk);
    gemm_run_geo(h16, ntok, DIM + 128, WB(W_V), DIM, v16, vt, 0, &gv); } TICK();
  attn_run(q16, k16, v16, o16, ns, ntok, meta[M_NVALID], vt); TICK();
  { if (!nolora) gemm_run(o16, ntok, DIM, WB(W_AO), 128, Lo, vt);
    gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 1; e.x32 = x; e.gate = gt_a;
    const gemm_geo_t go = {NCB, NCB + 4, 0, Lo, 4, 4};
    gemm_run_geo(o16, ntok, DIM + 128, WB(W_O), DIM, t16, vt, &e, &go); } TICK();
  /* ---- FFN half ---- */
  k_rms_prep(LM, LN, gain + 256 + DIM, sh_f, sc_f);
  pool_run(p_ln, &P); TICK();
  { if (!nolora) gemm_run(h16, ntok, DIM, WB(W_AUG), 128, Lu, vt);
    gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 5; e.tabs = g_tabs; e.scr = g_scr;
    const gemm_geo_t gu = {NCB, NCB + 4, 0, Lu, 4, 4};
    gemm_run_geo(h16, ntok, DIM + 128, WB(W_UG), 16384, mid, vt, &e, &gu); } TICK();
  { if (!nolora) gemm_run(mid, ntok, 8192, WB(W_ADN), 128, Ld, vt);
    const gemm_geo_t g0 = {256, 260, 0, 0, 0, 0}, g1 = {256, 260, 128, Ld, 4, 4};
    gemm_run_geo(mid, ntok, 4096, WB(W_DN), DIM, t16, vt, 0, &g0);
    gemm_epi_t e; memset(&e, 0, sizeof e); e.mode = 4; e.x32 = x; e.gate = gt_f; e.prev = t16;
    gemm_run_geo(mid, ntok, 4096 + 128, WB(W_DN), DIM, t16, vt, &e, &g1); } TICK();
done:
  for (int i = 0; i < 6 && 17 + i < ntimes; i++) times[17 + i] = (int)g_acc[i];
  memset(g_acc, 0, sizeof g_acc);
  HAP_compute_res_hmx_unlock(ctx);
  if (meta[M_REL]) { pool_quit(); HAP_compute_res_release(g_ctx); g_vt = 0; }
  return 0;
}

static void cache_sync(void *p, size_t bytes) { for (char *c = (char *)p; c < (char *)p + bytes; c += 32) Q6_dccleaninva_A(c); }
int mega_block(remote_handle64 hd, const int *meta, int nmeta, int wfd, int bfd, const uint8_t *mod, int modlen, int *times, int ntimes, int *rc) {
  void *bp = 0; uint64 bpa = 0;
  g_ncb = meta[M_NCB] ? meta[M_NCB] : 96;
  if (HAP_mmap_get(bfd, &bp, &bpa) || !bp) { *rc = -7; return 0; }
  g_blob = (uint8_t *)bp;
  const size_t xb = (size_t)meta[M_NTOK] * DIM * 4;
  if (meta[M_FIRST]) { cache_sync(g_blob + meta[M_X], xb); cache_sync(g_blob + meta[M_COS], (size_t)meta[M_NTOK] / 32 * 8192); cache_sync(g_blob + meta[M_SIN], (size_t)meta[M_NTOK] / 32 * 8192); }
  int r = mega_block_impl(hd, meta, nmeta, wfd, mod, modlen, times, ntimes, rc);
  if (meta[M_REL]) cache_sync(g_blob + meta[M_X], xb);
  HAP_mmap_put(bfd);
  if (*rc != -5) HAP_mmap_put(wfd);
  return r;
}
