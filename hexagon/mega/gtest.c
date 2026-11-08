#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "HAP_compute_res.h"
#include "HAP_perf.h"
#include "HAP_power.h"
#include "mega.h"
void power_max(void *);
int power_hmx_rc(void);
#include "gemm2.h"
#include "pool.h"
static float gen(unsigned a, unsigned b, unsigned s) { unsigned h = a * 2654435761u + b * 40503u + s; h ^= h >> 13; h *= 0x5bd1e995; h ^= h >> 15; return ((int)(h & 255) - 128) / 256.f; }
static inline size_t tin(int r, int c) { return ((r & 31) >> 1) * 64 + (c & 31) * 2 + (r & 1); }
int mega_gtest(remote_handle64 h, int M, int K, int N, int nw, int64_t *us, float *err, int *rcout) {
  *us = 0; *err = -1; *rcout = 0;
  const int kb = K / 32, nb = N / 32;
  compute_res_attr_t attr; HAP_compute_res_attr_init(&attr);
  HAP_compute_res_attr_set_vtcm_param(&attr, 8u << 20, 1); HAP_compute_res_attr_set_hmx_param(&attr, 1);
  uint32_t ctx = HAP_compute_res_acquire(&attr, 100000); if (!ctx) { *rcout = -1; return 0; }
  uint8_t *vt = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&attr);
  hmx_scales_init(vt + (7u << 20) + 131072);   /* gemm_run uses at most 7 MB */
  __fp16 *A = (__fp16 *)memalign(128, (size_t)M * K * 2), *W = (__fp16 *)memalign(128, (size_t)K * N * 2), *O = (__fp16 *)memalign(128, (size_t)M * N * 2);
  if (!A || !W || !O) { *rcout = -4; HAP_compute_res_release(ctx); return 0; }
  for (int m = 0; m < M; m++) for (int k = 0; k < K; k++) A[((size_t)(m >> 5) * kb + (k >> 5)) * 1024 + tin(m, k)] = (__fp16)gen(m, k, 7);
  for (int k = 0; k < K; k++) for (int n = 0; n < N; n++) W[((size_t)(n >> 5) * kb + (k >> 5)) * 1024 + tin(k, n)] = (__fp16)gen(k, n, 11);
  HAP_power_request_t rq; memset(&rq, 0, sizeof rq); rq.type = HAP_power_set_HMX; rq.hmx.power_up = TRUE; HAP_power_set((void *)h, &rq); power_max((void *)h);
  if (HAP_compute_res_hmx_lock(ctx)) { *rcout = -2; HAP_compute_res_release(ctx); return 0; }
  pool_init(nw);
  gemm_run((uint16_t *)A, M, K, (uint16_t *)W, N, (uint16_t *)O, vt);  /* warm */
  uint64_t t0 = HAP_perf_get_time_us();
  gemm_run((uint16_t *)A, M, K, (uint16_t *)W, N, (uint16_t *)O, vt);
  *us = HAP_perf_get_time_us() - t0;
  float me = 0;
  for (int s = 0; s < 150; s++) {
    int m = (s * 37) % M, n = (s * 91 + 5) % N; double a = 0;
    for (int k = 0; k < K; k++) a += (double)(float)(__fp16)gen(m, k, 7) * (float)(__fp16)gen(k, n, 11);
    float g = (float)O[((size_t)(m >> 5) * nb + (n >> 5)) * 1024 + tin(m, n)];
    float e = fabsf(g - (float)a) / (fabsf((float)a) + 1.f); if (e > me) me = e;
  }
  *err = me; *rcout = 1000 + power_hmx_rc();
  pool_quit();
  free(A); free(W); free(O); HAP_compute_res_hmx_unlock(ctx); HAP_compute_res_release(ctx);
  return 0;
}
