#pragma once
#include <stdint.h>
#include <stddef.h>
#include <hexagon_types.h>
/* O[M x N] = A[M x K] * W[K x N]; all fp16 crouton. A,O: [row slab][col block]; W: [n block][k block]. A, W, O in DDR. vtcm: >= 6 MB scratch. */
void hmx_acc(uint8_t *, uint32_t, uint8_t *, uint32_t);
void hmx_store(uint8_t *, uint32_t);
/* Hexagon v73+ (V81 = Snapdragon 8 Elite Gen 5): the legacy store "mxmem(..):after.hf = acc" writes nothing (zeros), the accumulator is read
 * through "cvt.hf = acc" + "mxmem(..) = cvt", which applies a per-column scale/bias block (bias = mxmem2): 256 bytes in VTCM, 32 words with fp16 1.0
 * in the low half, then 128 bytes of zero bias. A zeroed block means scale 0, i.e. all-zero output. Call after every VTCM acquire (no-op on v69). */
void hmx_scales_init(uint8_t *vtcm256);
int hmx_cvt(void);   /* 1 when the v73+ store is in use (decided from the DSP arch at run time in a v69 build) */
typedef struct {   /* optional epilogue run by the pool workers on the finished output tiles (VTCM) */
  int mode;                      /* 0 none (copy to O), 1 residual: x32 += gate*tile (O unused), 2 GELU in place then copy, 3 copy + rope */
  float *x32; const float *gate; /* mode 1: x32 tiles [slab][64], gate[2048] */
  const uint16_t *t2; HVX_Vector *scr;  /* mode 2: table + 8 scratch vectors per worker */
  const uint16_t *prev;                 /* mode 4: residual with a K-split partial: x += gate*(tile + prev tile) */
  const float *gain, *cosT, *sinT; int rope;   /* mode 3: per-head RMSNorm(+RoPE) on q/k once both panels of a head are written (O = DDR output) */
} gemm_epi_t;
typedef struct { int lda, ldw, kofs; } gemm_geo_t;   /* tiles per A slab row, tiles per W n-block, first k tile of this (partial) K range */
void gemm_run_geo(const uint16_t *A, int M, int K, const uint16_t *W, int N, uint16_t *O, uint8_t *vtcm, const gemm_epi_t *epi, const gemm_geo_t *geo);
void gemm_run_ex(const uint16_t *A, int M, int K, const uint16_t *W, int N, uint16_t *O, uint8_t *vtcm, const gemm_epi_t *epi);
void gemm_run(const uint16_t *A, int M, int K, const uint16_t *W, int N, uint16_t *O, uint8_t *vtcm);
void gemm_set_q8(int on);   /* weights W are q8 tiles (128 B scales + 1024 B int8, 1152 B per tile) dequantised to fp16 on the way into VTCM */
