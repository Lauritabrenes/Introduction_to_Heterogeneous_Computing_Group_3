// softmax_hls.h - Laboratorio 6, Ejercicio C (Vitis HLS, KV260)
#ifndef SOFTMAX_HLS_H
#define SOFTMAX_HLS_H

#include <hls_half.h>
#include <hls_math.h>
#include <hls_stream.h>

// ---- Precision: compile with -DDATA_FP16 for FP16, default FP32 ----
#ifdef DATA_FP16
typedef half data_t;
#else
typedef float data_t;
#endif

// ---- Accumulator for the sum reduction ----
// Default: float (documented decision: avoids precision loss when adding
// up to 1000 terms in FP16). Compile with -DACC_HALF for pure-FP16 sum.
#ifdef ACC_HALF
typedef data_t acc_t;
#else
typedef float acc_t;
#endif

// ---- Optimization variant ----
// 0 = sequential (pipelining explicitly disabled)
// 1 = + PIPELINE II=1
// 2 = + ARRAY_PARTITION cyclic (UF lanes, partial reductions)
// 3 = + DATAFLOW
#ifndef VARIANT
#define VARIANT 3
#endif

#ifndef UF
#define UF 8          // partition / unroll factor (if changed, update TCU_MAX tripcount)
#endif

#define MAXN 1000

void softmax_hls(const data_t x[MAXN], data_t y[MAXN], int n);

#endif
