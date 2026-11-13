// softmax_hls.cpp - Laboratorio 6, Ejercicio C (Vitis HLS, KV260)
// Stable softmax: m = max(x), e_i = exp(x_i - m), y_i = e_i / sum(e)
// Uses hls::exp (NOT std::exp) in the synthesizable kernel.
// Requires 1 <= n <= MAXN.
#include "softmax_hls.h"

// Vitis HLS auto-pipelines loops by default, so the sequential version
// must turn pipelining OFF explicitly.
#if VARIANT == 0
#define LOOP_PIPE _Pragma("HLS PIPELINE off")
#else
#define LOOP_PIPE _Pragma("HLS PIPELINE II=1")
#endif
// Trip counts so the latency report is meaningful with a variable n
#define TC  _Pragma("HLS LOOP_TRIPCOUNT min=10 max=1000 avg=400")
#define TCU _Pragma("HLS LOOP_TRIPCOUNT min=2 max=125 avg=50")   // = MAXN/UF for UF=8

// ---------------- stage: load ----------------
static void load(const data_t* x, data_t a[MAXN], int n) {
L_load:
    for (int i = 0; i < n; i++) {
        LOOP_PIPE TC
        a[i] = x[i];
    }
}

// Dataflow needs a single consumer per buffer -> two copies
static void load2(const data_t* x, data_t a[MAXN], data_t b[MAXN], int n) {
L_load2:
    for (int i = 0; i < n; i++) {
        LOOP_PIPE TC
        data_t v = x[i];
        a[i] = v;
        b[i] = v;
    }
}

// ---------------- stage 1: max reduction ----------------
static data_t reduce_max(const data_t a[MAXN], int n) {
#if VARIANT >= 2
    data_t part[UF];
#pragma HLS ARRAY_PARTITION variable=part complete
    for (int l = 0; l < UF; l++) {
#pragma HLS UNROLL
        part[l] = a[0];
    }
L_max:
    for (int i = 0; i < n; i += UF) {
        LOOP_PIPE TCU
        for (int l = 0; l < UF; l++) {
#pragma HLS UNROLL
            if (i + l < n) {
                data_t v = a[i + l];
                if (v > part[l]) part[l] = v;
            }
        }
    }
    data_t m = part[0];
    for (int l = 1; l < UF; l++) {
#pragma HLS UNROLL
        if (part[l] > m) m = part[l];
    }
    return m;
#else
    data_t m = a[0];
L_max:
    for (int i = 1; i < n; i++) {
        LOOP_PIPE TC
        if (a[i] > m) m = a[i];
    }
    return m;
#endif
}

// ---------------- stage 2: exp(x - m) + sum reduction ----------------
static acc_t exp_sum(const data_t a[MAXN], data_t m, data_t e[MAXN], int n) {
#if VARIANT >= 2
    acc_t part[UF];
#pragma HLS ARRAY_PARTITION variable=part complete
    for (int l = 0; l < UF; l++) {
#pragma HLS UNROLL
        part[l] = 0;
    }
L_exp:
    for (int i = 0; i < n; i += UF) {
        LOOP_PIPE TCU
        for (int l = 0; l < UF; l++) {
#pragma HLS UNROLL
            if (i + l < n) {
                data_t v = hls::exp((data_t)(a[i + l] - m));
                e[i + l] = v;
                part[l] += (acc_t)v;
            }
        }
    }
    acc_t s = 0;
    for (int l = 0; l < UF; l++) {
#pragma HLS UNROLL
        s += part[l];
    }
    return s;
#else
    acc_t s = 0;
L_exp:
    for (int i = 0; i < n; i++) {
        LOOP_PIPE TC
        data_t v = hls::exp((data_t)(a[i] - m));
        e[i] = v;
        s += (acc_t)v;   // loop-carried add: II may be > 1 (see report)
    }
    return s;
#endif
}

// ---------------- stage 3: normalization ----------------
static void normalize(const data_t e[MAXN], acc_t s, data_t* y, int n) {
    acc_t inv = (acc_t)1 / s;
L_norm:
    for (int i = 0; i < n; i++) {
        LOOP_PIPE TC
        y[i] = (data_t)((acc_t)e[i] * inv);
    }
}

// ---------------- dataflow wrappers (scalars passed by streams) ----------------
static void max_df(const data_t a[MAXN], int n, hls::stream<data_t>& ms) {
    ms.write(reduce_max(a, n));
}
static void exp_df(const data_t a[MAXN], hls::stream<data_t>& ms, data_t e[MAXN],
                   hls::stream<acc_t>& ss, int n) {
    data_t m = ms.read();
    ss.write(exp_sum(a, m, e, n));
}
static void norm_df(const data_t e[MAXN], hls::stream<acc_t>& ss, data_t* y, int n) {
    normalize(e, ss.read(), y, n);
}

// ---------------- top function ----------------
void softmax_hls(const data_t x[MAXN], data_t y[MAXN], int n) {
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem0 depth=1000
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem1 depth=1000
#pragma HLS INTERFACE s_axilite port=n
#pragma HLS INTERFACE s_axilite port=return

#if VARIANT == 3
#pragma HLS DATAFLOW
    data_t xa[MAXN], xb[MAXN], e[MAXN];
#pragma HLS ARRAY_PARTITION variable=xa cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=xb cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=e  cyclic factor=8
    hls::stream<data_t> ms("max_stream");
    hls::stream<acc_t>  ss("sum_stream");
    load2(x, xa, xb, n);
    max_df(xa, n, ms);
    exp_df(xb, ms, e, ss, n);
    norm_df(e, ss, y, n);
#else
    // ARRAY_PARTITION cannot be applied to the m_axi argument x itself,
    // so it is applied to the local on-chip copy.
    data_t xl[MAXN], e[MAXN];
#if VARIANT >= 2
#pragma HLS ARRAY_PARTITION variable=xl cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=e  cyclic factor=8
#endif
    if (n < 1) return;
    if (n > MAXN) n = MAXN;
    load(x, xl, n);
    data_t m = reduce_max(xl, n);
    acc_t  s = exp_sum(xl, m, e, n);
    normalize(e, s, y, n);
#endif
}
