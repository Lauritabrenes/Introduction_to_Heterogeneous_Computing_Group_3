// softmax_cuda.cu - Laboratorio 6, Ejercicio B (GPU CUDA, Jetson Nano)
// Build: nvcc -O3 -std=c++14 -arch=sm_53 softmax_cuda.cu -o softmax_cuda
// Usage: ./softmax_cuda --precision fp16|fp32 --n N --reps R --measure kernel|end-to-end
//        [--runs 5] [--seed 42] [--csv results.csv]
//
// Design: 1 block x 1024 threads (N <= 1000 fits in one block).
//   Global memory: x (input), y (output). Shared memory: s[1024] float partials.
//   Stage 1: max reduction (tree in shared mem)
//   Stage 2: e_i = exp(x_i - m), sum reduction (tree in shared mem)
//   Stage 3: y_i = e_i / sum
// FP16: data stored/transferred as __half; exp and accumulation done in FP32
//       to avoid precision loss in the reduction (documented decision).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <string>

#define CK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)

static const int THREADS = 1024;

template<typename T> __device__ __forceinline__ float toF(T v);
template<> __device__ __forceinline__ float toF<float>(float v) { return v; }
template<> __device__ __forceinline__ float toF<__half>(__half v) { return __half2float(v); }
template<typename T> __device__ __forceinline__ T fromF(float v);
template<> __device__ __forceinline__ float fromF<float>(float v) { return v; }
template<> __device__ __forceinline__ __half fromF<__half>(float v) { return __float2half(v); }

template<typename T>
__global__ void softmax_kernel(const T* __restrict__ x, T* __restrict__ y, int n) {
    __shared__ float s[THREADS];
    int t = threadIdx.x;

    // Stage 1: max
    float m = -INFINITY;
    for (int i = t; i < n; i += blockDim.x) m = fmaxf(m, toF(x[i]));
    s[t] = m;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (t < k) s[t] = fmaxf(s[t], s[t + k]);
        __syncthreads();
    }
    m = s[0];
    __syncthreads();

    // Stage 2: exp + sum
    float acc = 0.f;
    for (int i = t; i < n; i += blockDim.x) {
        float e = expf(toF(x[i]) - m);
        y[i] = fromF<T>(e);
        acc += e;
    }
    s[t] = acc;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (t < k) s[t] += s[t + k];
        __syncthreads();
    }
    float inv = 1.f / s[0];

    // Stage 3: normalization
    for (int i = t; i < n; i += blockDim.x) y[i] = fromF<T>(toF(y[i]) * inv);
}

// ---------------- host helpers ----------------
static float h2f(__half h) { return __half2float(h); }
static __half f2h(float f) { return __float2half(f); }

static void softmax_ref(const std::vector<double>& x, std::vector<double>& y) {
    double m = *std::max_element(x.begin(), x.end()), s = 0;
    y.resize(x.size());
    for (size_t i = 0; i < x.size(); ++i) { y[i] = std::exp(x[i] - m); s += y[i]; }
    for (auto& v : y) v /= s;
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

template<typename T>
static void run(const std::vector<float>& xf, int n, int reps, int runs, bool e2e,
                const char* prec, const char* csv) {
    std::vector<T> hx(n), hy(n);
    for (int i = 0; i < n; ++i) {
        if (sizeof(T) == 2) ((__half*)hx.data())[i] = f2h(xf[i]);
        else ((float*)hx.data())[i] = xf[i];
    }
    T *dx, *dy;
    CK(cudaMalloc(&dx, n * sizeof(T)));
    CK(cudaMalloc(&dy, n * sizeof(T)));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));

    // warm-up
    CK(cudaMemcpy(dx, hx.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    softmax_kernel<T><<<1, THREADS>>>(dx, dy, n);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<double> times_us;
    for (int r = 0; r < runs; ++r) {
        CK(cudaEventRecord(a));
        for (int k = 0; k < reps; ++k) {
            if (e2e) CK(cudaMemcpy(dx, hx.data(), n * sizeof(T), cudaMemcpyHostToDevice));
            softmax_kernel<T><<<1, THREADS>>>(dx, dy, n);
            if (e2e) {
                CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(hy.data(), dy, n * sizeof(T), cudaMemcpyDeviceToHost));
            }
        }
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        times_us.push_back(ms * 1000.0 / reps);
    }
    CK(cudaMemcpy(hy.data(), dy, n * sizeof(T), cudaMemcpyDeviceToHost));

    // verification vs FP64
    std::vector<double> xd(n), yref;
    for (int i = 0; i < n; ++i)
        xd[i] = (sizeof(T) == 2) ? (double)h2f(((__half*)hx.data())[i]) : (double)xf[i];
    softmax_ref(xd, yref);
    double sum = 0, maxabs = 0, maxrel = 0;
    bool inrange = true;
    int am = 0, amref = 0;
    std::vector<double> yo(n);
    for (int i = 0; i < n; ++i) {
        yo[i] = (sizeof(T) == 2) ? (double)h2f(((__half*)hy.data())[i]) : (double)((float*)hy.data())[i];
        sum += yo[i];
        if (yo[i] < 0 || yo[i] > 1) inrange = false;
        double ae = std::fabs(yo[i] - yref[i]);
        maxabs = std::max(maxabs, ae);
        if (yref[i] > 0) maxrel = std::max(maxrel, ae / yref[i]);
        if (yo[i] > yo[am]) am = i;
        if (yref[i] > yref[amref]) amref = i;
    }
    double med = median(times_us);
    double mn = *std::min_element(times_us.begin(), times_us.end());
    double mx = *std::max_element(times_us.begin(), times_us.end());

    printf("precision=%s n=%d reps=%d measure=%s\n", prec, n, reps, e2e ? "end-to-end" : "kernel");
    printf("time_us median=%.3f min=%.3f max=%.3f\n", med, mn, mx);
    printf("sum=%.6f in_range=%s argmax_ok=%s max_abs_err=%.3e max_rel_err=%.3e\n",
           sum, inrange ? "yes" : "no", am == amref ? "yes" : "no", maxabs, maxrel);

    if (csv) {
        FILE* f = fopen(csv, "a");
        if (f) {
            for (int r = 0; r < runs; ++r)
                fprintf(f, "GPU,%s,%d,%d,%s,%d,%.4f,%.6e,%.6e\n", prec, n, reps,
                        e2e ? "end-to-end" : "kernel", r, times_us[r], maxabs, maxrel);
            fclose(f);
        }
    }
    cudaFree(dx); cudaFree(dy);
    cudaEventDestroy(a); cudaEventDestroy(b);
}

int main(int argc, char** argv) {
    std::string prec = "fp32", meas = "kernel";
    int n = 100, reps = 10000, runs = 5;
    unsigned seed = 42;
    const char* csv = "results_gpu.csv";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--precision" && i + 1 < argc) prec = argv[++i];
        else if (a == "--n" && i + 1 < argc) n = atoi(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = atoi(argv[++i]);
        else if (a == "--measure" && i + 1 < argc) meas = argv[++i];
        else if (a == "--runs" && i + 1 < argc) runs = atoi(argv[++i]);
        else if (a == "--seed" && i + 1 < argc) seed = (unsigned)atoi(argv[++i]);
        else if (a == "--csv" && i + 1 < argc) csv = argv[++i];
    }
    if (n < 1 || n > 4096) { fprintf(stderr, "n must be in [1,4096]\n"); return 1; }
    if (runs < 5) runs = 5;

    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-4.f, 4.f);
    std::vector<float> x(n);
    for (auto& v : x) v = dist(gen);

    bool e2e = (meas == "end-to-end");
    if (prec == "fp16") run<__half>(x, n, reps, runs, e2e, "fp16", csv);
    else run<float>(x, n, reps, runs, e2e, "fp32", csv);
    return 0;
}
