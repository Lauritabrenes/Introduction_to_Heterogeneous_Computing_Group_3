// softmax_tb.cpp - testbench for C simulation and C/RTL cosimulation
// Same input generator and seed as CPU/GPU (mt19937, uniform [-4,4], seed 42, reseeded per N).
// Returns 0 if all checks pass (required for cosim to report PASS).
#include "softmax_hls.h"
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>

int main() {
    const int Ns[4] = {10, 100, 500, 1000};
#ifdef DATA_FP16
    const double TOL_ABS = 2e-3, TOL_SUM = 1e-2; const char* prec = "fp16";
#else
    const double TOL_ABS = 1e-5, TOL_SUM = 1e-4; const char* prec = "fp32";
#endif
    int fails = 0;
    FILE* f = fopen("tb_results.csv", "w");
    if (f) fprintf(f, "precision,n,sum,max_abs_err,max_rel_err,in_range,argmax_ok\n");

    for (int t = 0; t < 4; t++) {
        int n = Ns[t];
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-4.f, 4.f);
        static data_t x[MAXN], y[MAXN];
        for (int i = 0; i < MAXN; i++) { x[i] = (data_t)0.f; y[i] = (data_t)0.f; }
        for (int i = 0; i < n; i++) x[i] = (data_t)dist(gen);

        softmax_hls(x, y, n);

        // FP64 reference (std::exp is fine here: testbench is not synthesized)
        std::vector<double> xr(n), yr(n);
        for (int i = 0; i < n; i++) xr[i] = (double)(float)x[i];
        double m = *std::max_element(xr.begin(), xr.end()), s = 0;
        for (int i = 0; i < n; i++) { yr[i] = std::exp(xr[i] - m); s += yr[i]; }
        for (int i = 0; i < n; i++) yr[i] /= s;

        double sum = 0, maxabs = 0, maxrel = 0;
        bool inrange = true;
        int am = 0, amr = 0;
        for (int i = 0; i < n; i++) {
            double yo = (double)(float)y[i];
            sum += yo;
            if (yo < 0.0 || yo > 1.0) inrange = false;
            double ae = std::fabs(yo - yr[i]);
            maxabs = std::max(maxabs, ae);
            if (yr[i] > 0) maxrel = std::max(maxrel, ae / yr[i]);
            if (yo > (double)(float)y[am]) am = i;
            if (yr[i] > yr[amr]) amr = i;
        }
        bool argok = (am == amr) || ((float)y[am] == (float)y[amr]);   // tie after rounding is OK
        bool ok = inrange && argok && std::fabs(sum - 1.0) < TOL_SUM && maxabs < TOL_ABS;
        if (!ok) fails++;

        printf("[%s] N=%4d sum=%.6f max_abs=%.3e max_rel=%.3e range=%s argmax=%s -> %s\n",
               prec, n, sum, maxabs, maxrel, inrange ? "ok" : "BAD", argok ? "ok" : "BAD",
               ok ? "PASS" : "FAIL");
        if (f) fprintf(f, "%s,%d,%.8f,%.6e,%.6e,%d,%d\n", prec, n, sum, maxabs, maxrel, inrange, argok);
    }
    if (f) fclose(f);
    printf(fails ? "TEST FAILED (%d)\n" : "TEST PASSED\n", fails);
    return fails ? 1 : 0;
}
