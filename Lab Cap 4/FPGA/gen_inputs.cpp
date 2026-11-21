// gen_inputs.cpp - writes inputs_N.bin (float32) with the SAME generator as CPU/GPU/testbench
// Build & run on the KV260:  g++ -O2 gen_inputs.cpp -o gen_inputs && ./gen_inputs
#include <cstdio>
#include <random>
int main() {
    const int Ns[4] = {10, 100, 500, 1000};
    for (int n : Ns) {
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-4.f, 4.f);
        char name[64];
        snprintf(name, sizeof(name), "inputs_%d.bin", n);
        FILE* f = fopen(name, "wb");
        for (int i = 0; i < n; i++) { float v = dist(gen); fwrite(&v, sizeof(float), 1, f); }
        fclose(f);
        printf("wrote %s\n", name);
    }
    return 0;
}
