#include "drop_cuda.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

// ============================================================================
// Funciones Helper (__device__ corre en la GPU y solo es llamada por la GPU)
// ============================================================================

__device__ float border_absorption_cuda(int x, int y, int width, int height, float damping, float edge_damping) {
    const int band = 32;
    // Encontramos la distancia mínima a cualquiera de los 4 bordes
    int dist_x = x < (width - 1 - x) ? x : (width - 1 - x);
    int dist_y = y < (height - 1 - y) ? y : (height - 1 - y);
    int dist = dist_x < dist_y ? dist_x : dist_y;

    if (dist >= band) {
        return damping; // Interior: solo damping base
    }

    // Rampa cuadrática para los bordes
    float t = 1.0f - (float)dist / (float)band;
    return damping + edge_damping * t * t;
}

// ============================================================================
// KERNEL 1: Simulación de la física (Ecuación de onda)
// ============================================================================
__global__ void simulate_step_kernel(const float* previous, const float* current, float* next, 
                                     int width, int height, float c2, float damping, float edge_damping) {
    
    // 1. Identificador global del hilo en 2D (Basado en la teoría SIMT del curso)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 2. Manejo de bordes: Solo procesamos celdas internas, los bordes (x=0, y=0, etc.) se mantienen en 0
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int idx = y * width + x;

        // Laplaciano (stencil de 5 puntos)
        float laplacian = current[idx - 1] + current[idx + 1] + 
                          current[idx - width] + current[idx + width] - 
                          4.0f * current[idx];

        // Velocidad (diferencias finitas)
        float velocity = current[idx] - previous[idx];

        // Absorción en los bordes
        float local_damping = border_absorption_cuda(x, y, width, height, damping, edge_damping);

        // Actualización de la ecuación de onda
        next[idx] = 2.0f * current[idx] - previous[idx] + c2 * laplacian - local_damping * velocity;
    }
}

// ============================================================================
// KERNEL 2: Renderizado del cuadro a píxeles
// ============================================================================
__global__ void render_frame_kernel(const float* height_map, uint8_t* image_out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Validamos que el hilo esté dentro de los límites de la imagen
    if (x >= 0 && x < width && y >= 0 && y < height) {
        
        // Clamping manual para los vecinos en los bordes
        int xm = (x - 1 > 0) ? x - 1 : 0;
        int xp = (x + 1 < width - 1) ? x + 1 : width - 1;
        int ym = (y - 1 > 0) ? y - 1 : 0;
        int yp = (y + 1 < height - 1) ? y + 1 : height - 1;

        // Gradiente usando diferencias centrales
        float dx = height_map[y * width + xm] - height_map[y * width + xp];
        float dy = height_map[ym * width + x] - height_map[yp * width + x];

        // Normalización del vector normal
        float nx = 2.8f * dx;
        float ny = 2.8f * dy;
        float nz = 1.0f;
        float inv_len = rsqrtf(nx * nx + ny * ny + nz * nz); // rsqrtf es una operación matemática rápida en CUDA
        nx *= inv_len;
        ny *= inv_len;
        nz *= inv_len;

        // Vector de luz pre-normalizado (-0.35, -0.55, 0.76)
        float lx = -0.3495f;
        float ly = -0.5493f;
        float lz = 0.7589f;

        // Iluminación difusa y onda
        float dot_product = nx * lx + ny * ly + nz * lz;
        float diffuse = fmaxf(0.0f, dot_product);
        
        float wave = 0.5f + 1.8f * height_map[y * width + x];
        wave = fminf(fmaxf(wave, 0.0f), 1.0f);

        // Brillo especular
        float specular = powf(fmaxf(0.0f, diffuse), 24.0f);

        // Combinación de intensidades
        float intensity = 35.0f + 120.0f * wave;
        intensity *= 0.60f + 0.65f * diffuse;
        intensity += 130.0f * specular;

        // Clamp final al rango de 8-bits [0, 255]
        uint8_t gray = (uint8_t)fminf(fmaxf(intensity, 0.0f), 255.0f);

        // Escribimos en formato BGR (3 canales por píxel)
        int out_idx = (y * width + x) * 3;
        image_out[out_idx]     = gray; // B
        image_out[out_idx + 1] = gray; // G
        image_out[out_idx + 2] = gray; // R
    }
}

// ============================================================================
// Funciones de lanzamiento (Wrappers)
// ============================================================================

void launch_simulate_step(const float* d_previous, const float* d_current, float* d_next, 
                          int width, int height, float wave_speed, float damping, float edge_damping) {
    
    float c2 = wave_speed * wave_speed;

    // Configuración de bloques: 16x16 hilos por bloque (256 hilos total, un estándar en CUDA)
    dim3 block(16, 16);
    // Configuración del Grid: Calculamos cuántos bloques necesitamos para cubrir todo el ancho y alto
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Lanzamiento del kernel
    simulate_step_kernel<<<grid, block>>>(d_previous, d_current, d_next, width, height, c2, damping, edge_damping);
}

void launch_render_frame(const float* d_height, uint8_t* d_image_out, int width, int height) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    render_frame_kernel<<<grid, block>>>(d_height, d_image_out, width, height);
}
