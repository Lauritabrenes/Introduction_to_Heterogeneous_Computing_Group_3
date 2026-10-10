#pragma once
#include <cstdint>

// ============================================================================
// Funciones envolventes (wrappers) para lanzar los kernels de CUDA
// ============================================================================

// Lanza el kernel que calcula la física de la onda (ecuación de onda)
void launch_simulate_step(const float* d_previous, 
                          const float* d_current, 
                          float* d_next, 
                          int width, 
                          int height, 
                          float wave_speed, 
                          float damping, 
                          float edge_damping);

// Lanza el kernel que calcula la iluminación y el color de cada píxel
void launch_render_frame(const float* d_height, 
                         uint8_t* d_image_out, 
                         int width, 
                         int height);
