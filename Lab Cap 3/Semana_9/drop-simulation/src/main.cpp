#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

// Incluimos las cabeceras de CUDA y nuestros wrappers
#include <cuda_runtime.h>
#include "drop_cuda.hpp"

namespace {

struct Config {
    int width = 640;
    int height = 640;
    double seconds = 30.0;
    int fps = 30;
    int steps_per_frame = 1;
    float wave_speed = 0.45f;
    float damping = 0.006f;
    float edge_damping = 0.035f;
    float drop_radius = 18.0f;
    float drop_strength = 1.0f;
    std::string output = "output/drop_simulation_gpu_base.mp4";
};

int index_of(int x, int y, int width) {
    return y * width + x;
}

void add_drop(std::vector<float>& current, std::vector<float>& previous, const Config& cfg) {
    const float cx = 0.5f * static_cast<float>(cfg.width - 1);
    const float cy = 0.5f * static_cast<float>(cfg.height - 1);
    const float sigma2 = cfg.drop_radius * cfg.drop_radius;

    for (int y = 1; y < cfg.height - 1; ++y) {
        for (int x = 1; x < cfg.width - 1; ++x) {
            const float dx = static_cast<float>(x) - cx;
            const float dy = static_cast<float>(y) - cy;
            const float r2 = dx * dx + dy * dy;
            const float pulse = cfg.drop_strength * std::exp(-r2 / (2.0f * sigma2));
            const int idx = index_of(x, y, cfg.width);
            current[idx] += pulse;
            previous[idx] -= 0.35f * pulse;
        }
    }
}

// FUNCIONES ORIGINALES EN CPU (Se conservan para validación)
float border_absorption(int x, int y, const Config& cfg) {
    constexpr int band = 32;
    const int dist = std::min({x, y, cfg.width - 1 - x, cfg.height - 1 - y});
    if (dist >= band) {
        return cfg.damping;
    }

    const float t = 1.0f - static_cast<float>(dist) / static_cast<float>(band);
    return cfg.damping + cfg.edge_damping * t * t;
}

void simulate_step(const std::vector<float>& previous,
                   const std::vector<float>& current,
                   std::vector<float>& next,
                   const Config& cfg) {
    const float c2 = cfg.wave_speed * cfg.wave_speed;

    std::fill(next.begin(), next.end(), 0.0f);

    for (int y = 1; y < cfg.height - 1; ++y) {
        for (int x = 1; x < cfg.width - 1; ++x) {
            const int idx = index_of(x, y, cfg.width);
            const float laplacian =
                current[idx - 1] + current[idx + 1] +
                current[idx - cfg.width] + current[idx + cfg.width] -
                4.0f * current[idx];
            const float velocity = current[idx] - previous[idx];
            const float local_damping = border_absorption(x, y, cfg);

            next[idx] = 2.0f * current[idx] - previous[idx] +
                        c2 * laplacian -
                        local_damping * velocity;
        }
    }
}

cv::Mat render_frame(const std::vector<float>& height, const Config& cfg, int frame_number) {
    cv::Mat image(cfg.height, cfg.width, CV_8UC3);

    const cv::Vec3f light_dir = cv::normalize(cv::Vec3f(-0.35f, -0.55f, 0.76f));

    for (int y = 0; y < cfg.height; ++y) {
        for (int x = 0; x < cfg.width; ++x) {
            const int xm = std::max(0, x - 1);
            const int xp = std::min(cfg.width - 1, x + 1);
            const int ym = std::max(0, y - 1);
            const int yp = std::min(cfg.height - 1, y + 1);

            const float dx = height[index_of(xm, y, cfg.width)] - height[index_of(xp, y, cfg.width)];
            const float dy = height[index_of(x, ym, cfg.width)] - height[index_of(x, yp, cfg.width)];
            const cv::Vec3f normal = cv::normalize(cv::Vec3f(2.8f * dx, 2.8f * dy, 1.0f));

            const float diffuse = std::max(0.0f, normal.dot(light_dir));
            
            const float raw_wave = 0.5f + 1.8f * height[index_of(x, y, cfg.width)];
            const float wave = std::max(0.0f, std::min(raw_wave, 1.0f));
            
            const float specular = std::pow(std::max(0.0f, diffuse), 24.0f);

            float intensity = 35.0f + 120.0f * wave;
            intensity *= 0.60f + 0.65f * diffuse;
            intensity += 130.0f * specular;

            const auto gray = static_cast<unsigned char>(std::max(0.0f, std::min(intensity, 255.0f)));
            image.at<cv::Vec3b>(y, x) = cv::Vec3b(gray, gray, gray);
        }
    }

    cv::putText(image,
                "GPU CUDA Baseline | frame " + std::to_string(frame_number),
                cv::Point(18, 32),
                cv::FONT_HERSHEY_SIMPLEX,
                0.65,
                cv::Scalar(235, 235, 235),
                1,
                cv::LINE_AA);

    return image;
}

}  // namespace

// ============================================================================
// main: VERSIÓN GPU CON VALIDACIÓN NUMÉRICA (EJERCICIO D y E)
// ============================================================================
int main() {
    try {
        const Config cfg;
        const int total_frames = static_cast<int>(std::round(cfg.seconds * cfg.fps));
        const std::size_t cells = static_cast<std::size_t>(cfg.width) * static_cast<std::size_t>(cfg.height);

        // Crear carpeta de salida compatible con Jetson Nano
        mkdir("output", 0777);

        // 1. Memoria en el Host (CPU) para la perturbación inicial
        std::vector<float> h_previous(cells, 0.0f);
        std::vector<float> h_current(cells, 0.0f);
        add_drop(h_current, h_previous, cfg);

        // === EJERCICIO E: Variables para validación ===
        std::vector<float> cpu_previous = h_previous;
        std::vector<float> cpu_current = h_current;
        std::vector<float> cpu_next(cells, 0.0f);
        std::vector<float> h_gpu_current(cells, 0.0f); // Para descargar la GPU y comparar
        float max_absolute_error = 0.0f;

        // 2. Reserva de memoria en el Device (GPU)
        float *d_previous, *d_current, *d_next;
        uint8_t *d_frame_data;
        
        size_t grid_bytes = cells * sizeof(float);
        size_t img_bytes = cells * 3 * sizeof(uint8_t); // 3 canales (BGR)

        cudaMalloc(&d_previous, grid_bytes);
        cudaMalloc(&d_current, grid_bytes);
        cudaMalloc(&d_next, grid_bytes);
        cudaMalloc(&d_frame_data, img_bytes);

        // 3. Copia inicial Host-to-Device (H2D)
        cudaMemcpy(d_previous, h_previous.data(), grid_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_current, h_current.data(), grid_bytes, cudaMemcpyHostToDevice);
        cudaMemset(d_next, 0, grid_bytes);

        cv::VideoWriter writer(
            cfg.output,
            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            static_cast<double>(cfg.fps),
            cv::Size(cfg.width, cfg.height));

        if (!writer.isOpened()) {
            throw std::runtime_error("No se pudo abrir el archivo de salida: " + cfg.output);
        }

        cv::Mat frame_img(cfg.height, cfg.width, CV_8UC3);

        double time_simulate = 0.0;
        double time_render = 0.0;
        double time_d2h_copy = 0.0;

        const auto start = std::chrono::steady_clock::now();

        // === Ciclo principal de simulación ===
        for (int frame = 0; frame < total_frames; ++frame) {
            
            for (int step = 0; step < cfg.steps_per_frame; ++step) {
                auto t0 = std::chrono::steady_clock::now();
                
                // --- Lanzar Kernel de Simulación en GPU ---
                launch_simulate_step(d_previous, d_current, d_next, cfg.width, cfg.height, 
                                     cfg.wave_speed, cfg.damping, cfg.edge_damping);
                
                cudaDeviceSynchronize(); 
                auto t1 = std::chrono::steady_clock::now();
                time_simulate += std::chrono::duration<double>(t1 - t0).count();

                std::swap(d_previous, d_current);
                std::swap(d_current, d_next);

                // --- Cálculo simultáneo en CPU para comparar ---
                simulate_step(cpu_previous, cpu_current, cpu_next, cfg);
                std::swap(cpu_previous, cpu_current);
                std::swap(cpu_current, cpu_next);
            }

            // === EJERCICIO E: Comparación Numérica ===
            cudaMemcpy(h_gpu_current.data(), d_current, grid_bytes, cudaMemcpyDeviceToHost);
            for (size_t i = 0; i < cells; ++i) {
                float error = std::abs(cpu_current[i] - h_gpu_current[i]);
                if (error > max_absolute_error) {
                    max_absolute_error = error;
                }
            }

            auto t2 = std::chrono::steady_clock::now();
            // Lanzar Kernel de Renderizado
            launch_render_frame(d_current, d_frame_data, cfg.width, cfg.height);
            cudaDeviceSynchronize();
            auto t3 = std::chrono::steady_clock::now();
            time_render += std::chrono::duration<double>(t3 - t2).count();

            auto t4 = std::chrono::steady_clock::now();
            // Copiar la imagen generada de GPU a CPU (D2H)
            cudaMemcpy(frame_img.data, d_frame_data, img_bytes, cudaMemcpyDeviceToHost);
            auto t5 = std::chrono::steady_clock::now();
            time_d2h_copy += std::chrono::duration<double>(t5 - t4).count();

            writer.write(frame_img);

            if (frame % std::max(1, total_frames / 10) == 0) {
                std::cout << "Frame " << frame << " / " << total_frames << '\n';
            }
        }

        const auto end = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(end - start).count();
        const double simulated_steps = static_cast<double>(total_frames) * cfg.steps_per_frame;

        std::cout << "\n========================================\n";
        std::cout << "   RESULTADOS BASE GPU (EJERCICIO D y E)\n";
        std::cout << "========================================\n";
        std::cout << "Video generado: " << cfg.output << '\n';
        std::cout << "Tiempo Total: " << elapsed << " s\n";
        std::cout << "Tiempo simulate_step_kernel: " << time_simulate << " s\n";
        std::cout << "Tiempo render_frame_kernel: " << time_render << " s\n";
        std::cout << "Tiempo transferencias D2H: " << time_d2h_copy << " s\n";
        std::cout << "Rendimiento GPU: " << simulated_steps / elapsed << " pasos/s\n";
        std::cout << "Error Absoluto Máximo (CPU vs GPU): " << max_absolute_error << '\n';
        std::cout << "========================================\n";

        // Liberación de memoria en GPU
        cudaFree(d_previous);
        cudaFree(d_current);
        cudaFree(d_next);
        cudaFree(d_frame_data);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
