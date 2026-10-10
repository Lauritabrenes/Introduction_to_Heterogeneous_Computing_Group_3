# Resumen de Avances y Resultados: Simulación GPU (Ejercicios D y E)

Este documento contiene el contexto de los cambios realizados en el entorno y el código, así como los resultados oficiales de la línea base en GPU para documentar en el repositorio.

---

## Parte 1: Contexto y Resumen Técnico (Para continuar con el Ejercicio F)

### Configuración del Entorno y Conexión (Jetson Nano)
*   **Acceso SSH:** Para descargar archivos mediante el servidor puente de la universidad, el comando definitivo que se debe usar en la terminal local es: `scp -r -o "ProxyJump=USUARIO_PUENTE@IP_PUENTE:PUERTO" USUARIO_JETSON@IP_JETSON:/ruta/remota/ ~/Desktop/`
*   **Bypass a Meson:** La versión de Meson instalada en la Jetson Ubuntu 18.04 (0.45.1) no soporta la compilación nativa de CUDA. Para evitar perder tiempo actualizando el sistema, decidimos **ignorar Meson** y compilar directamente usando `nvcc`. 
*   **Comando de compilación oficial:** De ahora en adelante, para compilar el proyecto se debe usar este comando en la raíz del proyecto:
    ```bash
    mkdir -p build
    nvcc -std=c++14 -O3 src/main.cpp src/drop_cuda.cu -o build/drop_simulation `pkg-config --cflags --libs opencv4`
    ```

### Archivos Tocados y Modificaciones (`src/main.cpp`)
El trabajo se concentró en adaptar `src/main.cpp` al compilador GCC 7.5 (C++14 estable):
*   Se eliminaron dependencias de C++17: Cambiamos `<filesystem>` por `<sys/stat.h>` (para la carpeta `output/`) y `std::clamp` por combinaciones de `std::max`/`std::min`.
*   Se programó la **Línea Base GPU (Ejercicio D)** integrando `cudaMalloc`, `cudaMemcpy` (H2D y D2H) y los llamados a los kernels.
*   Se programó la **Validación Numérica (Ejercicio E)** haciendo que la CPU calcule a la par de la GPU para comparar los resultados de la matriz celda por celda al final de cada *step*.

### ¿Qué toca hacer para el Ejercicio F?
El `main.cpp` está listo y validado. El código actual sufre del *cuello de botella* esperado (accesos lentos a memoria global y latencia por transferencias). Para el **Ejercicio F**, se debe modificar exclusivamente `src/drop_cuda.cu` (los kernels) e implementar 3 técnicas de optimización (ej. Memoria Compartida, `--use_fast_math`, ajustar `blockDim`, etc.) para superar el tiempo original de la CPU.

---

## Parte 2: Resultados para el Repositorio (Ejercicios D y E)

### 1. Ejercicio D: Resultados de la Línea Base (GPU)
Medición del rendimiento de la primera versión en CUDA sin aplicar técnicas de optimización. Se identificaron altos costos de latencia debido a la transferencia de memoria Device-To-Host (D2H) en cada iteración y accesos no optimizados a la memoria global.

| Métrica | Resultado |
| :--- | :--- |
| **Tiempo Total** | 29.7413 s |
| **Tiempo `simulate_step` (Kernel)** | 3.21792 s |
| **Tiempo `render_frame` (Kernel)** | 6.39198 s |
| **Tiempo transferencias D2H** | 2.00285 s |
| **Rendimiento GPU** | **30.261 pasos/s** |
| **Archivo de salida** | `output/drop_simulation_gpu_base.mp4` |

> *Nota: El rendimiento base en GPU (~30 pasos/s) es menor al de la CPU (~93 pasos/s) debido a los accesos directos a memoria global y las constantes transferencias a través del bus PCI. Estos cuellos de botella serán abordados en el Ejercicio F.*

### 2. Ejercicio E: Validación de Resultados
Se implementó una validación en tiempo de ejecución ejecutando la simulación en CPU y GPU en paralelo, comparando la malla de alturas celda por celda tras cada paso.

*   **Validación Numérica:** 
    *   **Error Absoluto Máximo (CPU vs GPU): `0.0`**
    *   *Conclusión:* La lógica física y el cálculo de punto flotante en la GPU producen resultados 100% equivalentes a la implementación original en C++.
*   **Validación Visual:** 
    *   El archivo de video generado (`drop_simulation_gpu_base.mp4`) fue extraído de la Jetson Nano y se comprobó mediante inspección visual que la propagación de ondas y la iluminación se renderizan correctamente sin artefactos.
========================================
   RESULTADOS BASE GPU (EJERCICIO D y E)
========================================
Video generado: output/drop_simulation_gpu_base.mp4
Tiempo Total: 43.8217 s
Tiempo simulate_step_kernel: 3.59851 s
Tiempo render_frame_kernel: 7.68577 s
Tiempo transferencias D2H: 2.64887 s
Rendimiento GPU: 20.5378 pasos/s
Error Absoluto Máximo (CPU vs GPU): 0
========================================
