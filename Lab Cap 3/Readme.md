# Laboratorio 5 — Computación Heterogénea con GPU

**Proyecto:** `drop-simulation` — Simulación de gota sobre estanque con ecuación de onda 2D amortiguada, implementada en CPU y acelerada con CUDA sobre NVIDIA Jetson Nano.

---

## Estructura del repositorio

```
.
├── Laboratorio_5___Computacion_Heterogenea___GPU.pdf   # Enunciado original del laboratorio
├── Readme.md                                            # Este archivo
├── ReporteLab5.md                                       # ← REPORTE FINAL DEL LABORATORIO
│
├── Parts A-B-C/                                         # Entregables de las fases A, B y C
│   └── drop-simulation/
│       ├── A-Baseline metrcis/                          # Métricas del código base CPU
│       ├── B-Program explanation/                       # Explicación del código
│       ├── C-Manual-Perf profiling/                     # Perfilado con timers y perf
│       ├── codigo comentado/                            # main.cpp original comentado
│       ├── meson.build                                  # Build system (solo CPU)
│       ├── perf.data                                    # Datos crudos de perf
│       ├── perf.data.old                                # Datos crudos previos
│       ├── perf_results.txt                             # Salida de perf report
│       ├── README.md                                    # README específico de A-B-C
│       └── src/
│           ├── main (original).cpp                      # Código CPU original comentado
│           └── main.cpp                                 # Código CPU con timers de profiling
│
└── Parts D-E-F/                                         # Entregables de las fases D, E y F
    ├── Semana 9/
    │   ├── Capitulo3-GPU-optimization.pdf               # Material de lectura
    │   ├── Laboratorio_5___Computacion_Heterogenea___GPU.pdf
    │   └── Laura/
    │       └── Readme.md                                # Notas específicas
    │
    └── Semana_9/
        ├── Capitulo3-GPU-optimization.pdf
        ├── Laboratorio_5___Computacion_Heterogenea___GPU.pdf
        │
        └── drop-simulation/                             # Proyecto principal con CUDA
            ├── baseline metrcis/                        # Métricas base GPU
            ├── build/                                   # Artefactos de compilación
            ├── codigo comentado/                        # Código con comentarios
            ├── Ejercicio c/                             # Material del Ejercicio C
            ├── meson.buid                               # (typo, ver meson.build)
            ├── meson.build                              # Build system
            ├── output/                                  # Videos generados
            │   └── drop_simulation_gpu_base_ejercicio_E.mp4
            ├── perf.data                                # Datos crudos de perf
            ├── perf.data.old
            ├── perf_results.txt
            ├── program explanation                      # Explicación del programa
            ├── README.md                                # README específico del proyecto CUDA
            │
            ├── resultados/                              # Todos los resultados experimentales
            │   ├── base_prueba_1.txt                    # GPU base, prueba 1
            │   ├── base_prueba_2.txt                    # GPU base, prueba 2
            │   ├── base_prueba_3.txt                    # GPU base, prueba 3
            │   ├── cpu_base.txt                         # CPU baseline
            │   ├── cpu_prueba_1.txt                     # CPU, prueba 1
            │   ├── cpu_prueba_2.txt                     # CPU, prueba 2
            │   ├── cpu_prueba_3.txt                     # CPU, prueba 3
            │   ├── gpu_base.txt                         # GPU baseline consolidado
            │   ├── opt1_prueba_{1,2,3}.txt              # Opt1 (32×8), 3 pruebas
            │   ├── opt2_prueba_{1,2,3}.txt              # Opt2 (shared mem), 3 pruebas
            │   ├── opt3_prueba_{1,2,3}.txt              # Opt3 (shared+32×8), 3 pruebas
            │   ├── optimizacion_{1,2,3}.txt             # Resultados consolidados por opt
            │   └── perfilados/                          # Salidas de Nsight Systems
            │       ├── estadisticas_opt1.txt
            │       ├── estadisticas_opt2.txt
            │       └── estadisticas_opt3.txt
            │
            ├── Resultados Ejercicio D y E.md            # Documentación de contexto D y E
            ├── Screenshot From 2026-10-09 21-39-12.png  # Captura de pantalla
            │
            └── src/                                     # Código fuente CUDA
                ├── drop_cuda.cu                         # Kernels CUDA principales
                ├── drop_cuda.hpp                        # Declaraciones (wrappers)
                ├── drop_cuda_base.cu                    # Kernel base (16×16)
                ├── drop_cuda_opt1.cu                    # Opt1: bloques 32×8
                ├── drop_cuda_opt2.cu                    # Opt2: shared memory
                ├── drop_cuda_opt3.cu                    # Opt3: shared memory + 32×8
                ├── main (original).cpp                  # CPU original (referencia)
                └── main.cpp                             # Programa principal con GPU + validación
```

---

## Guía rápida: ¿Dónde está cada cosa?

### 📄 Documentación

| Documento | Ubicación | Contenido |
|---|---|---|
| **Reporte final del laboratorio** | `ReporteLab5.md` (raíz) | Reporte completo |
| **Enunciado original** | Raíz y `Semana_9/` | PDF del laboratorio |
| **Explicación del programa** | `Parts D-E-F/Semana_9/drop-simulation/program explanation/` | Análisis del código |
| **Contexto D y E** | `Parts D-E-F/Semana_9/drop-simulation/Resultados Ejercicio D y E.md` | Notas sobre el porteo |
| **Material de lectura** | `Parts D-E-F/Semana_9/Capitulo3-GPU-optimization.pdf` | Capítulo 3 del libro |

### 💻 Código fuente

| Archivo | Ubicación | Propósito |
|---|---|---|
| **CPU original comentado** | `Parts A-B-C/drop-simulation/src/main (original).cpp` | Código de referencia |
| **CPU con profiling** | `Parts A-B-C/drop-simulation/src/main.cpp` | Versión A-B-C con timers |
| **CUDA base** | `Parts D-E-F/Semana_9/drop-simulation/src/drop_cuda.cu` | Kernels CUDA principales |
| **CUDA optimizado 1** | `Parts D-E-F/Semana_9/drop-simulation/src/drop_cuda_opt1.cu` | Bloques 32×8 |
| **CUDA optimizado 2** | `Parts D-E-F/Semana_9/drop-simulation/src/drop_cuda_opt2.cu` | Shared memory |
| **CUDA optimizado 3** | `Parts D-E-F/Semana_9/drop-simulation/src/drop_cuda_opt3.cu` | Shared memory + 32×8 |
| **Main GPU** | `Parts D-E-F/Semana_9/drop-simulation/src/main.cpp` | Programa principal con validación |

### 📊 Resultados

| Tipo | Ubicación | Contenido |
|---|---|---|
| **Métricas CPU** | `Parts D-E-F/Semana_9/drop-simulation/resultados/cpu_*.txt` | 4 archivos (base + 3 pruebas) |
| **Métricas GPU base** | `Parts D-E-F/Semana_9/drop-simulation/resultados/base_prueba_*.txt` | 3 pruebas |
| **Métricas Opt1** | `.../resultados/opt1_prueba_*.txt` | 3 pruebas |
| **Métricas Opt2** | `.../resultados/opt2_prueba_*.txt` | 3 pruebas |
| **Métricas Opt3** | `.../resultados/opt3_prueba_*.txt` | 3 pruebas |
| **Perfilado Nsight** | `.../resultados/perfilados/estadisticas_opt{1,2,3}.txt` | Salidas de Nsight Systems |
| **Perfilado perf** | `Parts A-B-C/drop-simulation/perf_results.txt` | Salida de perf report |

### 🎥 Video generado

| Video | Ubicación |
|---|---|
| GPU base (Ejercicio E) | `Parts D-E-F/Semana_9/drop-simulation/output/drop_simulation_gpu_base_ejercicio_E.mp4` |

---

## Compilación y ejecución

### Versión CPU (A-B-C)

```bash
cd "Parts A-B-C/drop-simulation"
meson setup build
meson compile -C build
./build/drop_simulation
```

### Versión GPU (D-E-F)

```bash
cd "Parts D-E-F/Semana_9/drop-simulation"
mkdir -p build
nvcc -std=c++14 -O3 src/main.cpp src/drop_cuda.cu \
     -o build/drop_simulation `pkg-config --cflags --libs opencv4`
./build/drop_simulation
```

> **Nota:** Meson 0.45.1 (Ubuntu 18.04) no soporta CUDA nativo. Se usa `nvcc` directamente.

---

## Resultados principales

| Versión | Tiempo | Pasos/s | Speedup | Error máx |
|---|---|---|---|---|
| CPU base | 9.61 s | 93.6 | 1.00× | — |
| GPU base (16×16) | 3.40 s | 264.7 | 2.83× | 6.58e-05 |
| Opt1 (32×8) | 3.36 s | 268.3 | 2.87× | 6.58e-05 |
| Opt2 (shared mem) | 3.40 s | 264.4 | 2.83× | 6.58e-05 |
| Opt3 (shared + 32×8) | 3.39 s | 265.1 | 2.83× | 6.58e-05 |

---

## Requisitos del entorno

- NVIDIA Jetson Nano (o GPU con soporte CUDA)
- Ubuntu 18.04
- CUDA Toolkit
- OpenCV 4.14
- `nvcc` (para GPU)
- `g++` + Meson + Ninja (para CPU)
- Nsight Systems (opcional, para perfilado)

---

## Referencias

- **Enunciado:** `Laboratorio_5___Computacion_Heterogenea___GPU.pdf`
- **Capítulo de lectura:** `Capitulo3-GPU-optimization.pdf`
- **Reporte final:** `ReporteLab5.md` (raíz del repositorio)

---

## Autores

- Grupo 3 — Semana 9
- Curso: Computación Heterogénea
- Laboratorio 5

---

## Uso de IA

Documentado en:
- `Parts D-E-F/Semana_9/drop-simulation/Resultados Ejercicio D y E.md`
- Enlace a Copilot: [Conversación](https://copilot.cloud.microsoft/chat/share/eyJzaGFyZUlkIjoiMGFmODU3MTMtMWE4Yy00ZjhmLTliOGMtMzExNTM1MTVhZGRhIiwiY29udmVyc2F0aW9uSWQiOiJmMTc2NDhiOS1jN2ZkLTRiN2ItODU3Zi00NDRmZDJiNTg1MjAifQ%3D%3D)
- Enlace a Claude: [Conversación](https://claude.ai/share/962d7965-0eb0-40bf-a8c6-96ff8fcd8ca1)

---

*Fin del README.*
