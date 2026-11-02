# Ejercicio A: Implementación de Softmax en CPU

## 1. Descripción

Este ejercicio forma parte del Laboratorio 6 del curso de Computación Heterogénea del Instituto Tecnológico de Costa Rica.

Su objetivo es implementar la función Softmax utilizando diferentes precisiones numéricas y analizar su comportamiento en términos de exactitud y tiempo de ejecución.

El desarrollo inicial se realizó en Ubuntu 24.04 mediante GitHub Codespaces. Las mediciones obtenidas en este entorno son preliminares y deberán repetirse en la CPU de la NVIDIA Jetson Nano.

## 2. Implementación

Se desarrollaron las siguientes versiones:

- FP16: utiliza `_Float16` para representar los datos de entrada y salida. Las operaciones exponenciales y las reducciones se realizan utilizando FP32.
- FP32: utiliza `float` para los cálculos.
- FP64: utiliza `double` como referencia de exactitud.

Se implementó la versión numéricamente estable de Softmax:

\[
m = \max(x_i)
\]

\[
y_i = \frac{e^{x_i-m}}{\sum_j e^{x_j-m}}
\]

## 3. Organización de archivos

### Carpeta CPU

- `softmax_cpu.cpp`: implementación principal de Softmax, validaciones y mediciones.
- `softmax_cpu_validacion.cpp`: versión de respaldo de las validaciones iniciales.
- `softmax_cpu_respaldo_mediciones.cpp`: respaldo de la implementación con mediciones.

### Carpeta Resultados

- `validacion_fp32.txt`: primera validación de FP32.
- `validacion_fp16_fp32.txt`: comparación inicial entre FP16 y FP32.
- `validacion_fp16_fp32_fp64.txt`: comparación contra FP64.
- `validacion_tamanos_fp16_fp32.txt`: pruebas para diferentes tamaños.
- `validacion_completa_fp16_fp32_fp64.txt`: validación con errores absolutos y relativos.
- `mediciones_preliminares_cpu.txt`: primeras mediciones de tiempo.
- `mediciones_cpu.csv`: mediciones registradas en formato CSV.
- `mediciones_cpu_respaldo.csv`: respaldo de las mediciones.
- `medicion_individual_fp16.txt`: ejecución individual de FP16.
- `medicion_individual_fp32.txt`: ejecución individual de FP32.
- `tiempo_total_cpu.txt`: tiempo total del programa completo de pruebas.

Los archivos de resultados corresponden a pruebas preliminares realizadas en GitHub Codespaces.

## 4. Compilación en Ubuntu

Desde la carpeta principal del Ejercicio A:

```bash
g++ -O3 -std=c++17 CPU/softmax_cpu.cpp -o CPU/softmax_cpu -lm
```

## 5. Ejecución

### Pruebas completas

```bash
./CPU/softmax_cpu
```

Este comando ejecuta las validaciones y las mediciones para los tamaños establecidos.

### Ejecutar FP16

```bash
./CPU/softmax_cpu --precision fp16 --n 100 --reps 10000
```

### Ejecutar FP32

```bash
./CPU/softmax_cpu --precision fp32 --n 100 --reps 10000
```

Los parámetros disponibles son:

| Parámetro | Descripción |
|---|---|
| `--precision` | Selecciona FP16 o FP32 |
| `--n` | Tamaño del vector |
| `--reps` | Número de repeticiones |

## 6. Configuración experimental

Se utilizan vectores de entrada con valores comprendidos entre -4 y 4.

Para permitir la reproducibilidad se utiliza una semilla fija de 12345.

Los tamaños establecidos son:

- N = 10
- N = 100
- N = 500
- N = 1000

Las mediciones por lotes utilizan:

- Una ejecución inicial de calentamiento.
- 10 000 repeticiones por lote.
- Cinco mediciones por configuración.
- Cálculo del tiempo mínimo, mediana y máximo.

Los tiempos se expresan en microsegundos.

## 7. Validación numérica

Los resultados se comparan contra una referencia calculada en FP64.

Se calculan:

- Error absoluto máximo.
- Error relativo máximo.
- Suma de las salidas de Softmax.

En las pruebas preliminares, FP32 presentó errores menores que FP16 respecto a la referencia FP64.

## 8. Mediciones de rendimiento

Las mediciones se realizaron inicialmente en GitHub Codespaces sobre arquitectura x86_64.

En este entorno, la implementación FP32 presentó menores tiempos de ejecución que la variante FP16.

Estos resultados no deben interpretarse como mediciones representativas de la CPU de la Jetson Nano.

El archivo `mediciones_cpu.csv` contiene 40 mediciones preliminares correspondientes a cuatro tamaños, dos precisiones y cinco mediciones por configuración.

## 9. Trabajo pendiente

Para completar el experimento en el hardware solicitado se requiere:

- Ejecutar las pruebas en una NVIDIA Jetson Nano física.
- Registrar el modo de potencia, frecuencia, temperatura y núcleos activos.
- Comprobar automáticamente que las salidas estén en el intervalo [0,1].
- Verificar que la posición del máximo coincida con la referencia FP64.
- Realizar las mediciones definitivas de tiempo.
- Completar las mediciones de potencia y energía correspondientes al laboratorio.
- Comparar los resultados definitivos y elaborar las conclusiones.

## 10. Referencias

- Laboratorio 6, Computación Heterogénea, Instituto Tecnológico de Costa Rica.
- Material de las semanas 12 y 13 del curso.
- Documentación del compilador GCC y biblioteca estándar de C++.
