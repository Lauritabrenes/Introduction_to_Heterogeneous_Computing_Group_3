# Laboratorio 6: Computación Heterogénea - FPGA (Kria KV260)

## 1. Guía de Ejecución y Reproducción

### Fase 1: Síntesis de Hardware (Vitis HLS)
Para generar los bloques IP para FP32 y FP16, se utiliza el script de automatización `run_hls.tcl`.
* **Nota para FP16:** Se desactivó la cosimulación (`csim_design`) para evitar el error `SIGSEGV` provocado por la emulación de `hls::half` en la arquitectura x86 del host. Se añadió la bandera `-DACC_HALF` para forzar las reducciones estrictamente en 16 bits.
```cmd
vitis_hls -f run_hls.tcl
python extract_reports.py
```

### Fase 2: Implementación y Transferencia
1. En Vivado, se apunta el **IP Repository** al IP exportado (ej. `.../prj_FP32_v3/sol1/impl/ip`).
2. Se actualiza el *Block Design*, se genera el *Wrapper* y luego el **Bitstream**.
3. Se renombran los archivos generados a `softmax_fp32.bit` / `.hwh` (o `fp16`).
4. Se transfieren a la Kria KV260 vía SCP/rsync usando el servidor de salto:
```bash
scp -J el5859@51.161.8.220:9222 softmax_fp32.bit softmax_fp32.hwh el5859@192.168.0.83:~/Lab6_EjercicioC/
```

### Fase 3: Benchmark en la FPGA
Debido a restricciones de seguridad de `sudo` que borran el `PATH` de XRT, se inyectan las variables explícitamente en el comando. Ejecutado desde SSH en la Kria:
```bash
sudo env PATH=$PATH:/opt/xilinx/xrt/bin XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 host_kv260.py --bit softmax_fp32.bit --precision fp32 --n 100 --reps 1000
```
*(Se varía `--n` a 10, 100, 500, 1000 y `--precision` a fp32/fp16).*

### Fase 4: Medición de Consumo
Se lanza un benchmark infinito (`--reps 1000000`) y en una segunda terminal se lee el sensor interno INA260 de la Kria KV260:
```bash
cat /sys/class/hwmon/hwmon*/power1_input 2>/dev/null
```
* **Medición FP32:** Reposo = 3.17 W (3170000 $\mu W$) | Activo = 3.50 W (3500000 $\mu W$)
* **Medición FP16:** Reposo = 3.10 W (3100000 $\mu W$) | Activo = 3.51 W (3510000 $\mu W$)

---

## 2. Resultados: Tabla 8 (Kria KV260)

Las siguientes tablas muestran los resultados de la implementación HLS (Variante Dataflow) ejecutados en la Kria KV260. La energía se calculó utilizando la fórmula $E (\mu J) = Potencia_{activa} \times Tiempo$.

### Tabla 8.1: Precisión FP32 (Potencia Activa: 3.50 W)

| Plataforma | N | Tiempo (us) | Potencia (W) | Energía ($\mu$J) | J/elem. ($\mu$J) | Error máx. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| KV260 | 10 | 223.222 | 3.50 | 781.28 | 78.13 | 1.781e-08 |
| KV260 | 100 | 226.528 | 3.50 | 792.85 | 7.93 | 1.132e-08 |
| KV260 | 500 | 246.542 | 3.50 | 862.90 | 1.73 | 1.280e-09 |
| KV260 | 1000 | 277.793 | 3.50 | 972.28 | 0.97 | 1.243e-09 |

### Tabla 8.2: Precisión FP16 (Potencia Activa: 3.51 W)

| Plataforma | N | Tiempo (us) | Potencia (W) | Energía ($\mu$J) | J/elem. ($\mu$J) | Error máx. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| KV260 | 10 | 223.910 | 3.51 | 785.92 | 78.59 | 5.194e+04 |
| KV260 | 100 | 228.417 | 3.51 | 801.74 | 8.02 | 1.788e+00 |
| KV260 | 500 | 243.752 | 3.51 | 855.57 | 1.71 | NaN (Overflow) |
| KV260 | 1000 | 263.445 | 3.51 | 924.69 | 0.92 | NaN (Overflow) |

---

## 3. Observaciones y Análisis (Respuestas a Preguntas)

**1. Cuello de botella y overhead (Pregunta 1 y 4):**
El tiempo reportado incluye el overhead de transferir datos del procesador ARM a la FPGA vía AXI. Al analizar los logs, la latencia pura del bloque IP de hardware es minúscula (ej. 25 $\mu s$ para N=100), pero la transferencia toma más de 200 $\mu s$. Esto provoca que para vectores pequeños ($N=10$), el costo energético por elemento sea absurdo ($\sim 78 \ \mu J$), ya que la mayor parte del tiempo el acelerador está ocioso esperando datos. A medida que $N$ crece ($N=1000$), el cálculo se amortiza y el costo cae drásticamente a $\sim 0.9 \ \mu J/elem$.

**2. Análisis de Error FP32 vs FP16 (Pregunta 2):**
* **FP32:** Presenta un error microscópico (del orden de $10^{-8}$) respecto a la referencia de software en doble precisión (FP64), demostrando que es ideal para esta carga matemática.
* **FP16:** El hardware sufre un colapso matemático masivo por desbordamiento (*Overflow*). Al llegar a $N=500$ y $N=1000$, el error reporta `NaN`. Esto se debe directamente a la **reducción** (acumulador de sumas en el denominador de la softmax). Dado que el formato half de 16 bits tiene un límite superior de $\sim 65504$, la sumatoria de exponenciales supera rápidamente este valor. Esto demuestra que la caída de precisión se debe a la limitación del rango dinámico durante la reducción, no a la función exponencial individual.
* *Anomalía en N=10 (FP16):* El error gigantesco provocado por una sumatoria negativa (`-50053`) indica un problema de alineación o truncamiento en el bus de datos AXI al transferir floats de 16 bits a través de interfaces diseñadas para 32 o 64 bits, causando que los bits de datos se interpreten como bits de signo en la Kria.

**3. Consumo FP32 vs FP16 (Pregunta 6):**
A nivel de potencia instantánea, el acelerador bajo carga máxima demanda casi lo mismo (3.50 W en FP32 vs 3.51 W en FP16). Sin embargo, al observar las ejecuciones largas ($N=1000$), el tiempo total de FP16 ($263 \ \mu s$) es ligeramente inferior al de FP32 ($277 \ \mu s$). Esta reducción en la latencia se traduce en un menor consumo energético total por elemento ($0.92 \ \mu J$ vs $0.97 \ \mu J$).

Conclusión: el rendimiento en velocidad es similar por limitaciones de E/S, la precisión de 16 bits sufre penalizaciones numéricas en algoritmos con reducciones acumulativas grandes.