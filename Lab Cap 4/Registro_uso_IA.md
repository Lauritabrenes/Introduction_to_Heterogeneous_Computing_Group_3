# Registro de uso de IA — Lab Cap 4 (Laboratorio Semana 13)

| | |
|---|---|
| Herramienta | Claude (Anthropic), mediante Claude Code en VS Code |
| Fecha | 20 y 21 de noviembre de 2026 |
| Uso | Organizar el repositorio y redactar el documento de entrega (`Readme.md`) a partir de los datos ya medidos |

> Este archivo es un **resumen de la conversación** escrito por el asistente al final de la sesión, no una transcripción literal. Los pedidos del estudiante aparecen textuales o casi textuales; las respuestas aparecen resumidas. Solo cubre la parte relacionada con este laboratorio.

## Qué hizo la IA 
- Revisó la carpeta del laboratorio y el enunciado.
- Escribió el script `Analisis/generar_resultados.py`, que lee los datos crudos (`mediciones_cpu.csv`, `results_gpu.csv`, `Log_console.txt`) y genera las tablas y las figuras.
- Redactó el `Readme.md`. Después, a pedido del grupo, lo resumió y ajustó.
- Reorganizó las carpetas en `CPU/`, `GPU/`, `FPGA/` y `Analisis/` y subió esa reorganización al repositorio.
- Señaló inconsistencias en los datos. El grupo decidió cómo manejar cada una.

## Cronología

1. **Pedido:** "Revisa la carpeta. Tenemos todos los resultados del lab, también necesito modelar esos datos con gráficas."
   - Se revisaron el enunciado, el código y los datos de los tres ejercicios.
   - Se generaron tablas y figuras con un script reproducible y se redactó un primer `Readme.md`.
   - Se señalaron estos hallazgos:
     - La CPU se midió en Codespaces (x86) y no en la Jetson.
     - Las semillas difieren entre plataformas (12345 en CPU, 42 en GPU y FPGA).
     - `tegrastats.log` no contiene datos de potencia.
     - No están en el repositorio las salidas de `nvprof` ni los reportes de HLS.
     - La explicación de "desbordamiento" del error FP16 en la KV260 no se sostiene: con la forma estable, la suma vale como mucho N ≤ 1000.
     - `softmax_fp16.hwh` es idéntico a `softmax_fp32.hwh`.
2. **Pedido:** "Renómbralas y organízalas."
   - Se renombraron `Ejercicio A/B/C` a `CPU/GPU/FPGA` y se actualizaron las rutas en los README y en el script.
3. **Pedido:** las fórmulas no se veían bien en el visor de Markdown.
   - Se reemplazó el LaTeX por texto con símbolos Unicode.
4. **Pedido:** "¿Qué pasó con los datos de potencia y demás que pide el ejercicio?"
   - Se volvió a buscar en todos los archivos y se confirmó que solo la KV260 tiene potencia medida.
   - Se repuso la tabla de condiciones de medición del Ejercicio D.
5. **Pedido:** subir la estructura a `Lab Cap 4` del repositorio del grupo, conservando el `Readme.md` que ya estaba allí.
   - Se hizo el commit `2716ec8` solo dentro de `Lab Cap 4`, con confirmación explícita del estudiante antes del push.

