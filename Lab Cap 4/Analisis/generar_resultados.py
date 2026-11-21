#!/usr/bin/env python3
"""generar_resultados.py - Laboratorio 6 (Semana 13), Computacion Heterogenea.

Lee los datos crudos de las tres plataformas y genera:
  - Analisis/resultados_resumen.csv   (mediana/min/max, error, potencia, energia)
  - Analisis/figuras/*.png            (tiempo vs N, error vs FP64, desglose KV260, energia)
  - Tablas Markdown por stdout        (se copian al README.md)

Uso (desde la raiz del repositorio):
  python Analisis/generar_resultados.py

Fuentes:
  CPU   : CPU/Resultados/mediciones_cpu.csv
          CPU/Resultados/validacion_completa_fp16_fp32_fp64.txt
  GPU   : GPU/results_gpu.csv
  KV260 : FPGA/Log_console.txt   (salida de host_kv260.py en la placa)
          Potencia: sensor INA260 (hwmon power1_input) leido durante la ejecucion.
"""
import csv
import os
import re
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_A = os.path.join(RAIZ, "CPU", "Resultados")
DIR_B = os.path.join(RAIZ, "GPU")
DIR_C = os.path.join(RAIZ, "FPGA")
SALIDA = os.path.join(RAIZ, "Analisis")
FIGS = os.path.join(SALIDA, "figuras")
os.makedirs(FIGS, exist_ok=True)

NS = [10, 100, 500, 1000]
PRECS = ["fp32", "fp16"]

# Potencia de la placa KV260 (INA260, lectura del sistema completo de la placa) en W
KV_POT_ACTIVA = {"fp32": 3.50, "fp16": 3.51}
KV_POT_REPOSO = {"fp32": 3.17, "fp16": 3.10}


def resumen(vals):
    return st.median(vals), min(vals), max(vals)


# ---------------------------------------------------------------- CPU
cpu_t = {}
with open(os.path.join(DIR_A, "mediciones_cpu.csv"), newline="") as f:
    for r in csv.DictReader(f):
        cpu_t.setdefault((r["precision"].lower(), int(r["N"])), []).append(float(r["tiempo_us"]))

cpu_err = {}
txt = open(os.path.join(DIR_A, "validacion_completa_fp16_fp32_fp64.txt"), encoding="utf-8").read()
for bloque in re.split(r"\nN = ", txt)[1:]:
    n = int(bloque.split("\n", 1)[0])
    for tipo, clave in (("absoluto", "abs"), ("relativo", "rel")):
        for p in ("FP32", "FP16"):
            m = re.search(rf"Error {tipo} maximo {p}: ([0-9.eE+-]+)", bloque)
            cpu_err[(p.lower(), n, clave)] = float(m.group(1))

# ---------------------------------------------------------------- GPU
gpu_t, gpu_err = {}, {}
with open(os.path.join(DIR_B, "results_gpu.csv"), newline="") as f:
    for r in csv.DictReader(f):
        k = (r["precision"], int(r["n"]), r["measure"])
        gpu_t.setdefault(k, []).append(float(r["time_us"]))
        gpu_err[(r["precision"], int(r["n"]), "abs")] = float(r["max_abs_err"])
        gpu_err[(r["precision"], int(r["n"]), "rel")] = float(r["max_rel_err"])

# ---------------------------------------------------------------- KV260
kv = {}
log = open(os.path.join(DIR_C, "Log_console.txt"), encoding="utf-8").read().splitlines()
actual = None
for linea in log:
    m = re.search(r"--precision (fp16|fp32) --n (\d+) --reps (\d+)", linea)
    if m:
        actual = (m.group(1), int(m.group(2))) if m.group(3) == "1000" else None
        continue
    if actual is None:
        continue
    m = re.match(r"(transfer|ip|total)\s+us\s+median=\s*([\d.]+)\s+min=\s*([\d.]+)\s+max=\s*([\d.]+)", linea)
    if m:
        kv.setdefault(actual, {})[m.group(1)] = tuple(float(v) for v in m.group(2, 3, 4))
    m = re.match(r"sum=(\S+)\s+in_range=(\S+)\s+argmax_ok=(\S+)\s+max_abs=(\S+)\s+max_rel=(\S+)", linea)
    if m:
        kv[actual].update(sum=float(m.group(1)), in_range=m.group(2) == "True",
                          argmax_ok=m.group(3) == "True",
                          abs=float(m.group(4)), rel=float(m.group(5)))

faltan = [(p, n) for p in PRECS for n in NS if (p, n) not in kv]
assert not faltan, f"Faltan corridas KV260 en el log: {faltan}"

# ---------------------------------------------------------------- resumen CSV
filas = []
for p in PRECS:
    for n in NS:
        med, mn, mx = resumen(cpu_t[(p, n)])
        filas.append(dict(plataforma="CPU", medida="funcion", precision=p, N=n,
                          t_med=med, t_min=mn, t_max=mx,
                          err_abs=cpu_err[(p, n, "abs")], err_rel=cpu_err[(p, n, "rel")]))
        for meas in ("kernel", "end-to-end"):
            med, mn, mx = resumen(gpu_t[(p, n, meas)])
            filas.append(dict(plataforma="GPU", medida=meas, precision=p, N=n,
                              t_med=med, t_min=mn, t_max=mx,
                              err_abs=gpu_err[(p, n, "abs")], err_rel=gpu_err[(p, n, "rel")]))
        d = kv[(p, n)]
        for etapa in ("transfer", "ip", "total"):
            med, mn, mx = d[etapa]
            fila = dict(plataforma="KV260", medida=etapa, precision=p, N=n,
                        t_med=med, t_min=mn, t_max=mx, err_abs=d["abs"], err_rel=d["rel"])
            if etapa == "total":
                P = KV_POT_ACTIVA[p]
                fila.update(potencia_W=P, energia_uJ=P * med, uJ_por_elem=P * med / n,
                            energia_dinamica_uJ=(P - KV_POT_REPOSO[p]) * med)
            filas.append(fila)

campos = ["plataforma", "medida", "precision", "N", "t_med", "t_min", "t_max",
          "err_abs", "err_rel", "potencia_W", "energia_uJ", "uJ_por_elem", "energia_dinamica_uJ"]
with open(os.path.join(SALIDA, "resultados_resumen.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=campos)
    w.writeheader()
    w.writerows(filas)


def fila(plat, medida, p, n):
    return next(r for r in filas if r["plataforma"] == plat and r["medida"] == medida
                and r["precision"] == p and r["N"] == n)


def fmt_err(v):
    return "NaN" if v != v else f"{v:.2e}"


# ---------------------------------------------------------------- tablas Markdown
for p in PRECS:
    print(f"\n### {p.upper()}\n")
    print("| Plataforma | N | Tiempo (us) mediana [min-max] | Potencia (W) | Energía (uJ) | uJ/elem. | Error máx. abs. |")
    print("|---|---:|---|---:|---:|---:|---:|")
    for n in NS:
        r = fila("CPU", "funcion", p, n)
        print(f"| CPU (función) | {n} | {r['t_med']:.3f} [{r['t_min']:.3f}-{r['t_max']:.3f}] | N/D | N/D | N/D | {fmt_err(r['err_abs'])} |")
    for n in NS:
        k = fila("GPU", "kernel", p, n)
        e = fila("GPU", "end-to-end", p, n)
        print(f"| GPU kernel / end-to-end | {n} | {k['t_med']:.1f} [{k['t_min']:.1f}-{k['t_max']:.1f}] / "
              f"{e['t_med']:.1f} [{e['t_min']:.1f}-{e['t_max']:.1f}] | N/D | N/D | N/D | {fmt_err(k['err_abs'])} |")
    for n in NS:
        r = fila("KV260", "total", p, n)
        print(f"| KV260 (total host) | {n} | {r['t_med']:.1f} [{r['t_min']:.1f}-{r['t_max']:.1f}] | "
              f"{r['potencia_W']:.2f} | {r['energia_uJ']:.1f} | {r['uJ_por_elem']:.2f} | {fmt_err(r['err_abs'])} |")

print("\n### Desglose KV260 (mediana, us)\n")
print("| Precisión | N | Transferencia | IP (vista desde host) | Total | % transferencia | suma salida | rango/argmax |")
print("|---|---:|---:|---:|---:|---:|---:|---|")
for p in PRECS:
    for n in NS:
        d = kv[(p, n)]
        tr, ip, tot = d["transfer"][0], d["ip"][0], d["total"][0]
        print(f"| {p.upper()} | {n} | {tr:.1f} | {ip:.1f} | {tot:.1f} | {100 * tr / tot:.0f}% | "
              f"{d['sum']:.4g} | {'ok' if d['in_range'] else 'FALLA'}/{'ok' if d['argmax_ok'] else 'FALLA'} |")

print("\n### Error relativo máximo vs FP64\n")
print("| N | CPU FP32 | CPU FP16 | GPU FP32 | GPU FP16 | KV260 FP32 | KV260 FP16 |")
print("|---:|---:|---:|---:|---:|---:|---:|")
for n in NS:
    print(f"| {n} | {fmt_err(cpu_err[('fp32', n, 'rel')])} | {fmt_err(cpu_err[('fp16', n, 'rel')])} | "
          f"{fmt_err(gpu_err[('fp32', n, 'rel')])} | {fmt_err(gpu_err[('fp16', n, 'rel')])} | "
          f"{fmt_err(kv[('fp32', n)]['rel'])} | {fmt_err(kv[('fp16', n)]['rel'])} |")

# ---------------------------------------------------------------- figuras
# Paleta categórica validada (slots 1-3): CPU azul, GPU naranja, KV260 aqua
COL = {"CPU": "#2a78d6", "GPU": "#eb6834", "KV260": "#1baf7a"}
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "legend.frameon": False, "lines.linewidth": 2,
})


def serie(ax, xs, rows, color, label, ls="-", marker="o"):
    med = [r["t_med"] for r in rows]
    lo = [r["t_med"] - r["t_min"] for r in rows]
    hi = [r["t_max"] - r["t_med"] for r in rows]
    ax.errorbar(xs, med, yerr=[lo, hi], color=color, ls=ls, marker=marker, ms=6,
                capsize=3, elinewidth=1, label=label)


# 1) Tiempo vs N
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for ax, p in zip(axs, PRECS):
    serie(ax, NS, [fila("CPU", "funcion", p, n) for n in NS], COL["CPU"], "CPU (función)")
    serie(ax, NS, [fila("GPU", "kernel", p, n) for n in NS], COL["GPU"], "GPU kernel", marker="s")
    serie(ax, NS, [fila("GPU", "end-to-end", p, n) for n in NS], COL["GPU"], "GPU end-to-end", ls="--", marker="s")
    serie(ax, NS, [fila("KV260", "ip", p, n) for n in NS], COL["KV260"], "KV260 IP (host)", marker="^")
    serie(ax, NS, [fila("KV260", "total", p, n) for n in NS], COL["KV260"], "KV260 total host", ls="--", marker="^")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(NS); ax.set_xticklabels(NS)
    ax.set_title(f"{p.upper()} — tiempo por llamada (mediana, barras = min-max)", fontsize=10)
    ax.set_xlabel("N (elementos)")
axs[0].set_ylabel("Tiempo por llamada (µs, escala log)")
axs[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "tiempo_vs_N.png"), dpi=150)
plt.close(fig)

# 2) Error relativo vs FP64 (sin dispersión: la entrada es determinista, el error es idéntico en las 5 corridas)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, p in zip(axs, PRECS):
    ax.plot(NS, [cpu_err[(p, n, "rel")] for n in NS], "-o", color=COL["CPU"], label="CPU")
    ax.plot(NS, [gpu_err[(p, n, "rel")] for n in NS], "-s", color=COL["GPU"], label="GPU")
    kvv = [kv[(p, n)]["rel"] for n in NS]
    xs = [n for n, v in zip(NS, kvv) if v == v]
    ax.plot(xs, [v for v in kvv if v == v], "-^", color=COL["KV260"], label="KV260")
    if p == "fp16":
        ax.annotate("KV260 FP16: salida inválida\n(N=500 y 1000 dan NaN)", xy=(100, kvv[1]),
                    xytext=(160, 3e-1), fontsize=8, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=AXIS))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(NS); ax.set_xticklabels(NS)
    ax.set_title(f"{p.upper()} — error relativo máximo vs FP64", fontsize=10)
    ax.set_xlabel("N (elementos)")
axs[0].set_ylabel("Error relativo máximo (escala log)")
axs[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "error_vs_N.png"), dpi=150)
plt.close(fig)

# 3) Desglose de tiempo KV260 (barras apiladas transferencia + IP)
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, p in zip(axs, PRECS):
    xs = range(len(NS))
    tr = [kv[(p, n)]["transfer"][0] for n in NS]
    ip = [kv[(p, n)]["ip"][0] for n in NS]
    ax.bar(xs, tr, width=0.55, color=COL["KV260"], label="Transferencia (sync + copia)",
           edgecolor="#fcfcfb", linewidth=2)
    ax.bar(xs, ip, width=0.55, bottom=tr, color="#2a78d6", label="IP (start→done vista desde host)",
           edgecolor="#fcfcfb", linewidth=2)
    for x, a, b in zip(xs, tr, ip):
        ax.text(x, a + b + 4, f"{a + b:.0f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(list(xs)); ax.set_xticklabels(NS)
    ax.set_title(f"KV260 {p.upper()} — desglose del tiempo total", fontsize=10)
    ax.set_xlabel("N (elementos)")
    ax.grid(axis="x", visible=False)
axs[0].set_ylabel("Tiempo por llamada (µs, mediana)")
axs[1].legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "kv260_desglose.png"), dpi=150)
plt.close(fig)

# 4) Energía KV260: por llamada y por elemento (dos gráficos, un eje cada uno)
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for p, ls, mk in (("fp32", "-", "o"), ("fp16", "--", "s")):
    rows = [fila("KV260", "total", p, n) for n in NS]
    P = KV_POT_ACTIVA[p]
    e = [r["energia_uJ"] for r in rows]
    elo = [P * (r["t_med"] - r["t_min"]) for r in rows]
    ehi = [P * (r["t_max"] - r["t_med"]) for r in rows]
    axs[0].errorbar(NS, e, yerr=[elo, ehi], ls=ls, marker=mk, color=COL["KV260"] if p == "fp32" else "#2a78d6",
                    capsize=3, label=p.upper())
    axs[1].errorbar(NS, [v / n for v, n in zip(e, NS)],
                    yerr=[[v / n for v, n in zip(elo, NS)], [v / n for v, n in zip(ehi, NS)]],
                    ls=ls, marker=mk, color=COL["KV260"] if p == "fp32" else "#2a78d6", capsize=3, label=p.upper())
axs[0].set_title("KV260 — energía por llamada (placa completa)", fontsize=10)
axs[0].set_ylabel("Energía por llamada (µJ)")
axs[1].set_title("KV260 — energía por elemento", fontsize=10)
axs[1].set_ylabel("µJ por elemento (escala log)")
axs[1].set_yscale("log")
for ax in axs:
    ax.set_xscale("log"); ax.set_xticks(NS); ax.set_xticklabels(NS); ax.set_xlabel("N (elementos)")
    ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "kv260_energia.png"), dpi=150)
plt.close(fig)

print(f"\nEscrito: {os.path.relpath(os.path.join(SALIDA, 'resultados_resumen.csv'), RAIZ)} y figuras en Analisis/figuras/")
