#!/usr/bin/env python3
"""host_kv260.py - runs softmax_hls on the KV260 (Kria-PYNQ) and measures SEPARATELY:
  - transfer time : host copy + cache sync to device, and sync back + copy out
  - IP time       : start -> ap_done, measured from the host (includes polling overhead)
  - total time    : everything observed by the host for one call
Also compares against FP64 and appends raw data to results_kv260.csv.

Usage (as root, in the folder with softmax_fp32.bit/.hwh and inputs_*.bin):
  sudo python3 host_kv260.py --bit softmax_fp32.bit --precision fp32 --n 100 --reps 1000
  sudo python3 host_kv260.py --bit softmax_fp16.bit --precision fp16 --n 100 --reps 1000

IMPORTANT: verify the register offsets below against the generated driver header
prj_<..>_v3/sol1/impl/misc/drivers/softmax_hls_v1_0/src/xsoftmax_hls_hw.h
"""
import argparse
import csv
import os
import time

import numpy as np
from pynq import Overlay, allocate

# --- AXI-Lite register map (typical for this interface; CHECK xsoftmax_hls_hw.h) ---
CTRL = 0x00   # bit0 ap_start, bit1 ap_done, bit2 ap_idle
X_LO, X_HI = 0x10, 0x14
Y_LO, Y_HI = 0x1C, 0x20
N_REG = 0x28

ap = argparse.ArgumentParser()
ap.add_argument("--bit", required=True)
ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp32")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--reps", type=int, default=1000)
ap.add_argument("--runs", type=int, default=5)
ap.add_argument("--csv", default="results_kv260.csv")
a = ap.parse_args()

dt = np.float16 if a.precision == "fp16" else np.float32
x32 = np.fromfile(f"inputs_{a.n}.bin", dtype=np.float32)
x = x32.astype(dt)

ol = Overlay(a.bit)
ip = ol.softmax_hls_0
xin = allocate(shape=(1000,), dtype=dt)
yout = allocate(shape=(1000,), dtype=dt)

ip.write(X_LO, xin.device_address & 0xFFFFFFFF)
ip.write(X_HI, xin.device_address >> 32)
ip.write(Y_LO, yout.device_address & 0xFFFFFFFF)
ip.write(Y_HI, yout.device_address >> 32)
ip.write(N_REG, a.n)


def one_call():
    t0 = time.perf_counter()
    xin[:a.n] = x
    xin.sync_to_device()
    t1 = time.perf_counter()
    ip.write(CTRL, 1)
    while (ip.read(CTRL) & 0x2) == 0:
        pass
    t2 = time.perf_counter()
    yout.sync_from_device()
    y = np.array(yout[:a.n])
    t3 = time.perf_counter()
    return (t1 - t0) + (t3 - t2), t2 - t1, t3 - t0, y


one_call()  # warm-up (excluded)

new = not os.path.exists(a.csv)
f = open(a.csv, "a", newline="")
w = csv.writer(f)
if new:
    w.writerow(["platform", "precision", "n", "reps", "run",
                "transfer_us", "ip_us", "total_us", "max_abs_err", "max_rel_err"])

xr = x.astype(np.float64)
yr = np.exp(xr - xr.max()); yr /= yr.sum()

res = []
for r in range(max(5, a.runs)):
    tr = ipt = tot = 0.0
    for _ in range(a.reps):
        dtr, dip, dtot, y = one_call()
        tr += dtr; ipt += dip; tot += dtot
    yo = y.astype(np.float64)
    abs_err = np.abs(yo - yr)
    mabs, mrel = abs_err.max(), (abs_err / yr).max()
    row = [tr / a.reps * 1e6, ipt / a.reps * 1e6, tot / a.reps * 1e6]
    res.append(row)
    w.writerow(["KV260", a.precision, a.n, a.reps, r, *[f"{v:.3f}" for v in row],
                f"{mabs:.6e}", f"{mrel:.6e}"])
f.close()

res = np.array(res)
for i, name in enumerate(["transfer", "ip", "total"]):
    c = res[:, i]
    print(f"{name:8s} us  median={np.median(c):9.3f}  min={c.min():9.3f}  max={c.max():9.3f}")
print(f"sum={yo.sum():.6f}  in_range={bool(((yo >= 0) & (yo <= 1)).all())}  "
      f"argmax_ok={yo.argmax() == yr.argmax()}  max_abs={mabs:.3e}  max_rel={mrel:.3e}")
print("Pure IP latency = cosim cycles / clock (e.g. cycles / 100 MHz) -> compare with ip_us")
