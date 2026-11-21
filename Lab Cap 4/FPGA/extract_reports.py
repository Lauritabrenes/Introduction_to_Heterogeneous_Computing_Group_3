#!/usr/bin/env python3
"""extract_reports.py - collect latency, II, clock and resources from every
prj_*/sol1/syn/report/csynth.xml into hls_summary.csv (+ cosim latency if present).
Usage: python extract_reports.py   (run in the folder that contains prj_*)
"""
import csv
import glob
import os
import re
import xml.etree.ElementTree as ET


def txt(root, *paths):
    for p in paths:
        v = root.findtext(p)
        if v is not None:
            return v.strip()
    return ""


def cosim_latency(sol_dir):
    rpt = os.path.join(sol_dir, "sim", "report", "softmax_hls_cosim.rpt")
    if not os.path.exists(rpt):
        return "", "", "", ""
    with open(rpt, errors="ignore") as f:
        for line in f:
            # | Verilog| Pass| min| avg| max| ...
            if "Verilog" in line and "Pass" in line:
                cols = [c.strip() for c in line.split("|") if c.strip()]
                nums = [c for c in cols if re.fullmatch(r"\d+|NA", c)]
                return ("Pass",) + tuple((nums + ["", "", ""])[:3])
            if "Verilog" in line and "Fail" in line:
                return "Fail", "", "", ""
    return "", "", "", ""


rows = []
for xml in sorted(glob.glob("prj_*/sol1/syn/report/csynth.xml")):
    prj = xml.split(os.sep)[0]
    r = ET.parse(xml).getroot()
    pe = "PerformanceEstimates/"
    res = "AreaEstimates/Resources/"
    status, cmin, cavg, cmax = cosim_latency(os.path.join(prj, "sol1"))
    rows.append({
        "project": prj,
        "target_clk_ns": txt(r, "UserAssignments/TargetClockPeriod"),
        "est_clk_ns": txt(r, pe + "SummaryOfTimingAnalysis/EstimatedClockPeriod"),
        "lat_best_cyc": txt(r, pe + "SummaryOfOverallLatency/Best-caseLatency"),
        "lat_worst_cyc": txt(r, pe + "SummaryOfOverallLatency/Worst-caseLatency"),
        "II_min": txt(r, pe + "SummaryOfOverallLatency/Interval-min"),
        "II_max": txt(r, pe + "SummaryOfOverallLatency/Interval-max"),
        "LUT": txt(r, res + "LUT"),
        "FF": txt(r, res + "FF"),
        "BRAM_18K": txt(r, res + "BRAM_18K"),
        "URAM": txt(r, res + "URAM"),
        "DSP": txt(r, res + "DSP", res + "DSP48E"),
        "cosim": status, "cosim_lat_min": cmin, "cosim_lat_avg": cavg, "cosim_lat_max": cmax,
    })

if rows:
    with open("hls_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for row in rows:
        print(row)
    print("\nWrote hls_summary.csv")
else:
    print("No csynth.xml found. Run vitis_hls -f run_hls.tcl first.")
# Loop-level II (L_max, L_exp, ...) is in csynth.rpt -> 'Loop' table; copy it into the report.
