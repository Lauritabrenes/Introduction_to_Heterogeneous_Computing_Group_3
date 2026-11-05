#!/bin/bash
# Run on the Jetson Nano: bash run_all.sh
set -e
nvcc -O3 -std=c++14 -arch=sm_53 softmax_cuda.cu -o softmax_cuda
nvpmodel -q | tee power_mode.txt
echo "platform,precision,n,reps,measure,run,time_us,max_abs_err,max_rel_err" > results_gpu.csv

tegrastats --interval 100 --logfile tegrastats.log &
sleep 2
for N in 10 100 500 1000; do
  for P in fp16 fp32; do
    ./softmax_cuda --precision $P --n $N --reps 10000 --measure kernel
    ./softmax_cuda --precision $P --n $N --reps 10000 --measure end-to-end
  done
done
pkill tegrastats

# Profiler outputs
nvprof --print-gpu-trace --log-file nvprof_fp16_kernel.txt ./softmax_cuda --precision fp16 --n 100 --reps 1000 --measure kernel
nvprof --log-file nvprof_fp16_e2e.txt ./softmax_cuda --precision fp16 --n 100 --reps 1000 --measure end-to-end
nvprof --log-file nvprof_fp32_e2e.txt ./softmax_cuda --precision fp32 --n 100 --reps 1000 --measure end-to-end
echo "Done. See results_gpu.csv, tegrastats.log, nvprof_*.txt"
