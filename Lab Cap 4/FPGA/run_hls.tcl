# run_hls.tcl - builds every precision x variant combination
# Usage (in this folder):  vitis_hls -f run_hls.tcl
# Each combination goes into prj_<FP32|FP16>_v<0..3>/sol1
#   csim   -> sol1/csim/report
#   csynth -> sol1/syn/report/csynth.rpt (+ csynth.xml)
#   cosim  -> sol1/sim/report/softmax_hls_cosim.rpt  (runs N=10,100,500,1000 from the testbench)

set part     xck26-sfvc784-2LV-c   ;# KV260 (K26 SOM)
set period   10                    ;# ns -> 100 MHz target (record it in the report)
set do_cosim 1                     ;# set 0 to skip cosim (slow) while iterating
set variants {0 1 2 3}
set dtypes   {FP32 FP16}

foreach dt $dtypes {
    foreach v $variants {
        set flags "-DVARIANT=$v -DUF=8"
        if {$dt eq "FP16"} { append flags " -DDATA_FP16" }

        open_project -reset prj_${dt}_v${v}
        set_top softmax_hls
        add_files     softmax_hls.cpp -cflags $flags
        add_files -tb softmax_tb.cpp  -cflags $flags
        open_solution -reset sol1 -flow_target vivado
        set_part $part
        create_clock -period $period -name default

        csim_design
        csynth_design
        if {$do_cosim} { cosim_design -trace_level none }
        # Export the final (dataflow) variant as an IP for Vivado
        if {$v == 3} { export_design -format ip_catalog }
        close_project
    }
}
exit
