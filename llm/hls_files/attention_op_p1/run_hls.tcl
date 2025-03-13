# Auto-generated TCL file for HLS
open_project project_1

set_top top

add_files  top.cpp
add_files -tb tb_top.cpp
add_files -tb top.h

add_files -tb DRAM_attn_input.txt
add_files -tb DRAM_weights_q.txt
add_files -tb DRAM_weights_k.txt
add_files -tb DRAM_weights_v.txt
add_files -tb DRAM_output.txt
open_solution "solution1"

set_part xczu9eg-ffvb1156-2-e

create_clock -period 10 -name default

csim_design

csynth_design

cosim_design

export_design -format ip_catalog

exit