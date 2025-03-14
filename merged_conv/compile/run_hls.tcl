# Auto-generated TCL file for HLS
open_project project_1

set_top conv_via_tiling

add_files  merged_conv_top.cpp
add_files -tb merged_conv_tb.cpp
add_files -tb common.h

open_solution "solution1"

set_part xczu9eg-ffvb1156-2-e

create_clock -period 10 -name default

#csim_design

csynth_design

#cosim_design

#export_design -format ip_catalog

exit