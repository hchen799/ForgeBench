# Auto-generated TCL file for tiled ResNet-18 HLS
open_project -reset project_1

set_top top

add_files top.cpp
add_files top.h
add_files resnet18_tiled_scales.h

open_solution "solution1"

set_part xczu9eg-ffvb1156-2-e

create_clock -period 10 -name default

csynth_design

exit
