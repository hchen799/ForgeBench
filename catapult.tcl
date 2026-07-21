options defaults

options set Input/TargetPlatform x86_64

# Set the output directory
project new -name catapult_syrk

solution file add syrk.cpp -args {-I../../ForgeBench/catapult_shim}
solution file add syrk.h

go analyze

# Load Nangate Library
directive set -OPT_CONST_MULTS simple_one_adder
solution library add nangate-45nm_beh -- -rtlsyntool DesignCompiler -vendor Nangate -technology 045nm
solution library add ram_nangate-45nm-singleport_beh
directive set -PROTOTYPE_ROM false

solution design set kernel_syrk -top
go compile
go libraries

directive set -CLOCKS {clk {-CLOCK_PERIOD 10 -CLOCK_EDGE rising -CLOCK_UNCERTAINTY 0.0 -CLOCK_HIGH_TIME 5 -RESET_SYNC_NAME rst -RESET_ASYNC_NAME arst_n -RESET_KIND sync -RESET_SYNC_ACTIVE high -RESET_ASYNC_ACTIVE low -ENABLE_ACTIVE high}}

go assembly
go extract