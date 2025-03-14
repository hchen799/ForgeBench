set moduleName conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4
set isTopModule 0
set isCombinational 0
set isDatapathOnly 0
set isPipelined 1
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set hasInterrupt 0
set DLRegFirstOffset 0
set DLRegItemOffset 0
set C_modelName {conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4}
set C_modelType { void 0 }
set ap_memory_interface_dict [dict create]
dict set ap_memory_interface_dict weight_0_0 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict weight_0_1 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict weight_0_2 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict weight_1_0 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict weight_1_1 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict weight_1_2 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict weight_2_0 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict weight_2_1 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict weight_2_2 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict outData { MEM_WIDTH 32 MEM_SIZE 12544 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict inData_0_0 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict inData_0_1 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict inData_0_2 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict inData_1_0 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict inData_1_1 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict inData_1_2 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict inData_2_0 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict inData_2_1 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
dict set ap_memory_interface_dict inData_2_2 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
set C_modelArgList {
	{ outTileW int 32 regular  }
	{ zext_ln25_3 int 70 regular  }
	{ mul_ln25 int 64 regular  }
	{ zext_ln25 int 6 regular  }
	{ weight_0_0 float 32 regular {array 4096 { 1 3 } 1 1 }  }
	{ weight_0_1 float 32 regular {array 4096 { 1 3 } 1 1 }  }
	{ weight_0_2 float 32 regular {array 4096 { 1 3 } 1 1 }  }
	{ weight_1_0 float 32 regular {array 4096 { 1 3 } 1 1 }  }
	{ weight_1_1 float 32 regular {array 4096 { 1 3 } 1 1 }  }
	{ weight_1_2 float 32 regular {array 4096 { 1 3 } 1 1 }  }
	{ weight_2_0 float 32 regular {array 4096 { 1 3 } 1 1 }  }
	{ weight_2_1 float 32 regular {array 4096 { 1 3 } 1 1 }  }
	{ weight_2_2 float 32 regular {array 4096 { 1 3 } 1 1 }  }
	{ sext_ln25 int 32 regular  }
	{ zext_ln34 int 9 regular  }
	{ outData float 32 regular {array 3136 { 0 1 } 1 1 }  }
	{ inData_0_0 float 32 regular {array 4096 { 1 3 } 1 1 }  }
	{ inData_0_1 float 32 regular {array 4096 { 1 3 } 1 1 }  }
	{ inData_0_2 float 32 regular {array 4096 { 1 3 } 1 1 }  }
	{ inData_1_0 float 32 regular {array 4096 { 1 3 } 1 1 }  }
	{ inData_1_1 float 32 regular {array 4096 { 1 3 } 1 1 }  }
	{ inData_1_2 float 32 regular {array 4096 { 1 3 } 1 1 }  }
	{ inData_2_0 float 32 regular {array 4096 { 1 3 } 1 1 }  }
	{ inData_2_1 float 32 regular {array 4096 { 1 3 } 1 1 }  }
	{ inData_2_2 float 32 regular {array 4096 { 1 3 } 1 1 }  }
}
set hasAXIMCache 0
set hasAXIML2Cache 0
set AXIMCacheInstDict [dict create]
set C_modelArgMapList {[ 
	{ "Name" : "outTileW", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "zext_ln25_3", "interface" : "wire", "bitwidth" : 70, "direction" : "READONLY"} , 
 	{ "Name" : "mul_ln25", "interface" : "wire", "bitwidth" : 64, "direction" : "READONLY"} , 
 	{ "Name" : "zext_ln25", "interface" : "wire", "bitwidth" : 6, "direction" : "READONLY"} , 
 	{ "Name" : "weight_0_0", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "weight_0_1", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "weight_0_2", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "weight_1_0", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "weight_1_1", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "weight_1_2", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "weight_2_0", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "weight_2_1", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "weight_2_2", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "sext_ln25", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "zext_ln34", "interface" : "wire", "bitwidth" : 9, "direction" : "READONLY"} , 
 	{ "Name" : "outData", "interface" : "memory", "bitwidth" : 32, "direction" : "READWRITE"} , 
 	{ "Name" : "inData_0_0", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "inData_0_1", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "inData_0_2", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "inData_1_0", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "inData_1_1", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "inData_1_2", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "inData_2_0", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "inData_2_1", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "inData_2_2", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} ]}
# RTL Port declarations: 
set portNum 73
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ outTileW sc_in sc_lv 32 signal 0 } 
	{ zext_ln25_3 sc_in sc_lv 70 signal 1 } 
	{ mul_ln25 sc_in sc_lv 64 signal 2 } 
	{ zext_ln25 sc_in sc_lv 6 signal 3 } 
	{ weight_0_0_address0 sc_out sc_lv 12 signal 4 } 
	{ weight_0_0_ce0 sc_out sc_logic 1 signal 4 } 
	{ weight_0_0_q0 sc_in sc_lv 32 signal 4 } 
	{ weight_0_1_address0 sc_out sc_lv 12 signal 5 } 
	{ weight_0_1_ce0 sc_out sc_logic 1 signal 5 } 
	{ weight_0_1_q0 sc_in sc_lv 32 signal 5 } 
	{ weight_0_2_address0 sc_out sc_lv 12 signal 6 } 
	{ weight_0_2_ce0 sc_out sc_logic 1 signal 6 } 
	{ weight_0_2_q0 sc_in sc_lv 32 signal 6 } 
	{ weight_1_0_address0 sc_out sc_lv 12 signal 7 } 
	{ weight_1_0_ce0 sc_out sc_logic 1 signal 7 } 
	{ weight_1_0_q0 sc_in sc_lv 32 signal 7 } 
	{ weight_1_1_address0 sc_out sc_lv 12 signal 8 } 
	{ weight_1_1_ce0 sc_out sc_logic 1 signal 8 } 
	{ weight_1_1_q0 sc_in sc_lv 32 signal 8 } 
	{ weight_1_2_address0 sc_out sc_lv 12 signal 9 } 
	{ weight_1_2_ce0 sc_out sc_logic 1 signal 9 } 
	{ weight_1_2_q0 sc_in sc_lv 32 signal 9 } 
	{ weight_2_0_address0 sc_out sc_lv 12 signal 10 } 
	{ weight_2_0_ce0 sc_out sc_logic 1 signal 10 } 
	{ weight_2_0_q0 sc_in sc_lv 32 signal 10 } 
	{ weight_2_1_address0 sc_out sc_lv 12 signal 11 } 
	{ weight_2_1_ce0 sc_out sc_logic 1 signal 11 } 
	{ weight_2_1_q0 sc_in sc_lv 32 signal 11 } 
	{ weight_2_2_address0 sc_out sc_lv 12 signal 12 } 
	{ weight_2_2_ce0 sc_out sc_logic 1 signal 12 } 
	{ weight_2_2_q0 sc_in sc_lv 32 signal 12 } 
	{ sext_ln25 sc_in sc_lv 32 signal 13 } 
	{ zext_ln34 sc_in sc_lv 9 signal 14 } 
	{ outData_address0 sc_out sc_lv 12 signal 15 } 
	{ outData_ce0 sc_out sc_logic 1 signal 15 } 
	{ outData_we0 sc_out sc_logic 1 signal 15 } 
	{ outData_d0 sc_out sc_lv 32 signal 15 } 
	{ outData_address1 sc_out sc_lv 12 signal 15 } 
	{ outData_ce1 sc_out sc_logic 1 signal 15 } 
	{ outData_q1 sc_in sc_lv 32 signal 15 } 
	{ inData_0_0_address0 sc_out sc_lv 12 signal 16 } 
	{ inData_0_0_ce0 sc_out sc_logic 1 signal 16 } 
	{ inData_0_0_q0 sc_in sc_lv 32 signal 16 } 
	{ inData_0_1_address0 sc_out sc_lv 12 signal 17 } 
	{ inData_0_1_ce0 sc_out sc_logic 1 signal 17 } 
	{ inData_0_1_q0 sc_in sc_lv 32 signal 17 } 
	{ inData_0_2_address0 sc_out sc_lv 12 signal 18 } 
	{ inData_0_2_ce0 sc_out sc_logic 1 signal 18 } 
	{ inData_0_2_q0 sc_in sc_lv 32 signal 18 } 
	{ inData_1_0_address0 sc_out sc_lv 12 signal 19 } 
	{ inData_1_0_ce0 sc_out sc_logic 1 signal 19 } 
	{ inData_1_0_q0 sc_in sc_lv 32 signal 19 } 
	{ inData_1_1_address0 sc_out sc_lv 12 signal 20 } 
	{ inData_1_1_ce0 sc_out sc_logic 1 signal 20 } 
	{ inData_1_1_q0 sc_in sc_lv 32 signal 20 } 
	{ inData_1_2_address0 sc_out sc_lv 12 signal 21 } 
	{ inData_1_2_ce0 sc_out sc_logic 1 signal 21 } 
	{ inData_1_2_q0 sc_in sc_lv 32 signal 21 } 
	{ inData_2_0_address0 sc_out sc_lv 12 signal 22 } 
	{ inData_2_0_ce0 sc_out sc_logic 1 signal 22 } 
	{ inData_2_0_q0 sc_in sc_lv 32 signal 22 } 
	{ inData_2_1_address0 sc_out sc_lv 12 signal 23 } 
	{ inData_2_1_ce0 sc_out sc_logic 1 signal 23 } 
	{ inData_2_1_q0 sc_in sc_lv 32 signal 23 } 
	{ inData_2_2_address0 sc_out sc_lv 12 signal 24 } 
	{ inData_2_2_ce0 sc_out sc_logic 1 signal 24 } 
	{ inData_2_2_q0 sc_in sc_lv 32 signal 24 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "outTileW", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "outTileW", "role": "default" }} , 
 	{ "name": "zext_ln25_3", "direction": "in", "datatype": "sc_lv", "bitwidth":70, "type": "signal", "bundle":{"name": "zext_ln25_3", "role": "default" }} , 
 	{ "name": "mul_ln25", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "mul_ln25", "role": "default" }} , 
 	{ "name": "zext_ln25", "direction": "in", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "zext_ln25", "role": "default" }} , 
 	{ "name": "weight_0_0_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "weight_0_0", "role": "address0" }} , 
 	{ "name": "weight_0_0_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "weight_0_0", "role": "ce0" }} , 
 	{ "name": "weight_0_0_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "weight_0_0", "role": "q0" }} , 
 	{ "name": "weight_0_1_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "weight_0_1", "role": "address0" }} , 
 	{ "name": "weight_0_1_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "weight_0_1", "role": "ce0" }} , 
 	{ "name": "weight_0_1_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "weight_0_1", "role": "q0" }} , 
 	{ "name": "weight_0_2_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "weight_0_2", "role": "address0" }} , 
 	{ "name": "weight_0_2_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "weight_0_2", "role": "ce0" }} , 
 	{ "name": "weight_0_2_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "weight_0_2", "role": "q0" }} , 
 	{ "name": "weight_1_0_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "weight_1_0", "role": "address0" }} , 
 	{ "name": "weight_1_0_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "weight_1_0", "role": "ce0" }} , 
 	{ "name": "weight_1_0_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "weight_1_0", "role": "q0" }} , 
 	{ "name": "weight_1_1_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "weight_1_1", "role": "address0" }} , 
 	{ "name": "weight_1_1_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "weight_1_1", "role": "ce0" }} , 
 	{ "name": "weight_1_1_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "weight_1_1", "role": "q0" }} , 
 	{ "name": "weight_1_2_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "weight_1_2", "role": "address0" }} , 
 	{ "name": "weight_1_2_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "weight_1_2", "role": "ce0" }} , 
 	{ "name": "weight_1_2_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "weight_1_2", "role": "q0" }} , 
 	{ "name": "weight_2_0_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "weight_2_0", "role": "address0" }} , 
 	{ "name": "weight_2_0_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "weight_2_0", "role": "ce0" }} , 
 	{ "name": "weight_2_0_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "weight_2_0", "role": "q0" }} , 
 	{ "name": "weight_2_1_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "weight_2_1", "role": "address0" }} , 
 	{ "name": "weight_2_1_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "weight_2_1", "role": "ce0" }} , 
 	{ "name": "weight_2_1_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "weight_2_1", "role": "q0" }} , 
 	{ "name": "weight_2_2_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "weight_2_2", "role": "address0" }} , 
 	{ "name": "weight_2_2_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "weight_2_2", "role": "ce0" }} , 
 	{ "name": "weight_2_2_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "weight_2_2", "role": "q0" }} , 
 	{ "name": "sext_ln25", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "sext_ln25", "role": "default" }} , 
 	{ "name": "zext_ln34", "direction": "in", "datatype": "sc_lv", "bitwidth":9, "type": "signal", "bundle":{"name": "zext_ln34", "role": "default" }} , 
 	{ "name": "outData_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "outData", "role": "address0" }} , 
 	{ "name": "outData_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "outData", "role": "ce0" }} , 
 	{ "name": "outData_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "outData", "role": "we0" }} , 
 	{ "name": "outData_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "outData", "role": "d0" }} , 
 	{ "name": "outData_address1", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "outData", "role": "address1" }} , 
 	{ "name": "outData_ce1", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "outData", "role": "ce1" }} , 
 	{ "name": "outData_q1", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "outData", "role": "q1" }} , 
 	{ "name": "inData_0_0_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "inData_0_0", "role": "address0" }} , 
 	{ "name": "inData_0_0_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inData_0_0", "role": "ce0" }} , 
 	{ "name": "inData_0_0_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "inData_0_0", "role": "q0" }} , 
 	{ "name": "inData_0_1_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "inData_0_1", "role": "address0" }} , 
 	{ "name": "inData_0_1_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inData_0_1", "role": "ce0" }} , 
 	{ "name": "inData_0_1_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "inData_0_1", "role": "q0" }} , 
 	{ "name": "inData_0_2_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "inData_0_2", "role": "address0" }} , 
 	{ "name": "inData_0_2_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inData_0_2", "role": "ce0" }} , 
 	{ "name": "inData_0_2_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "inData_0_2", "role": "q0" }} , 
 	{ "name": "inData_1_0_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "inData_1_0", "role": "address0" }} , 
 	{ "name": "inData_1_0_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inData_1_0", "role": "ce0" }} , 
 	{ "name": "inData_1_0_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "inData_1_0", "role": "q0" }} , 
 	{ "name": "inData_1_1_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "inData_1_1", "role": "address0" }} , 
 	{ "name": "inData_1_1_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inData_1_1", "role": "ce0" }} , 
 	{ "name": "inData_1_1_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "inData_1_1", "role": "q0" }} , 
 	{ "name": "inData_1_2_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "inData_1_2", "role": "address0" }} , 
 	{ "name": "inData_1_2_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inData_1_2", "role": "ce0" }} , 
 	{ "name": "inData_1_2_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "inData_1_2", "role": "q0" }} , 
 	{ "name": "inData_2_0_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "inData_2_0", "role": "address0" }} , 
 	{ "name": "inData_2_0_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inData_2_0", "role": "ce0" }} , 
 	{ "name": "inData_2_0_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "inData_2_0", "role": "q0" }} , 
 	{ "name": "inData_2_1_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "inData_2_1", "role": "address0" }} , 
 	{ "name": "inData_2_1_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inData_2_1", "role": "ce0" }} , 
 	{ "name": "inData_2_1_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "inData_2_1", "role": "q0" }} , 
 	{ "name": "inData_2_2_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "inData_2_2", "role": "address0" }} , 
 	{ "name": "inData_2_2_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "inData_2_2", "role": "ce0" }} , 
 	{ "name": "inData_2_2_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "inData_2_2", "role": "q0" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48"],
		"CDFG" : "conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "2", "EstimateLatencyMax" : "3251",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "outTileW", "Type" : "None", "Direction" : "I"},
			{"Name" : "zext_ln25_3", "Type" : "None", "Direction" : "I"},
			{"Name" : "mul_ln25", "Type" : "None", "Direction" : "I"},
			{"Name" : "zext_ln25", "Type" : "None", "Direction" : "I"},
			{"Name" : "weight_0_0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "weight_0_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "weight_0_2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "weight_1_0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "weight_1_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "weight_1_2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "weight_2_0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "weight_2_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "weight_2_2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "sext_ln25", "Type" : "None", "Direction" : "I"},
			{"Name" : "zext_ln34", "Type" : "None", "Direction" : "I"},
			{"Name" : "outData", "Type" : "Memory", "Direction" : "IO"},
			{"Name" : "inData_0_0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inData_0_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inData_0_2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inData_1_0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inData_1_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inData_1_2", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inData_2_0", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inData_2_1", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "inData_2_2", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter115", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter115", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_4_full_dsp_1_U96", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_4_full_dsp_1_U97", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_4_full_dsp_1_U98", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_4_full_dsp_1_U99", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_4_full_dsp_1_U100", "Parent" : "0"},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_4_full_dsp_1_U101", "Parent" : "0"},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_4_full_dsp_1_U102", "Parent" : "0"},
	{"ID" : "8", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_4_full_dsp_1_U103", "Parent" : "0"},
	{"ID" : "9", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_4_full_dsp_1_U104", "Parent" : "0"},
	{"ID" : "10", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fadd_32ns_32ns_32_4_full_dsp_1_U105", "Parent" : "0"},
	{"ID" : "11", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_3_max_dsp_1_U106", "Parent" : "0"},
	{"ID" : "12", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_3_max_dsp_1_U107", "Parent" : "0"},
	{"ID" : "13", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_3_max_dsp_1_U108", "Parent" : "0"},
	{"ID" : "14", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_3_max_dsp_1_U109", "Parent" : "0"},
	{"ID" : "15", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_3_max_dsp_1_U110", "Parent" : "0"},
	{"ID" : "16", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_3_max_dsp_1_U111", "Parent" : "0"},
	{"ID" : "17", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_3_max_dsp_1_U112", "Parent" : "0"},
	{"ID" : "18", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_3_max_dsp_1_U113", "Parent" : "0"},
	{"ID" : "19", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fmul_32ns_32ns_32_3_max_dsp_1_U114", "Parent" : "0"},
	{"ID" : "20", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_64ns_66ns_129_1_1_U115", "Parent" : "0"},
	{"ID" : "21", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_64ns_66ns_129_1_1_U116", "Parent" : "0"},
	{"ID" : "22", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_64ns_66ns_129_1_1_U117", "Parent" : "0"},
	{"ID" : "23", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_64ns_66ns_129_1_1_U118", "Parent" : "0"},
	{"ID" : "24", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_64ns_66ns_129_1_1_U119", "Parent" : "0"},
	{"ID" : "25", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_64ns_66ns_129_1_1_U120", "Parent" : "0"},
	{"ID" : "26", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_7_2_32_1_1_U121", "Parent" : "0"},
	{"ID" : "27", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_7_2_32_1_1_U122", "Parent" : "0"},
	{"ID" : "28", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_7_2_32_1_1_U123", "Parent" : "0"},
	{"ID" : "29", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_7_2_32_1_1_U124", "Parent" : "0"},
	{"ID" : "30", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_7_2_32_1_1_U125", "Parent" : "0"},
	{"ID" : "31", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_7_2_32_1_1_U126", "Parent" : "0"},
	{"ID" : "32", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_7_2_32_1_1_U127", "Parent" : "0"},
	{"ID" : "33", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_7_2_32_1_1_U128", "Parent" : "0"},
	{"ID" : "34", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_7_2_32_1_1_U129", "Parent" : "0"},
	{"ID" : "35", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_3ns_32s_35_1_1_U130", "Parent" : "0"},
	{"ID" : "36", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.urem_64s_3ns_2_68_1_U131", "Parent" : "0"},
	{"ID" : "37", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_3ns_32s_35_1_1_U132", "Parent" : "0"},
	{"ID" : "38", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.urem_64s_3ns_2_68_1_U133", "Parent" : "0"},
	{"ID" : "39", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_7_2_32_1_1_U134", "Parent" : "0"},
	{"ID" : "40", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_7_2_32_1_1_U135", "Parent" : "0"},
	{"ID" : "41", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_7_2_32_1_1_U136", "Parent" : "0"},
	{"ID" : "42", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_7_2_32_1_1_U137", "Parent" : "0"},
	{"ID" : "43", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_7_2_32_1_1_U138", "Parent" : "0"},
	{"ID" : "44", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_7_2_32_1_1_U139", "Parent" : "0"},
	{"ID" : "45", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_7_2_32_1_1_U140", "Parent" : "0"},
	{"ID" : "46", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_7_2_32_1_1_U141", "Parent" : "0"},
	{"ID" : "47", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_7_2_32_1_1_U142", "Parent" : "0"},
	{"ID" : "48", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.flow_control_loop_pipe_sequential_init_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4 {
		outTileW {Type I LastRead 0 FirstWrite -1}
		zext_ln25_3 {Type I LastRead 0 FirstWrite -1}
		mul_ln25 {Type I LastRead 0 FirstWrite -1}
		zext_ln25 {Type I LastRead 0 FirstWrite -1}
		weight_0_0 {Type I LastRead 1 FirstWrite -1}
		weight_0_1 {Type I LastRead 1 FirstWrite -1}
		weight_0_2 {Type I LastRead 1 FirstWrite -1}
		weight_1_0 {Type I LastRead 1 FirstWrite -1}
		weight_1_1 {Type I LastRead 1 FirstWrite -1}
		weight_1_2 {Type I LastRead 1 FirstWrite -1}
		weight_2_0 {Type I LastRead 1 FirstWrite -1}
		weight_2_1 {Type I LastRead 1 FirstWrite -1}
		weight_2_2 {Type I LastRead 1 FirstWrite -1}
		sext_ln25 {Type I LastRead 0 FirstWrite -1}
		zext_ln34 {Type I LastRead 0 FirstWrite -1}
		outData {Type IO LastRead 1 FirstWrite 115}
		inData_0_0 {Type I LastRead 71 FirstWrite -1}
		inData_0_1 {Type I LastRead 71 FirstWrite -1}
		inData_0_2 {Type I LastRead 71 FirstWrite -1}
		inData_1_0 {Type I LastRead 71 FirstWrite -1}
		inData_1_1 {Type I LastRead 71 FirstWrite -1}
		inData_1_2 {Type I LastRead 71 FirstWrite -1}
		inData_2_0 {Type I LastRead 71 FirstWrite -1}
		inData_2_1 {Type I LastRead 71 FirstWrite -1}
		inData_2_2 {Type I LastRead 71 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "2", "Max" : "3251"}
	, {"Name" : "Interval", "Min" : "2", "Max" : "3251"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	outTileW { ap_none {  { outTileW in_data 0 32 } } }
	zext_ln25_3 { ap_none {  { zext_ln25_3 in_data 0 70 } } }
	mul_ln25 { ap_none {  { mul_ln25 in_data 0 64 } } }
	zext_ln25 { ap_none {  { zext_ln25 in_data 0 6 } } }
	weight_0_0 { ap_memory {  { weight_0_0_address0 mem_address 1 12 }  { weight_0_0_ce0 mem_ce 1 1 }  { weight_0_0_q0 mem_dout 0 32 } } }
	weight_0_1 { ap_memory {  { weight_0_1_address0 mem_address 1 12 }  { weight_0_1_ce0 mem_ce 1 1 }  { weight_0_1_q0 mem_dout 0 32 } } }
	weight_0_2 { ap_memory {  { weight_0_2_address0 mem_address 1 12 }  { weight_0_2_ce0 mem_ce 1 1 }  { weight_0_2_q0 mem_dout 0 32 } } }
	weight_1_0 { ap_memory {  { weight_1_0_address0 mem_address 1 12 }  { weight_1_0_ce0 mem_ce 1 1 }  { weight_1_0_q0 mem_dout 0 32 } } }
	weight_1_1 { ap_memory {  { weight_1_1_address0 mem_address 1 12 }  { weight_1_1_ce0 mem_ce 1 1 }  { weight_1_1_q0 mem_dout 0 32 } } }
	weight_1_2 { ap_memory {  { weight_1_2_address0 mem_address 1 12 }  { weight_1_2_ce0 mem_ce 1 1 }  { weight_1_2_q0 mem_dout 0 32 } } }
	weight_2_0 { ap_memory {  { weight_2_0_address0 mem_address 1 12 }  { weight_2_0_ce0 mem_ce 1 1 }  { weight_2_0_q0 mem_dout 0 32 } } }
	weight_2_1 { ap_memory {  { weight_2_1_address0 mem_address 1 12 }  { weight_2_1_ce0 mem_ce 1 1 }  { weight_2_1_q0 mem_dout 0 32 } } }
	weight_2_2 { ap_memory {  { weight_2_2_address0 mem_address 1 12 }  { weight_2_2_ce0 mem_ce 1 1 }  { weight_2_2_q0 mem_dout 0 32 } } }
	sext_ln25 { ap_none {  { sext_ln25 in_data 0 32 } } }
	zext_ln34 { ap_none {  { zext_ln34 in_data 0 9 } } }
	outData { ap_memory {  { outData_address0 mem_address 1 12 }  { outData_ce0 mem_ce 1 1 }  { outData_we0 mem_we 1 1 }  { outData_d0 mem_din 1 32 }  { outData_address1 MemPortADDR2 1 12 }  { outData_ce1 MemPortCE2 1 1 }  { outData_q1 MemPortDOUT2 0 32 } } }
	inData_0_0 { ap_memory {  { inData_0_0_address0 mem_address 1 12 }  { inData_0_0_ce0 mem_ce 1 1 }  { inData_0_0_q0 mem_dout 0 32 } } }
	inData_0_1 { ap_memory {  { inData_0_1_address0 mem_address 1 12 }  { inData_0_1_ce0 mem_ce 1 1 }  { inData_0_1_q0 mem_dout 0 32 } } }
	inData_0_2 { ap_memory {  { inData_0_2_address0 mem_address 1 12 }  { inData_0_2_ce0 mem_ce 1 1 }  { inData_0_2_q0 mem_dout 0 32 } } }
	inData_1_0 { ap_memory {  { inData_1_0_address0 mem_address 1 12 }  { inData_1_0_ce0 mem_ce 1 1 }  { inData_1_0_q0 mem_dout 0 32 } } }
	inData_1_1 { ap_memory {  { inData_1_1_address0 mem_address 1 12 }  { inData_1_1_ce0 mem_ce 1 1 }  { inData_1_1_q0 mem_dout 0 32 } } }
	inData_1_2 { ap_memory {  { inData_1_2_address0 mem_address 1 12 }  { inData_1_2_ce0 mem_ce 1 1 }  { inData_1_2_q0 mem_dout 0 32 } } }
	inData_2_0 { ap_memory {  { inData_2_0_address0 mem_address 1 12 }  { inData_2_0_ce0 mem_ce 1 1 }  { inData_2_0_q0 mem_dout 0 32 } } }
	inData_2_1 { ap_memory {  { inData_2_1_address0 mem_address 1 12 }  { inData_2_1_ce0 mem_ce 1 1 }  { inData_2_1_q0 mem_dout 0 32 } } }
	inData_2_2 { ap_memory {  { inData_2_2_address0 mem_address 1 12 }  { inData_2_2_ce0 mem_ce 1 1 }  { inData_2_2_q0 mem_dout 0 32 } } }
}
