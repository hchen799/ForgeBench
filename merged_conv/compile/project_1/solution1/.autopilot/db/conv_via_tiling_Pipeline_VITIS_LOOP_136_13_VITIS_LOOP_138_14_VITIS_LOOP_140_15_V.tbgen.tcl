set moduleName conv_via_tiling_Pipeline_VITIS_LOOP_136_13_VITIS_LOOP_138_14_VITIS_LOOP_140_15_V
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
set C_modelName {conv_via_tiling_Pipeline_VITIS_LOOP_136_13_VITIS_LOOP_138_14_VITIS_LOOP_140_15_V}
set C_modelType { void 0 }
set ap_memory_interface_dict [dict create]
dict set ap_memory_interface_dict localW { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict localW_1 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict localW_2 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict localW_3 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict localW_4 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict localW_5 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict localW_6 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict localW_7 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict localW_8 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
set C_modelArgList {
	{ localW float 32 regular {array 4096 { 0 3 } 0 1 }  }
	{ localW_1 float 32 regular {array 4096 { 0 3 } 0 1 }  }
	{ localW_2 float 32 regular {array 4096 { 0 3 } 0 1 }  }
	{ localW_3 float 32 regular {array 4096 { 0 3 } 0 1 }  }
	{ localW_4 float 32 regular {array 4096 { 0 3 } 0 1 }  }
	{ localW_5 float 32 regular {array 4096 { 0 3 } 0 1 }  }
	{ localW_6 float 32 regular {array 4096 { 0 3 } 0 1 }  }
	{ localW_7 float 32 regular {array 4096 { 0 3 } 0 1 }  }
	{ localW_8 float 32 regular {array 4096 { 0 3 } 0 1 }  }
}
set hasAXIMCache 0
set hasAXIML2Cache 0
set AXIMCacheInstDict [dict create]
set C_modelArgMapList {[ 
	{ "Name" : "localW", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "localW_1", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "localW_2", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "localW_3", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "localW_4", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "localW_5", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "localW_6", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "localW_7", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "localW_8", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 42
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ localW_address0 sc_out sc_lv 12 signal 0 } 
	{ localW_ce0 sc_out sc_logic 1 signal 0 } 
	{ localW_we0 sc_out sc_logic 1 signal 0 } 
	{ localW_d0 sc_out sc_lv 32 signal 0 } 
	{ localW_1_address0 sc_out sc_lv 12 signal 1 } 
	{ localW_1_ce0 sc_out sc_logic 1 signal 1 } 
	{ localW_1_we0 sc_out sc_logic 1 signal 1 } 
	{ localW_1_d0 sc_out sc_lv 32 signal 1 } 
	{ localW_2_address0 sc_out sc_lv 12 signal 2 } 
	{ localW_2_ce0 sc_out sc_logic 1 signal 2 } 
	{ localW_2_we0 sc_out sc_logic 1 signal 2 } 
	{ localW_2_d0 sc_out sc_lv 32 signal 2 } 
	{ localW_3_address0 sc_out sc_lv 12 signal 3 } 
	{ localW_3_ce0 sc_out sc_logic 1 signal 3 } 
	{ localW_3_we0 sc_out sc_logic 1 signal 3 } 
	{ localW_3_d0 sc_out sc_lv 32 signal 3 } 
	{ localW_4_address0 sc_out sc_lv 12 signal 4 } 
	{ localW_4_ce0 sc_out sc_logic 1 signal 4 } 
	{ localW_4_we0 sc_out sc_logic 1 signal 4 } 
	{ localW_4_d0 sc_out sc_lv 32 signal 4 } 
	{ localW_5_address0 sc_out sc_lv 12 signal 5 } 
	{ localW_5_ce0 sc_out sc_logic 1 signal 5 } 
	{ localW_5_we0 sc_out sc_logic 1 signal 5 } 
	{ localW_5_d0 sc_out sc_lv 32 signal 5 } 
	{ localW_6_address0 sc_out sc_lv 12 signal 6 } 
	{ localW_6_ce0 sc_out sc_logic 1 signal 6 } 
	{ localW_6_we0 sc_out sc_logic 1 signal 6 } 
	{ localW_6_d0 sc_out sc_lv 32 signal 6 } 
	{ localW_7_address0 sc_out sc_lv 12 signal 7 } 
	{ localW_7_ce0 sc_out sc_logic 1 signal 7 } 
	{ localW_7_we0 sc_out sc_logic 1 signal 7 } 
	{ localW_7_d0 sc_out sc_lv 32 signal 7 } 
	{ localW_8_address0 sc_out sc_lv 12 signal 8 } 
	{ localW_8_ce0 sc_out sc_logic 1 signal 8 } 
	{ localW_8_we0 sc_out sc_logic 1 signal 8 } 
	{ localW_8_d0 sc_out sc_lv 32 signal 8 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "localW_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "localW", "role": "address0" }} , 
 	{ "name": "localW_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localW", "role": "ce0" }} , 
 	{ "name": "localW_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localW", "role": "we0" }} , 
 	{ "name": "localW_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "localW", "role": "d0" }} , 
 	{ "name": "localW_1_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "localW_1", "role": "address0" }} , 
 	{ "name": "localW_1_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localW_1", "role": "ce0" }} , 
 	{ "name": "localW_1_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localW_1", "role": "we0" }} , 
 	{ "name": "localW_1_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "localW_1", "role": "d0" }} , 
 	{ "name": "localW_2_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "localW_2", "role": "address0" }} , 
 	{ "name": "localW_2_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localW_2", "role": "ce0" }} , 
 	{ "name": "localW_2_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localW_2", "role": "we0" }} , 
 	{ "name": "localW_2_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "localW_2", "role": "d0" }} , 
 	{ "name": "localW_3_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "localW_3", "role": "address0" }} , 
 	{ "name": "localW_3_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localW_3", "role": "ce0" }} , 
 	{ "name": "localW_3_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localW_3", "role": "we0" }} , 
 	{ "name": "localW_3_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "localW_3", "role": "d0" }} , 
 	{ "name": "localW_4_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "localW_4", "role": "address0" }} , 
 	{ "name": "localW_4_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localW_4", "role": "ce0" }} , 
 	{ "name": "localW_4_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localW_4", "role": "we0" }} , 
 	{ "name": "localW_4_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "localW_4", "role": "d0" }} , 
 	{ "name": "localW_5_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "localW_5", "role": "address0" }} , 
 	{ "name": "localW_5_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localW_5", "role": "ce0" }} , 
 	{ "name": "localW_5_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localW_5", "role": "we0" }} , 
 	{ "name": "localW_5_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "localW_5", "role": "d0" }} , 
 	{ "name": "localW_6_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "localW_6", "role": "address0" }} , 
 	{ "name": "localW_6_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localW_6", "role": "ce0" }} , 
 	{ "name": "localW_6_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localW_6", "role": "we0" }} , 
 	{ "name": "localW_6_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "localW_6", "role": "d0" }} , 
 	{ "name": "localW_7_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "localW_7", "role": "address0" }} , 
 	{ "name": "localW_7_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localW_7", "role": "ce0" }} , 
 	{ "name": "localW_7_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localW_7", "role": "we0" }} , 
 	{ "name": "localW_7_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "localW_7", "role": "d0" }} , 
 	{ "name": "localW_8_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "localW_8", "role": "address0" }} , 
 	{ "name": "localW_8_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localW_8", "role": "ce0" }} , 
 	{ "name": "localW_8_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localW_8", "role": "we0" }} , 
 	{ "name": "localW_8_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "localW_8", "role": "d0" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1"],
		"CDFG" : "conv_via_tiling_Pipeline_VITIS_LOOP_136_13_VITIS_LOOP_138_14_VITIS_LOOP_140_15_V",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "36866", "EstimateLatencyMax" : "36866",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "localW", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localW_1", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localW_2", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localW_3", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localW_4", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localW_5", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localW_6", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localW_7", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localW_8", "Type" : "Memory", "Direction" : "O"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_136_13_VITIS_LOOP_138_14_VITIS_LOOP_140_15_VITIS_LOOP_142_16", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter1", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter1", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.flow_control_loop_pipe_sequential_init_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	conv_via_tiling_Pipeline_VITIS_LOOP_136_13_VITIS_LOOP_138_14_VITIS_LOOP_140_15_V {
		localW {Type O LastRead -1 FirstWrite 1}
		localW_1 {Type O LastRead -1 FirstWrite 1}
		localW_2 {Type O LastRead -1 FirstWrite 1}
		localW_3 {Type O LastRead -1 FirstWrite 1}
		localW_4 {Type O LastRead -1 FirstWrite 1}
		localW_5 {Type O LastRead -1 FirstWrite 1}
		localW_6 {Type O LastRead -1 FirstWrite 1}
		localW_7 {Type O LastRead -1 FirstWrite 1}
		localW_8 {Type O LastRead -1 FirstWrite 1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "36866", "Max" : "36866"}
	, {"Name" : "Interval", "Min" : "36866", "Max" : "36866"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	localW { ap_memory {  { localW_address0 mem_address 1 12 }  { localW_ce0 mem_ce 1 1 }  { localW_we0 mem_we 1 1 }  { localW_d0 mem_din 1 32 } } }
	localW_1 { ap_memory {  { localW_1_address0 mem_address 1 12 }  { localW_1_ce0 mem_ce 1 1 }  { localW_1_we0 mem_we 1 1 }  { localW_1_d0 mem_din 1 32 } } }
	localW_2 { ap_memory {  { localW_2_address0 mem_address 1 12 }  { localW_2_ce0 mem_ce 1 1 }  { localW_2_we0 mem_we 1 1 }  { localW_2_d0 mem_din 1 32 } } }
	localW_3 { ap_memory {  { localW_3_address0 mem_address 1 12 }  { localW_3_ce0 mem_ce 1 1 }  { localW_3_we0 mem_we 1 1 }  { localW_3_d0 mem_din 1 32 } } }
	localW_4 { ap_memory {  { localW_4_address0 mem_address 1 12 }  { localW_4_ce0 mem_ce 1 1 }  { localW_4_we0 mem_we 1 1 }  { localW_4_d0 mem_din 1 32 } } }
	localW_5 { ap_memory {  { localW_5_address0 mem_address 1 12 }  { localW_5_ce0 mem_ce 1 1 }  { localW_5_we0 mem_we 1 1 }  { localW_5_d0 mem_din 1 32 } } }
	localW_6 { ap_memory {  { localW_6_address0 mem_address 1 12 }  { localW_6_ce0 mem_ce 1 1 }  { localW_6_we0 mem_we 1 1 }  { localW_6_d0 mem_din 1 32 } } }
	localW_7 { ap_memory {  { localW_7_address0 mem_address 1 12 }  { localW_7_ce0 mem_ce 1 1 }  { localW_7_we0 mem_we 1 1 }  { localW_7_d0 mem_din 1 32 } } }
	localW_8 { ap_memory {  { localW_8_address0 mem_address 1 12 }  { localW_8_ce0 mem_ce 1 1 }  { localW_8_we0 mem_we 1 1 }  { localW_8_d0 mem_din 1 32 } } }
}
