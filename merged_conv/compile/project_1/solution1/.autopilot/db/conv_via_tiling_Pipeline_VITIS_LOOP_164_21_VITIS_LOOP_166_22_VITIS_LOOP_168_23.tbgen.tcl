set moduleName conv_via_tiling_Pipeline_VITIS_LOOP_164_21_VITIS_LOOP_166_22_VITIS_LOOP_168_23
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
set C_modelName {conv_via_tiling_Pipeline_VITIS_LOOP_164_21_VITIS_LOOP_166_22_VITIS_LOOP_168_23}
set C_modelType { void 0 }
set ap_memory_interface_dict [dict create]
dict set ap_memory_interface_dict localIn { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict localIn_1 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict localIn_2 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict localIn_3 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict localIn_4 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict localIn_5 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict localIn_6 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict localIn_7 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
dict set ap_memory_interface_dict localIn_8 { MEM_WIDTH 32 MEM_SIZE 16384 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 0 }
set C_modelArgList {
	{ localIn float 32 regular {array 4096 { 0 3 } 0 1 }  }
	{ localIn_1 float 32 regular {array 4096 { 0 3 } 0 1 }  }
	{ localIn_2 float 32 regular {array 4096 { 0 3 } 0 1 }  }
	{ localIn_3 float 32 regular {array 4096 { 0 3 } 0 1 }  }
	{ localIn_4 float 32 regular {array 4096 { 0 3 } 0 1 }  }
	{ localIn_5 float 32 regular {array 4096 { 0 3 } 0 1 }  }
	{ localIn_6 float 32 regular {array 4096 { 0 3 } 0 1 }  }
	{ localIn_7 float 32 regular {array 4096 { 0 3 } 0 1 }  }
	{ localIn_8 float 32 regular {array 4096 { 0 3 } 0 1 }  }
}
set hasAXIMCache 0
set hasAXIML2Cache 0
set AXIMCacheInstDict [dict create]
set C_modelArgMapList {[ 
	{ "Name" : "localIn", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "localIn_1", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "localIn_2", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "localIn_3", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "localIn_4", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "localIn_5", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "localIn_6", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "localIn_7", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "localIn_8", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} ]}
# RTL Port declarations: 
set portNum 42
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ localIn_address0 sc_out sc_lv 12 signal 0 } 
	{ localIn_ce0 sc_out sc_logic 1 signal 0 } 
	{ localIn_we0 sc_out sc_logic 1 signal 0 } 
	{ localIn_d0 sc_out sc_lv 32 signal 0 } 
	{ localIn_1_address0 sc_out sc_lv 12 signal 1 } 
	{ localIn_1_ce0 sc_out sc_logic 1 signal 1 } 
	{ localIn_1_we0 sc_out sc_logic 1 signal 1 } 
	{ localIn_1_d0 sc_out sc_lv 32 signal 1 } 
	{ localIn_2_address0 sc_out sc_lv 12 signal 2 } 
	{ localIn_2_ce0 sc_out sc_logic 1 signal 2 } 
	{ localIn_2_we0 sc_out sc_logic 1 signal 2 } 
	{ localIn_2_d0 sc_out sc_lv 32 signal 2 } 
	{ localIn_3_address0 sc_out sc_lv 12 signal 3 } 
	{ localIn_3_ce0 sc_out sc_logic 1 signal 3 } 
	{ localIn_3_we0 sc_out sc_logic 1 signal 3 } 
	{ localIn_3_d0 sc_out sc_lv 32 signal 3 } 
	{ localIn_4_address0 sc_out sc_lv 12 signal 4 } 
	{ localIn_4_ce0 sc_out sc_logic 1 signal 4 } 
	{ localIn_4_we0 sc_out sc_logic 1 signal 4 } 
	{ localIn_4_d0 sc_out sc_lv 32 signal 4 } 
	{ localIn_5_address0 sc_out sc_lv 12 signal 5 } 
	{ localIn_5_ce0 sc_out sc_logic 1 signal 5 } 
	{ localIn_5_we0 sc_out sc_logic 1 signal 5 } 
	{ localIn_5_d0 sc_out sc_lv 32 signal 5 } 
	{ localIn_6_address0 sc_out sc_lv 12 signal 6 } 
	{ localIn_6_ce0 sc_out sc_logic 1 signal 6 } 
	{ localIn_6_we0 sc_out sc_logic 1 signal 6 } 
	{ localIn_6_d0 sc_out sc_lv 32 signal 6 } 
	{ localIn_7_address0 sc_out sc_lv 12 signal 7 } 
	{ localIn_7_ce0 sc_out sc_logic 1 signal 7 } 
	{ localIn_7_we0 sc_out sc_logic 1 signal 7 } 
	{ localIn_7_d0 sc_out sc_lv 32 signal 7 } 
	{ localIn_8_address0 sc_out sc_lv 12 signal 8 } 
	{ localIn_8_ce0 sc_out sc_logic 1 signal 8 } 
	{ localIn_8_we0 sc_out sc_logic 1 signal 8 } 
	{ localIn_8_d0 sc_out sc_lv 32 signal 8 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "localIn_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "localIn", "role": "address0" }} , 
 	{ "name": "localIn_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localIn", "role": "ce0" }} , 
 	{ "name": "localIn_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localIn", "role": "we0" }} , 
 	{ "name": "localIn_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "localIn", "role": "d0" }} , 
 	{ "name": "localIn_1_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "localIn_1", "role": "address0" }} , 
 	{ "name": "localIn_1_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localIn_1", "role": "ce0" }} , 
 	{ "name": "localIn_1_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localIn_1", "role": "we0" }} , 
 	{ "name": "localIn_1_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "localIn_1", "role": "d0" }} , 
 	{ "name": "localIn_2_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "localIn_2", "role": "address0" }} , 
 	{ "name": "localIn_2_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localIn_2", "role": "ce0" }} , 
 	{ "name": "localIn_2_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localIn_2", "role": "we0" }} , 
 	{ "name": "localIn_2_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "localIn_2", "role": "d0" }} , 
 	{ "name": "localIn_3_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "localIn_3", "role": "address0" }} , 
 	{ "name": "localIn_3_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localIn_3", "role": "ce0" }} , 
 	{ "name": "localIn_3_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localIn_3", "role": "we0" }} , 
 	{ "name": "localIn_3_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "localIn_3", "role": "d0" }} , 
 	{ "name": "localIn_4_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "localIn_4", "role": "address0" }} , 
 	{ "name": "localIn_4_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localIn_4", "role": "ce0" }} , 
 	{ "name": "localIn_4_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localIn_4", "role": "we0" }} , 
 	{ "name": "localIn_4_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "localIn_4", "role": "d0" }} , 
 	{ "name": "localIn_5_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "localIn_5", "role": "address0" }} , 
 	{ "name": "localIn_5_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localIn_5", "role": "ce0" }} , 
 	{ "name": "localIn_5_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localIn_5", "role": "we0" }} , 
 	{ "name": "localIn_5_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "localIn_5", "role": "d0" }} , 
 	{ "name": "localIn_6_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "localIn_6", "role": "address0" }} , 
 	{ "name": "localIn_6_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localIn_6", "role": "ce0" }} , 
 	{ "name": "localIn_6_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localIn_6", "role": "we0" }} , 
 	{ "name": "localIn_6_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "localIn_6", "role": "d0" }} , 
 	{ "name": "localIn_7_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "localIn_7", "role": "address0" }} , 
 	{ "name": "localIn_7_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localIn_7", "role": "ce0" }} , 
 	{ "name": "localIn_7_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localIn_7", "role": "we0" }} , 
 	{ "name": "localIn_7_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "localIn_7", "role": "d0" }} , 
 	{ "name": "localIn_8_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "localIn_8", "role": "address0" }} , 
 	{ "name": "localIn_8_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localIn_8", "role": "ce0" }} , 
 	{ "name": "localIn_8_we0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localIn_8", "role": "we0" }} , 
 	{ "name": "localIn_8_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "localIn_8", "role": "d0" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5"],
		"CDFG" : "conv_via_tiling_Pipeline_VITIS_LOOP_164_21_VITIS_LOOP_166_22_VITIS_LOOP_168_23",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "33865", "EstimateLatencyMax" : "33865",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "localIn", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localIn_1", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localIn_2", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localIn_3", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localIn_4", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localIn_5", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localIn_6", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localIn_7", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localIn_8", "Type" : "Memory", "Direction" : "O"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_164_21_VITIS_LOOP_166_22_VITIS_LOOP_168_23", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter9", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter9", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_5ns_7ns_11_1_1_U53", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.urem_5ns_3ns_2_9_1_U54", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_5ns_7ns_11_1_1_U55", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.urem_5ns_3ns_2_9_1_U56", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.flow_control_loop_pipe_sequential_init_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	conv_via_tiling_Pipeline_VITIS_LOOP_164_21_VITIS_LOOP_166_22_VITIS_LOOP_168_23 {
		localIn {Type O LastRead -1 FirstWrite 9}
		localIn_1 {Type O LastRead -1 FirstWrite 9}
		localIn_2 {Type O LastRead -1 FirstWrite 9}
		localIn_3 {Type O LastRead -1 FirstWrite 9}
		localIn_4 {Type O LastRead -1 FirstWrite 9}
		localIn_5 {Type O LastRead -1 FirstWrite 9}
		localIn_6 {Type O LastRead -1 FirstWrite 9}
		localIn_7 {Type O LastRead -1 FirstWrite 9}
		localIn_8 {Type O LastRead -1 FirstWrite 9}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "33865", "Max" : "33865"}
	, {"Name" : "Interval", "Min" : "33865", "Max" : "33865"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	localIn { ap_memory {  { localIn_address0 mem_address 1 12 }  { localIn_ce0 mem_ce 1 1 }  { localIn_we0 mem_we 1 1 }  { localIn_d0 mem_din 1 32 } } }
	localIn_1 { ap_memory {  { localIn_1_address0 mem_address 1 12 }  { localIn_1_ce0 mem_ce 1 1 }  { localIn_1_we0 mem_we 1 1 }  { localIn_1_d0 mem_din 1 32 } } }
	localIn_2 { ap_memory {  { localIn_2_address0 mem_address 1 12 }  { localIn_2_ce0 mem_ce 1 1 }  { localIn_2_we0 mem_we 1 1 }  { localIn_2_d0 mem_din 1 32 } } }
	localIn_3 { ap_memory {  { localIn_3_address0 mem_address 1 12 }  { localIn_3_ce0 mem_ce 1 1 }  { localIn_3_we0 mem_we 1 1 }  { localIn_3_d0 mem_din 1 32 } } }
	localIn_4 { ap_memory {  { localIn_4_address0 mem_address 1 12 }  { localIn_4_ce0 mem_ce 1 1 }  { localIn_4_we0 mem_we 1 1 }  { localIn_4_d0 mem_din 1 32 } } }
	localIn_5 { ap_memory {  { localIn_5_address0 mem_address 1 12 }  { localIn_5_ce0 mem_ce 1 1 }  { localIn_5_we0 mem_we 1 1 }  { localIn_5_d0 mem_din 1 32 } } }
	localIn_6 { ap_memory {  { localIn_6_address0 mem_address 1 12 }  { localIn_6_ce0 mem_ce 1 1 }  { localIn_6_we0 mem_we 1 1 }  { localIn_6_d0 mem_din 1 32 } } }
	localIn_7 { ap_memory {  { localIn_7_address0 mem_address 1 12 }  { localIn_7_ce0 mem_ce 1 1 }  { localIn_7_we0 mem_we 1 1 }  { localIn_7_d0 mem_din 1 32 } } }
	localIn_8 { ap_memory {  { localIn_8_address0 mem_address 1 12 }  { localIn_8_ce0 mem_ce 1 1 }  { localIn_8_we0 mem_we 1 1 }  { localIn_8_d0 mem_din 1 32 } } }
}
