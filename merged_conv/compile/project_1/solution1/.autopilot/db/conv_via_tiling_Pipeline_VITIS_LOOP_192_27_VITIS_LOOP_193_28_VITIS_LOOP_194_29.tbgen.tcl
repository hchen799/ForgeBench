set moduleName conv_via_tiling_Pipeline_VITIS_LOOP_192_27_VITIS_LOOP_193_28_VITIS_LOOP_194_29
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
set C_modelName {conv_via_tiling_Pipeline_VITIS_LOOP_192_27_VITIS_LOOP_193_28_VITIS_LOOP_194_29}
set C_modelType { void 0 }
set ap_memory_interface_dict [dict create]
dict set ap_memory_interface_dict localOut { MEM_WIDTH 32 MEM_SIZE 12544 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
set C_modelArgList {
	{ mem2 int 32 regular {axi_master 1}  }
	{ zext_ln104 int 32 regular  }
	{ mul_ln90_2 int 31 regular  }
	{ sext_ln75 int 32 regular  }
	{ zext_ln91_2 int 32 regular  }
	{ output_r int 64 regular  }
	{ xor_ln91 int 32 regular  }
	{ mul_ln192 int 96 regular  }
	{ localOut float 32 regular {array 3136 { 1 3 } 1 1 }  }
	{ zext_ln102 int 32 regular  }
	{ mul_ln91_3 int 64 regular  }
	{ tmp_cast_mid1257 int 31 regular  }
	{ select_ln91 int 32 regular  }
}
set hasAXIMCache 0
set hasAXIML2Cache 0
set AXIMCacheInstDict [dict create]
set C_modelArgMapList {[ 
	{ "Name" : "mem2", "interface" : "axi_master", "bitwidth" : 32, "direction" : "WRITEONLY", "bitSlice":[ {"cElement": [{"cName": "output_r","offset": { "type": "dynamic","port_name": "output_r","bundle": "control"},"direction": "READWRITE"}]}]} , 
 	{ "Name" : "zext_ln104", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "mul_ln90_2", "interface" : "wire", "bitwidth" : 31, "direction" : "READONLY"} , 
 	{ "Name" : "sext_ln75", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "zext_ln91_2", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "output_r", "interface" : "wire", "bitwidth" : 64, "direction" : "READONLY"} , 
 	{ "Name" : "xor_ln91", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "mul_ln192", "interface" : "wire", "bitwidth" : 96, "direction" : "READONLY"} , 
 	{ "Name" : "localOut", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "zext_ln102", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "mul_ln91_3", "interface" : "wire", "bitwidth" : 64, "direction" : "READONLY"} , 
 	{ "Name" : "tmp_cast_mid1257", "interface" : "wire", "bitwidth" : 31, "direction" : "READONLY"} , 
 	{ "Name" : "select_ln91", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} ]}
# RTL Port declarations: 
set portNum 66
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ m_axi_mem2_AWVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_mem2_AWREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_mem2_AWADDR sc_out sc_lv 64 signal 0 } 
	{ m_axi_mem2_AWID sc_out sc_lv 1 signal 0 } 
	{ m_axi_mem2_AWLEN sc_out sc_lv 32 signal 0 } 
	{ m_axi_mem2_AWSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_mem2_AWBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_mem2_AWLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_mem2_AWCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_mem2_AWPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_mem2_AWQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_mem2_AWREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_mem2_AWUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_mem2_WVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_mem2_WREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_mem2_WDATA sc_out sc_lv 32 signal 0 } 
	{ m_axi_mem2_WSTRB sc_out sc_lv 4 signal 0 } 
	{ m_axi_mem2_WLAST sc_out sc_logic 1 signal 0 } 
	{ m_axi_mem2_WID sc_out sc_lv 1 signal 0 } 
	{ m_axi_mem2_WUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_mem2_ARVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_mem2_ARREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_mem2_ARADDR sc_out sc_lv 64 signal 0 } 
	{ m_axi_mem2_ARID sc_out sc_lv 1 signal 0 } 
	{ m_axi_mem2_ARLEN sc_out sc_lv 32 signal 0 } 
	{ m_axi_mem2_ARSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_mem2_ARBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_mem2_ARLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_mem2_ARCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_mem2_ARPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_mem2_ARQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_mem2_ARREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_mem2_ARUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_mem2_RVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_mem2_RREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_mem2_RDATA sc_in sc_lv 32 signal 0 } 
	{ m_axi_mem2_RLAST sc_in sc_logic 1 signal 0 } 
	{ m_axi_mem2_RID sc_in sc_lv 1 signal 0 } 
	{ m_axi_mem2_RFIFONUM sc_in sc_lv 9 signal 0 } 
	{ m_axi_mem2_RUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_mem2_RRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_mem2_BVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_mem2_BREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_mem2_BRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_mem2_BID sc_in sc_lv 1 signal 0 } 
	{ m_axi_mem2_BUSER sc_in sc_lv 1 signal 0 } 
	{ zext_ln104 sc_in sc_lv 32 signal 1 } 
	{ mul_ln90_2 sc_in sc_lv 31 signal 2 } 
	{ sext_ln75 sc_in sc_lv 32 signal 3 } 
	{ zext_ln91_2 sc_in sc_lv 32 signal 4 } 
	{ output_r sc_in sc_lv 64 signal 5 } 
	{ xor_ln91 sc_in sc_lv 32 signal 6 } 
	{ mul_ln192 sc_in sc_lv 96 signal 7 } 
	{ localOut_address0 sc_out sc_lv 12 signal 8 } 
	{ localOut_ce0 sc_out sc_logic 1 signal 8 } 
	{ localOut_q0 sc_in sc_lv 32 signal 8 } 
	{ zext_ln102 sc_in sc_lv 32 signal 9 } 
	{ mul_ln91_3 sc_in sc_lv 64 signal 10 } 
	{ tmp_cast_mid1257 sc_in sc_lv 31 signal 11 } 
	{ select_ln91 sc_in sc_lv 32 signal 12 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "m_axi_mem2_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem2", "role": "AWVALID" }} , 
 	{ "name": "m_axi_mem2_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem2", "role": "AWREADY" }} , 
 	{ "name": "m_axi_mem2_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "mem2", "role": "AWADDR" }} , 
 	{ "name": "m_axi_mem2_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem2", "role": "AWID" }} , 
 	{ "name": "m_axi_mem2_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "mem2", "role": "AWLEN" }} , 
 	{ "name": "m_axi_mem2_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "mem2", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_mem2_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "mem2", "role": "AWBURST" }} , 
 	{ "name": "m_axi_mem2_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "mem2", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_mem2_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "mem2", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_mem2_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "mem2", "role": "AWPROT" }} , 
 	{ "name": "m_axi_mem2_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "mem2", "role": "AWQOS" }} , 
 	{ "name": "m_axi_mem2_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "mem2", "role": "AWREGION" }} , 
 	{ "name": "m_axi_mem2_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem2", "role": "AWUSER" }} , 
 	{ "name": "m_axi_mem2_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem2", "role": "WVALID" }} , 
 	{ "name": "m_axi_mem2_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem2", "role": "WREADY" }} , 
 	{ "name": "m_axi_mem2_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "mem2", "role": "WDATA" }} , 
 	{ "name": "m_axi_mem2_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "mem2", "role": "WSTRB" }} , 
 	{ "name": "m_axi_mem2_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem2", "role": "WLAST" }} , 
 	{ "name": "m_axi_mem2_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem2", "role": "WID" }} , 
 	{ "name": "m_axi_mem2_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem2", "role": "WUSER" }} , 
 	{ "name": "m_axi_mem2_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem2", "role": "ARVALID" }} , 
 	{ "name": "m_axi_mem2_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem2", "role": "ARREADY" }} , 
 	{ "name": "m_axi_mem2_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "mem2", "role": "ARADDR" }} , 
 	{ "name": "m_axi_mem2_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem2", "role": "ARID" }} , 
 	{ "name": "m_axi_mem2_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "mem2", "role": "ARLEN" }} , 
 	{ "name": "m_axi_mem2_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "mem2", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_mem2_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "mem2", "role": "ARBURST" }} , 
 	{ "name": "m_axi_mem2_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "mem2", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_mem2_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "mem2", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_mem2_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "mem2", "role": "ARPROT" }} , 
 	{ "name": "m_axi_mem2_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "mem2", "role": "ARQOS" }} , 
 	{ "name": "m_axi_mem2_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "mem2", "role": "ARREGION" }} , 
 	{ "name": "m_axi_mem2_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem2", "role": "ARUSER" }} , 
 	{ "name": "m_axi_mem2_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem2", "role": "RVALID" }} , 
 	{ "name": "m_axi_mem2_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem2", "role": "RREADY" }} , 
 	{ "name": "m_axi_mem2_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "mem2", "role": "RDATA" }} , 
 	{ "name": "m_axi_mem2_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem2", "role": "RLAST" }} , 
 	{ "name": "m_axi_mem2_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem2", "role": "RID" }} , 
 	{ "name": "m_axi_mem2_RFIFONUM", "direction": "in", "datatype": "sc_lv", "bitwidth":9, "type": "signal", "bundle":{"name": "mem2", "role": "RFIFONUM" }} , 
 	{ "name": "m_axi_mem2_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem2", "role": "RUSER" }} , 
 	{ "name": "m_axi_mem2_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "mem2", "role": "RRESP" }} , 
 	{ "name": "m_axi_mem2_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem2", "role": "BVALID" }} , 
 	{ "name": "m_axi_mem2_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem2", "role": "BREADY" }} , 
 	{ "name": "m_axi_mem2_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "mem2", "role": "BRESP" }} , 
 	{ "name": "m_axi_mem2_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem2", "role": "BID" }} , 
 	{ "name": "m_axi_mem2_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem2", "role": "BUSER" }} , 
 	{ "name": "zext_ln104", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "zext_ln104", "role": "default" }} , 
 	{ "name": "mul_ln90_2", "direction": "in", "datatype": "sc_lv", "bitwidth":31, "type": "signal", "bundle":{"name": "mul_ln90_2", "role": "default" }} , 
 	{ "name": "sext_ln75", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "sext_ln75", "role": "default" }} , 
 	{ "name": "zext_ln91_2", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "zext_ln91_2", "role": "default" }} , 
 	{ "name": "output_r", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "output_r", "role": "default" }} , 
 	{ "name": "xor_ln91", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "xor_ln91", "role": "default" }} , 
 	{ "name": "mul_ln192", "direction": "in", "datatype": "sc_lv", "bitwidth":96, "type": "signal", "bundle":{"name": "mul_ln192", "role": "default" }} , 
 	{ "name": "localOut_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":12, "type": "signal", "bundle":{"name": "localOut", "role": "address0" }} , 
 	{ "name": "localOut_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "localOut", "role": "ce0" }} , 
 	{ "name": "localOut_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "localOut", "role": "q0" }} , 
 	{ "name": "zext_ln102", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "zext_ln102", "role": "default" }} , 
 	{ "name": "mul_ln91_3", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "mul_ln91_3", "role": "default" }} , 
 	{ "name": "tmp_cast_mid1257", "direction": "in", "datatype": "sc_lv", "bitwidth":31, "type": "signal", "bundle":{"name": "tmp_cast_mid1257", "role": "default" }} , 
 	{ "name": "select_ln91", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "select_ln91", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2"],
		"CDFG" : "conv_via_tiling_Pipeline_VITIS_LOOP_192_27_VITIS_LOOP_193_28_VITIS_LOOP_194_29",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "-1", "EstimateLatencyMax" : "-1",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "mem2", "Type" : "MAXI", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "mem2_blk_n_B", "Type" : "RtlSignal"},
					{"Name" : "mem2_blk_n_W", "Type" : "RtlSignal"},
					{"Name" : "mem2_blk_n_AW", "Type" : "RtlSignal"}]},
			{"Name" : "zext_ln104", "Type" : "None", "Direction" : "I"},
			{"Name" : "mul_ln90_2", "Type" : "None", "Direction" : "I"},
			{"Name" : "sext_ln75", "Type" : "None", "Direction" : "I"},
			{"Name" : "zext_ln91_2", "Type" : "None", "Direction" : "I"},
			{"Name" : "output_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "xor_ln91", "Type" : "None", "Direction" : "I"},
			{"Name" : "mul_ln192", "Type" : "None", "Direction" : "I"},
			{"Name" : "localOut", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "zext_ln102", "Type" : "None", "Direction" : "I"},
			{"Name" : "mul_ln91_3", "Type" : "None", "Direction" : "I"},
			{"Name" : "tmp_cast_mid1257", "Type" : "None", "Direction" : "I"},
			{"Name" : "select_ln91", "Type" : "None", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_192_27_VITIS_LOOP_193_28_VITIS_LOOP_194_29", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter9", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter9", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_31ns_32s_62_1_1_U200", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.flow_control_loop_pipe_sequential_init_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	conv_via_tiling_Pipeline_VITIS_LOOP_192_27_VITIS_LOOP_193_28_VITIS_LOOP_194_29 {
		mem2 {Type O LastRead 5 FirstWrite 4}
		zext_ln104 {Type I LastRead 0 FirstWrite -1}
		mul_ln90_2 {Type I LastRead 0 FirstWrite -1}
		sext_ln75 {Type I LastRead 0 FirstWrite -1}
		zext_ln91_2 {Type I LastRead 0 FirstWrite -1}
		output_r {Type I LastRead 0 FirstWrite -1}
		xor_ln91 {Type I LastRead 0 FirstWrite -1}
		mul_ln192 {Type I LastRead 0 FirstWrite -1}
		localOut {Type I LastRead 2 FirstWrite -1}
		zext_ln102 {Type I LastRead 0 FirstWrite -1}
		mul_ln91_3 {Type I LastRead 0 FirstWrite -1}
		tmp_cast_mid1257 {Type I LastRead 0 FirstWrite -1}
		select_ln91 {Type I LastRead 0 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "-1", "Max" : "-1"}
	, {"Name" : "Interval", "Min" : "-1", "Max" : "-1"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	 { m_axi {  { m_axi_mem2_AWVALID VALID 1 1 }  { m_axi_mem2_AWREADY READY 0 1 }  { m_axi_mem2_AWADDR ADDR 1 64 }  { m_axi_mem2_AWID ID 1 1 }  { m_axi_mem2_AWLEN SIZE 1 32 }  { m_axi_mem2_AWSIZE BURST 1 3 }  { m_axi_mem2_AWBURST LOCK 1 2 }  { m_axi_mem2_AWLOCK CACHE 1 2 }  { m_axi_mem2_AWCACHE PROT 1 4 }  { m_axi_mem2_AWPROT QOS 1 3 }  { m_axi_mem2_AWQOS REGION 1 4 }  { m_axi_mem2_AWREGION USER 1 4 }  { m_axi_mem2_AWUSER DATA 1 1 }  { m_axi_mem2_WVALID VALID 1 1 }  { m_axi_mem2_WREADY READY 0 1 }  { m_axi_mem2_WDATA FIFONUM 1 32 }  { m_axi_mem2_WSTRB STRB 1 4 }  { m_axi_mem2_WLAST LAST 1 1 }  { m_axi_mem2_WID ID 1 1 }  { m_axi_mem2_WUSER DATA 1 1 }  { m_axi_mem2_ARVALID VALID 1 1 }  { m_axi_mem2_ARREADY READY 0 1 }  { m_axi_mem2_ARADDR ADDR 1 64 }  { m_axi_mem2_ARID ID 1 1 }  { m_axi_mem2_ARLEN SIZE 1 32 }  { m_axi_mem2_ARSIZE BURST 1 3 }  { m_axi_mem2_ARBURST LOCK 1 2 }  { m_axi_mem2_ARLOCK CACHE 1 2 }  { m_axi_mem2_ARCACHE PROT 1 4 }  { m_axi_mem2_ARPROT QOS 1 3 }  { m_axi_mem2_ARQOS REGION 1 4 }  { m_axi_mem2_ARREGION USER 1 4 }  { m_axi_mem2_ARUSER DATA 1 1 }  { m_axi_mem2_RVALID VALID 0 1 }  { m_axi_mem2_RREADY READY 1 1 }  { m_axi_mem2_RDATA FIFONUM 0 32 }  { m_axi_mem2_RLAST LAST 0 1 }  { m_axi_mem2_RID ID 0 1 }  { m_axi_mem2_RFIFONUM LEN 0 9 }  { m_axi_mem2_RUSER DATA 0 1 }  { m_axi_mem2_RRESP RESP 0 2 }  { m_axi_mem2_BVALID VALID 0 1 }  { m_axi_mem2_BREADY READY 1 1 }  { m_axi_mem2_BRESP RESP 0 2 }  { m_axi_mem2_BID ID 0 1 }  { m_axi_mem2_BUSER DATA 0 1 } } }
	zext_ln104 { ap_none {  { zext_ln104 in_data 0 32 } } }
	mul_ln90_2 { ap_none {  { mul_ln90_2 in_data 0 31 } } }
	sext_ln75 { ap_none {  { sext_ln75 in_data 0 32 } } }
	zext_ln91_2 { ap_none {  { zext_ln91_2 in_data 0 32 } } }
	output_r { ap_none {  { output_r in_data 0 64 } } }
	xor_ln91 { ap_none {  { xor_ln91 in_data 0 32 } } }
	mul_ln192 { ap_none {  { mul_ln192 in_data 0 96 } } }
	localOut { ap_memory {  { localOut_address0 mem_address 1 12 }  { localOut_ce0 mem_ce 1 1 }  { localOut_q0 mem_dout 0 32 } } }
	zext_ln102 { ap_none {  { zext_ln102 in_data 0 32 } } }
	mul_ln91_3 { ap_none {  { mul_ln91_3 in_data 0 64 } } }
	tmp_cast_mid1257 { ap_none {  { tmp_cast_mid1257 in_data 0 31 } } }
	select_ln91 { ap_none {  { select_ln91 in_data 0 32 } } }
}
