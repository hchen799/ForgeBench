set moduleName conv_via_tiling_Pipeline_VITIS_LOOP_150_17_VITIS_LOOP_151_18_VITIS_LOOP_152_19_V
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
set C_modelName {conv_via_tiling_Pipeline_VITIS_LOOP_150_17_VITIS_LOOP_151_18_VITIS_LOOP_152_19_V}
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
	{ mem1 int 32 regular {axi_master 0}  }
	{ mul_ln131_1 int 68 regular  }
	{ localW float 32 regular {array 4096 { 0 3 } 0 1 }  }
	{ localW_1 float 32 regular {array 4096 { 0 3 } 0 1 }  }
	{ localW_2 float 32 regular {array 4096 { 0 3 } 0 1 }  }
	{ localW_3 float 32 regular {array 4096 { 0 3 } 0 1 }  }
	{ localW_4 float 32 regular {array 4096 { 0 3 } 0 1 }  }
	{ localW_5 float 32 regular {array 4096 { 0 3 } 0 1 }  }
	{ localW_6 float 32 regular {array 4096 { 0 3 } 0 1 }  }
	{ localW_7 float 32 regular {array 4096 { 0 3 } 0 1 }  }
	{ localW_8 float 32 regular {array 4096 { 0 3 } 0 1 }  }
	{ zext_ln150 int 35 regular  }
	{ mul_ln131 int 36 regular  }
	{ zext_ln104 int 32 regular  }
	{ weight int 64 regular  }
	{ p_cast44 int 38 regular  }
}
set hasAXIMCache 0
set hasAXIML2Cache 0
set AXIMCacheInstDict [dict create]
set C_modelArgMapList {[ 
	{ "Name" : "mem1", "interface" : "axi_master", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[ {"cElement": [{"cName": "input_r","offset": { "type": "dynamic","port_name": "input_r","bundle": "control"},"direction": "READONLY"},{"cName": "weight","offset": { "type": "dynamic","port_name": "weight","bundle": "control"},"direction": "READONLY"},{"cName": "bias","offset": { "type": "dynamic","port_name": "bias","bundle": "control"},"direction": "READONLY"}]}]} , 
 	{ "Name" : "mul_ln131_1", "interface" : "wire", "bitwidth" : 68, "direction" : "READONLY"} , 
 	{ "Name" : "localW", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "localW_1", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "localW_2", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "localW_3", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "localW_4", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "localW_5", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "localW_6", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "localW_7", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "localW_8", "interface" : "memory", "bitwidth" : 32, "direction" : "WRITEONLY"} , 
 	{ "Name" : "zext_ln150", "interface" : "wire", "bitwidth" : 35, "direction" : "READONLY"} , 
 	{ "Name" : "mul_ln131", "interface" : "wire", "bitwidth" : 36, "direction" : "READONLY"} , 
 	{ "Name" : "zext_ln104", "interface" : "wire", "bitwidth" : 32, "direction" : "READONLY"} , 
 	{ "Name" : "weight", "interface" : "wire", "bitwidth" : 64, "direction" : "READONLY"} , 
 	{ "Name" : "p_cast44", "interface" : "wire", "bitwidth" : 38, "direction" : "READONLY"} ]}
# RTL Port declarations: 
set portNum 94
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ m_axi_mem1_AWVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_mem1_AWREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_mem1_AWADDR sc_out sc_lv 64 signal 0 } 
	{ m_axi_mem1_AWID sc_out sc_lv 1 signal 0 } 
	{ m_axi_mem1_AWLEN sc_out sc_lv 32 signal 0 } 
	{ m_axi_mem1_AWSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_mem1_AWBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_mem1_AWLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_mem1_AWCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_mem1_AWPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_mem1_AWQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_mem1_AWREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_mem1_AWUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_mem1_WVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_mem1_WREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_mem1_WDATA sc_out sc_lv 32 signal 0 } 
	{ m_axi_mem1_WSTRB sc_out sc_lv 4 signal 0 } 
	{ m_axi_mem1_WLAST sc_out sc_logic 1 signal 0 } 
	{ m_axi_mem1_WID sc_out sc_lv 1 signal 0 } 
	{ m_axi_mem1_WUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_mem1_ARVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_mem1_ARREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_mem1_ARADDR sc_out sc_lv 64 signal 0 } 
	{ m_axi_mem1_ARID sc_out sc_lv 1 signal 0 } 
	{ m_axi_mem1_ARLEN sc_out sc_lv 32 signal 0 } 
	{ m_axi_mem1_ARSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_mem1_ARBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_mem1_ARLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_mem1_ARCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_mem1_ARPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_mem1_ARQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_mem1_ARREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_mem1_ARUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_mem1_RVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_mem1_RREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_mem1_RDATA sc_in sc_lv 32 signal 0 } 
	{ m_axi_mem1_RLAST sc_in sc_logic 1 signal 0 } 
	{ m_axi_mem1_RID sc_in sc_lv 1 signal 0 } 
	{ m_axi_mem1_RFIFONUM sc_in sc_lv 9 signal 0 } 
	{ m_axi_mem1_RUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_mem1_RRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_mem1_BVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_mem1_BREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_mem1_BRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_mem1_BID sc_in sc_lv 1 signal 0 } 
	{ m_axi_mem1_BUSER sc_in sc_lv 1 signal 0 } 
	{ mul_ln131_1 sc_in sc_lv 68 signal 1 } 
	{ localW_address0 sc_out sc_lv 12 signal 2 } 
	{ localW_ce0 sc_out sc_logic 1 signal 2 } 
	{ localW_we0 sc_out sc_logic 1 signal 2 } 
	{ localW_d0 sc_out sc_lv 32 signal 2 } 
	{ localW_1_address0 sc_out sc_lv 12 signal 3 } 
	{ localW_1_ce0 sc_out sc_logic 1 signal 3 } 
	{ localW_1_we0 sc_out sc_logic 1 signal 3 } 
	{ localW_1_d0 sc_out sc_lv 32 signal 3 } 
	{ localW_2_address0 sc_out sc_lv 12 signal 4 } 
	{ localW_2_ce0 sc_out sc_logic 1 signal 4 } 
	{ localW_2_we0 sc_out sc_logic 1 signal 4 } 
	{ localW_2_d0 sc_out sc_lv 32 signal 4 } 
	{ localW_3_address0 sc_out sc_lv 12 signal 5 } 
	{ localW_3_ce0 sc_out sc_logic 1 signal 5 } 
	{ localW_3_we0 sc_out sc_logic 1 signal 5 } 
	{ localW_3_d0 sc_out sc_lv 32 signal 5 } 
	{ localW_4_address0 sc_out sc_lv 12 signal 6 } 
	{ localW_4_ce0 sc_out sc_logic 1 signal 6 } 
	{ localW_4_we0 sc_out sc_logic 1 signal 6 } 
	{ localW_4_d0 sc_out sc_lv 32 signal 6 } 
	{ localW_5_address0 sc_out sc_lv 12 signal 7 } 
	{ localW_5_ce0 sc_out sc_logic 1 signal 7 } 
	{ localW_5_we0 sc_out sc_logic 1 signal 7 } 
	{ localW_5_d0 sc_out sc_lv 32 signal 7 } 
	{ localW_6_address0 sc_out sc_lv 12 signal 8 } 
	{ localW_6_ce0 sc_out sc_logic 1 signal 8 } 
	{ localW_6_we0 sc_out sc_logic 1 signal 8 } 
	{ localW_6_d0 sc_out sc_lv 32 signal 8 } 
	{ localW_7_address0 sc_out sc_lv 12 signal 9 } 
	{ localW_7_ce0 sc_out sc_logic 1 signal 9 } 
	{ localW_7_we0 sc_out sc_logic 1 signal 9 } 
	{ localW_7_d0 sc_out sc_lv 32 signal 9 } 
	{ localW_8_address0 sc_out sc_lv 12 signal 10 } 
	{ localW_8_ce0 sc_out sc_logic 1 signal 10 } 
	{ localW_8_we0 sc_out sc_logic 1 signal 10 } 
	{ localW_8_d0 sc_out sc_lv 32 signal 10 } 
	{ zext_ln150 sc_in sc_lv 35 signal 11 } 
	{ mul_ln131 sc_in sc_lv 36 signal 12 } 
	{ zext_ln104 sc_in sc_lv 32 signal 13 } 
	{ weight sc_in sc_lv 64 signal 14 } 
	{ p_cast44 sc_in sc_lv 38 signal 15 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "m_axi_mem1_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem1", "role": "AWVALID" }} , 
 	{ "name": "m_axi_mem1_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem1", "role": "AWREADY" }} , 
 	{ "name": "m_axi_mem1_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "mem1", "role": "AWADDR" }} , 
 	{ "name": "m_axi_mem1_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem1", "role": "AWID" }} , 
 	{ "name": "m_axi_mem1_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "mem1", "role": "AWLEN" }} , 
 	{ "name": "m_axi_mem1_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "mem1", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_mem1_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "mem1", "role": "AWBURST" }} , 
 	{ "name": "m_axi_mem1_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "mem1", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_mem1_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "mem1", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_mem1_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "mem1", "role": "AWPROT" }} , 
 	{ "name": "m_axi_mem1_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "mem1", "role": "AWQOS" }} , 
 	{ "name": "m_axi_mem1_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "mem1", "role": "AWREGION" }} , 
 	{ "name": "m_axi_mem1_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem1", "role": "AWUSER" }} , 
 	{ "name": "m_axi_mem1_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem1", "role": "WVALID" }} , 
 	{ "name": "m_axi_mem1_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem1", "role": "WREADY" }} , 
 	{ "name": "m_axi_mem1_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "mem1", "role": "WDATA" }} , 
 	{ "name": "m_axi_mem1_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "mem1", "role": "WSTRB" }} , 
 	{ "name": "m_axi_mem1_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem1", "role": "WLAST" }} , 
 	{ "name": "m_axi_mem1_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem1", "role": "WID" }} , 
 	{ "name": "m_axi_mem1_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem1", "role": "WUSER" }} , 
 	{ "name": "m_axi_mem1_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem1", "role": "ARVALID" }} , 
 	{ "name": "m_axi_mem1_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem1", "role": "ARREADY" }} , 
 	{ "name": "m_axi_mem1_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "mem1", "role": "ARADDR" }} , 
 	{ "name": "m_axi_mem1_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem1", "role": "ARID" }} , 
 	{ "name": "m_axi_mem1_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "mem1", "role": "ARLEN" }} , 
 	{ "name": "m_axi_mem1_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "mem1", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_mem1_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "mem1", "role": "ARBURST" }} , 
 	{ "name": "m_axi_mem1_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "mem1", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_mem1_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "mem1", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_mem1_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "mem1", "role": "ARPROT" }} , 
 	{ "name": "m_axi_mem1_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "mem1", "role": "ARQOS" }} , 
 	{ "name": "m_axi_mem1_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "mem1", "role": "ARREGION" }} , 
 	{ "name": "m_axi_mem1_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem1", "role": "ARUSER" }} , 
 	{ "name": "m_axi_mem1_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem1", "role": "RVALID" }} , 
 	{ "name": "m_axi_mem1_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem1", "role": "RREADY" }} , 
 	{ "name": "m_axi_mem1_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "mem1", "role": "RDATA" }} , 
 	{ "name": "m_axi_mem1_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem1", "role": "RLAST" }} , 
 	{ "name": "m_axi_mem1_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem1", "role": "RID" }} , 
 	{ "name": "m_axi_mem1_RFIFONUM", "direction": "in", "datatype": "sc_lv", "bitwidth":9, "type": "signal", "bundle":{"name": "mem1", "role": "RFIFONUM" }} , 
 	{ "name": "m_axi_mem1_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem1", "role": "RUSER" }} , 
 	{ "name": "m_axi_mem1_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "mem1", "role": "RRESP" }} , 
 	{ "name": "m_axi_mem1_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem1", "role": "BVALID" }} , 
 	{ "name": "m_axi_mem1_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem1", "role": "BREADY" }} , 
 	{ "name": "m_axi_mem1_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "mem1", "role": "BRESP" }} , 
 	{ "name": "m_axi_mem1_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem1", "role": "BID" }} , 
 	{ "name": "m_axi_mem1_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem1", "role": "BUSER" }} , 
 	{ "name": "mul_ln131_1", "direction": "in", "datatype": "sc_lv", "bitwidth":68, "type": "signal", "bundle":{"name": "mul_ln131_1", "role": "default" }} , 
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
 	{ "name": "localW_8_d0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "localW_8", "role": "d0" }} , 
 	{ "name": "zext_ln150", "direction": "in", "datatype": "sc_lv", "bitwidth":35, "type": "signal", "bundle":{"name": "zext_ln150", "role": "default" }} , 
 	{ "name": "mul_ln131", "direction": "in", "datatype": "sc_lv", "bitwidth":36, "type": "signal", "bundle":{"name": "mul_ln131", "role": "default" }} , 
 	{ "name": "zext_ln104", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "zext_ln104", "role": "default" }} , 
 	{ "name": "weight", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "weight", "role": "default" }} , 
 	{ "name": "p_cast44", "direction": "in", "datatype": "sc_lv", "bitwidth":38, "type": "signal", "bundle":{"name": "p_cast44", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1"],
		"CDFG" : "conv_via_tiling_Pipeline_VITIS_LOOP_150_17_VITIS_LOOP_151_18_VITIS_LOOP_152_19_V",
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
			{"Name" : "mem1", "Type" : "MAXI", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "mem1_blk_n_R", "Type" : "RtlSignal"},
					{"Name" : "mem1_blk_n_AR", "Type" : "RtlSignal"}]},
			{"Name" : "mul_ln131_1", "Type" : "None", "Direction" : "I"},
			{"Name" : "localW", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localW_1", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localW_2", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localW_3", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localW_4", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localW_5", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localW_6", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localW_7", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localW_8", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "zext_ln150", "Type" : "None", "Direction" : "I"},
			{"Name" : "mul_ln131", "Type" : "None", "Direction" : "I"},
			{"Name" : "zext_ln104", "Type" : "None", "Direction" : "I"},
			{"Name" : "weight", "Type" : "None", "Direction" : "I"},
			{"Name" : "p_cast44", "Type" : "None", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_150_17_VITIS_LOOP_151_18_VITIS_LOOP_152_19_VITIS_LOOP_153_20", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter11", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter11", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.flow_control_loop_pipe_sequential_init_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	conv_via_tiling_Pipeline_VITIS_LOOP_150_17_VITIS_LOOP_151_18_VITIS_LOOP_152_19_V {
		mem1 {Type I LastRead 10 FirstWrite -1}
		mul_ln131_1 {Type I LastRead 0 FirstWrite -1}
		localW {Type O LastRead -1 FirstWrite 11}
		localW_1 {Type O LastRead -1 FirstWrite 11}
		localW_2 {Type O LastRead -1 FirstWrite 11}
		localW_3 {Type O LastRead -1 FirstWrite 11}
		localW_4 {Type O LastRead -1 FirstWrite 11}
		localW_5 {Type O LastRead -1 FirstWrite 11}
		localW_6 {Type O LastRead -1 FirstWrite 11}
		localW_7 {Type O LastRead -1 FirstWrite 11}
		localW_8 {Type O LastRead -1 FirstWrite 11}
		zext_ln150 {Type I LastRead 0 FirstWrite -1}
		mul_ln131 {Type I LastRead 0 FirstWrite -1}
		zext_ln104 {Type I LastRead 0 FirstWrite -1}
		weight {Type I LastRead 0 FirstWrite -1}
		p_cast44 {Type I LastRead 0 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "-1", "Max" : "-1"}
	, {"Name" : "Interval", "Min" : "-1", "Max" : "-1"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	 { m_axi {  { m_axi_mem1_AWVALID VALID 1 1 }  { m_axi_mem1_AWREADY READY 0 1 }  { m_axi_mem1_AWADDR ADDR 1 64 }  { m_axi_mem1_AWID ID 1 1 }  { m_axi_mem1_AWLEN SIZE 1 32 }  { m_axi_mem1_AWSIZE BURST 1 3 }  { m_axi_mem1_AWBURST LOCK 1 2 }  { m_axi_mem1_AWLOCK CACHE 1 2 }  { m_axi_mem1_AWCACHE PROT 1 4 }  { m_axi_mem1_AWPROT QOS 1 3 }  { m_axi_mem1_AWQOS REGION 1 4 }  { m_axi_mem1_AWREGION USER 1 4 }  { m_axi_mem1_AWUSER DATA 1 1 }  { m_axi_mem1_WVALID VALID 1 1 }  { m_axi_mem1_WREADY READY 0 1 }  { m_axi_mem1_WDATA FIFONUM 1 32 }  { m_axi_mem1_WSTRB STRB 1 4 }  { m_axi_mem1_WLAST LAST 1 1 }  { m_axi_mem1_WID ID 1 1 }  { m_axi_mem1_WUSER DATA 1 1 }  { m_axi_mem1_ARVALID VALID 1 1 }  { m_axi_mem1_ARREADY READY 0 1 }  { m_axi_mem1_ARADDR ADDR 1 64 }  { m_axi_mem1_ARID ID 1 1 }  { m_axi_mem1_ARLEN SIZE 1 32 }  { m_axi_mem1_ARSIZE BURST 1 3 }  { m_axi_mem1_ARBURST LOCK 1 2 }  { m_axi_mem1_ARLOCK CACHE 1 2 }  { m_axi_mem1_ARCACHE PROT 1 4 }  { m_axi_mem1_ARPROT QOS 1 3 }  { m_axi_mem1_ARQOS REGION 1 4 }  { m_axi_mem1_ARREGION USER 1 4 }  { m_axi_mem1_ARUSER DATA 1 1 }  { m_axi_mem1_RVALID VALID 0 1 }  { m_axi_mem1_RREADY READY 1 1 }  { m_axi_mem1_RDATA FIFONUM 0 32 }  { m_axi_mem1_RLAST LAST 0 1 }  { m_axi_mem1_RID ID 0 1 }  { m_axi_mem1_RFIFONUM LEN 0 9 }  { m_axi_mem1_RUSER DATA 0 1 }  { m_axi_mem1_RRESP RESP 0 2 }  { m_axi_mem1_BVALID VALID 0 1 }  { m_axi_mem1_BREADY READY 1 1 }  { m_axi_mem1_BRESP RESP 0 2 }  { m_axi_mem1_BID ID 0 1 }  { m_axi_mem1_BUSER DATA 0 1 } } }
	mul_ln131_1 { ap_none {  { mul_ln131_1 in_data 0 68 } } }
	localW { ap_memory {  { localW_address0 mem_address 1 12 }  { localW_ce0 mem_ce 1 1 }  { localW_we0 mem_we 1 1 }  { localW_d0 mem_din 1 32 } } }
	localW_1 { ap_memory {  { localW_1_address0 mem_address 1 12 }  { localW_1_ce0 mem_ce 1 1 }  { localW_1_we0 mem_we 1 1 }  { localW_1_d0 mem_din 1 32 } } }
	localW_2 { ap_memory {  { localW_2_address0 mem_address 1 12 }  { localW_2_ce0 mem_ce 1 1 }  { localW_2_we0 mem_we 1 1 }  { localW_2_d0 mem_din 1 32 } } }
	localW_3 { ap_memory {  { localW_3_address0 mem_address 1 12 }  { localW_3_ce0 mem_ce 1 1 }  { localW_3_we0 mem_we 1 1 }  { localW_3_d0 mem_din 1 32 } } }
	localW_4 { ap_memory {  { localW_4_address0 mem_address 1 12 }  { localW_4_ce0 mem_ce 1 1 }  { localW_4_we0 mem_we 1 1 }  { localW_4_d0 mem_din 1 32 } } }
	localW_5 { ap_memory {  { localW_5_address0 mem_address 1 12 }  { localW_5_ce0 mem_ce 1 1 }  { localW_5_we0 mem_we 1 1 }  { localW_5_d0 mem_din 1 32 } } }
	localW_6 { ap_memory {  { localW_6_address0 mem_address 1 12 }  { localW_6_ce0 mem_ce 1 1 }  { localW_6_we0 mem_we 1 1 }  { localW_6_d0 mem_din 1 32 } } }
	localW_7 { ap_memory {  { localW_7_address0 mem_address 1 12 }  { localW_7_ce0 mem_ce 1 1 }  { localW_7_we0 mem_we 1 1 }  { localW_7_d0 mem_din 1 32 } } }
	localW_8 { ap_memory {  { localW_8_address0 mem_address 1 12 }  { localW_8_ce0 mem_ce 1 1 }  { localW_8_we0 mem_we 1 1 }  { localW_8_d0 mem_din 1 32 } } }
	zext_ln150 { ap_none {  { zext_ln150 in_data 0 35 } } }
	mul_ln131 { ap_none {  { mul_ln131 in_data 0 36 } } }
	zext_ln104 { ap_none {  { zext_ln104 in_data 0 32 } } }
	weight { ap_none {  { weight in_data 0 64 } } }
	p_cast44 { ap_none {  { p_cast44 in_data 0 38 } } }
}
