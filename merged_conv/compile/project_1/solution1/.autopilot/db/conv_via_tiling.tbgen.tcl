set moduleName conv_via_tiling
set isTopModule 1
set isCombinational 0
set isDatapathOnly 0
set isPipelined 0
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set hasInterrupt 0
set DLRegFirstOffset 0
set DLRegItemOffset 0
set C_modelName {conv_via_tiling}
set C_modelType { void 0 }
set ap_memory_interface_dict [dict create]
set C_modelArgList {
	{ mem1 int 32 regular {axi_master 0}  }
	{ mem2 int 32 regular {axi_master 2}  }
	{ in_ch int 32 regular {axi_slave 0}  }
	{ out_ch int 32 regular {axi_slave 0}  }
	{ H int 32 regular {axi_slave 0}  }
	{ W int 32 regular {axi_slave 0}  }
	{ input_r int 64 regular {axi_slave 0}  }
	{ weight int 64 regular {axi_slave 0}  }
	{ bias int 64 regular {axi_slave 0}  }
	{ output_r int 64 regular {axi_slave 0}  }
	{ stride int 32 regular {axi_slave 0}  }
	{ pad int 32 regular {axi_slave 0}  }
}
set hasAXIMCache 0
set hasAXIML2Cache 0
set AXIMCacheInstDict [dict create]
set C_modelArgMapList {[ 
	{ "Name" : "mem1", "interface" : "axi_master", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[ {"cElement": [{"cName": "input_r","offset": { "type": "dynamic","port_name": "input_r","bundle": "control"},"direction": "READONLY"},{"cName": "weight","offset": { "type": "dynamic","port_name": "weight","bundle": "control"},"direction": "READONLY"},{"cName": "bias","offset": { "type": "dynamic","port_name": "bias","bundle": "control"},"direction": "READONLY"}]}]} , 
 	{ "Name" : "mem2", "interface" : "axi_master", "bitwidth" : 32, "direction" : "READWRITE", "bitSlice":[ {"cElement": [{"cName": "output_r","offset": { "type": "dynamic","port_name": "output_r","bundle": "control"},"direction": "READWRITE"}]}]} , 
 	{ "Name" : "in_ch", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":16}, "offset_end" : {"in":23}} , 
 	{ "Name" : "out_ch", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":24}, "offset_end" : {"in":31}} , 
 	{ "Name" : "H", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":32}, "offset_end" : {"in":39}} , 
 	{ "Name" : "W", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":40}, "offset_end" : {"in":47}} , 
 	{ "Name" : "input_r", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 64, "direction" : "READONLY", "offset" : {"in":48}, "offset_end" : {"in":59}} , 
 	{ "Name" : "weight", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 64, "direction" : "READONLY", "offset" : {"in":60}, "offset_end" : {"in":71}} , 
 	{ "Name" : "bias", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 64, "direction" : "READONLY", "offset" : {"in":72}, "offset_end" : {"in":83}} , 
 	{ "Name" : "output_r", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 64, "direction" : "READONLY", "offset" : {"in":84}, "offset_end" : {"in":95}} , 
 	{ "Name" : "stride", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":96}, "offset_end" : {"in":103}} , 
 	{ "Name" : "pad", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":104}, "offset_end" : {"in":111}} ]}
# RTL Port declarations: 
set portNum 113
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst_n sc_in sc_logic 1 reset -1 active_low_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ m_axi_mem1_AWVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_mem1_AWREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_mem1_AWADDR sc_out sc_lv 64 signal 0 } 
	{ m_axi_mem1_AWID sc_out sc_lv 1 signal 0 } 
	{ m_axi_mem1_AWLEN sc_out sc_lv 8 signal 0 } 
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
	{ m_axi_mem1_ARLEN sc_out sc_lv 8 signal 0 } 
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
	{ m_axi_mem1_RUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_mem1_RRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_mem1_BVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_mem1_BREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_mem1_BRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_mem1_BID sc_in sc_lv 1 signal 0 } 
	{ m_axi_mem1_BUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_mem2_AWVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_mem2_AWREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_mem2_AWADDR sc_out sc_lv 64 signal 1 } 
	{ m_axi_mem2_AWID sc_out sc_lv 1 signal 1 } 
	{ m_axi_mem2_AWLEN sc_out sc_lv 8 signal 1 } 
	{ m_axi_mem2_AWSIZE sc_out sc_lv 3 signal 1 } 
	{ m_axi_mem2_AWBURST sc_out sc_lv 2 signal 1 } 
	{ m_axi_mem2_AWLOCK sc_out sc_lv 2 signal 1 } 
	{ m_axi_mem2_AWCACHE sc_out sc_lv 4 signal 1 } 
	{ m_axi_mem2_AWPROT sc_out sc_lv 3 signal 1 } 
	{ m_axi_mem2_AWQOS sc_out sc_lv 4 signal 1 } 
	{ m_axi_mem2_AWREGION sc_out sc_lv 4 signal 1 } 
	{ m_axi_mem2_AWUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_mem2_WVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_mem2_WREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_mem2_WDATA sc_out sc_lv 32 signal 1 } 
	{ m_axi_mem2_WSTRB sc_out sc_lv 4 signal 1 } 
	{ m_axi_mem2_WLAST sc_out sc_logic 1 signal 1 } 
	{ m_axi_mem2_WID sc_out sc_lv 1 signal 1 } 
	{ m_axi_mem2_WUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_mem2_ARVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_mem2_ARREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_mem2_ARADDR sc_out sc_lv 64 signal 1 } 
	{ m_axi_mem2_ARID sc_out sc_lv 1 signal 1 } 
	{ m_axi_mem2_ARLEN sc_out sc_lv 8 signal 1 } 
	{ m_axi_mem2_ARSIZE sc_out sc_lv 3 signal 1 } 
	{ m_axi_mem2_ARBURST sc_out sc_lv 2 signal 1 } 
	{ m_axi_mem2_ARLOCK sc_out sc_lv 2 signal 1 } 
	{ m_axi_mem2_ARCACHE sc_out sc_lv 4 signal 1 } 
	{ m_axi_mem2_ARPROT sc_out sc_lv 3 signal 1 } 
	{ m_axi_mem2_ARQOS sc_out sc_lv 4 signal 1 } 
	{ m_axi_mem2_ARREGION sc_out sc_lv 4 signal 1 } 
	{ m_axi_mem2_ARUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_mem2_RVALID sc_in sc_logic 1 signal 1 } 
	{ m_axi_mem2_RREADY sc_out sc_logic 1 signal 1 } 
	{ m_axi_mem2_RDATA sc_in sc_lv 32 signal 1 } 
	{ m_axi_mem2_RLAST sc_in sc_logic 1 signal 1 } 
	{ m_axi_mem2_RID sc_in sc_lv 1 signal 1 } 
	{ m_axi_mem2_RUSER sc_in sc_lv 1 signal 1 } 
	{ m_axi_mem2_RRESP sc_in sc_lv 2 signal 1 } 
	{ m_axi_mem2_BVALID sc_in sc_logic 1 signal 1 } 
	{ m_axi_mem2_BREADY sc_out sc_logic 1 signal 1 } 
	{ m_axi_mem2_BRESP sc_in sc_lv 2 signal 1 } 
	{ m_axi_mem2_BID sc_in sc_lv 1 signal 1 } 
	{ m_axi_mem2_BUSER sc_in sc_lv 1 signal 1 } 
	{ s_axi_control_AWVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_AWREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_AWADDR sc_in sc_lv 7 signal -1 } 
	{ s_axi_control_WVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_WREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_WDATA sc_in sc_lv 32 signal -1 } 
	{ s_axi_control_WSTRB sc_in sc_lv 4 signal -1 } 
	{ s_axi_control_ARVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_ARREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_ARADDR sc_in sc_lv 7 signal -1 } 
	{ s_axi_control_RVALID sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_RREADY sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_RDATA sc_out sc_lv 32 signal -1 } 
	{ s_axi_control_RRESP sc_out sc_lv 2 signal -1 } 
	{ s_axi_control_BVALID sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_BREADY sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_BRESP sc_out sc_lv 2 signal -1 } 
}
set NewPortList {[ 
	{ "name": "s_axi_control_AWADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "control", "role": "AWADDR" },"address":[{"name":"in_ch","role":"data","value":"16"},{"name":"out_ch","role":"data","value":"24"},{"name":"H","role":"data","value":"32"},{"name":"W","role":"data","value":"40"},{"name":"input_r","role":"data","value":"48"},{"name":"weight","role":"data","value":"60"},{"name":"bias","role":"data","value":"72"},{"name":"output_r","role":"data","value":"84"},{"name":"stride","role":"data","value":"96"},{"name":"pad","role":"data","value":"104"}] },
	{ "name": "s_axi_control_AWVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "AWVALID" } },
	{ "name": "s_axi_control_AWREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "AWREADY" } },
	{ "name": "s_axi_control_WVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "WVALID" } },
	{ "name": "s_axi_control_WREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "WREADY" } },
	{ "name": "s_axi_control_WDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "control", "role": "WDATA" } },
	{ "name": "s_axi_control_WSTRB", "direction": "in", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "control", "role": "WSTRB" } },
	{ "name": "s_axi_control_ARADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "control", "role": "ARADDR" },"address":[] },
	{ "name": "s_axi_control_ARVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "ARVALID" } },
	{ "name": "s_axi_control_ARREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "ARREADY" } },
	{ "name": "s_axi_control_RVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "RVALID" } },
	{ "name": "s_axi_control_RREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "RREADY" } },
	{ "name": "s_axi_control_RDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "control", "role": "RDATA" } },
	{ "name": "s_axi_control_RRESP", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "control", "role": "RRESP" } },
	{ "name": "s_axi_control_BVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "BVALID" } },
	{ "name": "s_axi_control_BREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "BREADY" } },
	{ "name": "s_axi_control_BRESP", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "control", "role": "BRESP" } }, 
 	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst_n", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "m_axi_mem1_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem1", "role": "AWVALID" }} , 
 	{ "name": "m_axi_mem1_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem1", "role": "AWREADY" }} , 
 	{ "name": "m_axi_mem1_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "mem1", "role": "AWADDR" }} , 
 	{ "name": "m_axi_mem1_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem1", "role": "AWID" }} , 
 	{ "name": "m_axi_mem1_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "mem1", "role": "AWLEN" }} , 
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
 	{ "name": "m_axi_mem1_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "mem1", "role": "ARLEN" }} , 
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
 	{ "name": "m_axi_mem1_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem1", "role": "RUSER" }} , 
 	{ "name": "m_axi_mem1_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "mem1", "role": "RRESP" }} , 
 	{ "name": "m_axi_mem1_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem1", "role": "BVALID" }} , 
 	{ "name": "m_axi_mem1_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem1", "role": "BREADY" }} , 
 	{ "name": "m_axi_mem1_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "mem1", "role": "BRESP" }} , 
 	{ "name": "m_axi_mem1_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem1", "role": "BID" }} , 
 	{ "name": "m_axi_mem1_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem1", "role": "BUSER" }} , 
 	{ "name": "m_axi_mem2_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem2", "role": "AWVALID" }} , 
 	{ "name": "m_axi_mem2_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem2", "role": "AWREADY" }} , 
 	{ "name": "m_axi_mem2_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "mem2", "role": "AWADDR" }} , 
 	{ "name": "m_axi_mem2_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem2", "role": "AWID" }} , 
 	{ "name": "m_axi_mem2_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "mem2", "role": "AWLEN" }} , 
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
 	{ "name": "m_axi_mem2_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "mem2", "role": "ARLEN" }} , 
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
 	{ "name": "m_axi_mem2_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem2", "role": "RUSER" }} , 
 	{ "name": "m_axi_mem2_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "mem2", "role": "RRESP" }} , 
 	{ "name": "m_axi_mem2_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem2", "role": "BVALID" }} , 
 	{ "name": "m_axi_mem2_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "mem2", "role": "BREADY" }} , 
 	{ "name": "m_axi_mem2_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "mem2", "role": "BRESP" }} , 
 	{ "name": "m_axi_mem2_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem2", "role": "BID" }} , 
 	{ "name": "m_axi_mem2_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "mem2", "role": "BUSER" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "71", "73", "75", "80", "82", "88", "90", "97", "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110", "111", "112", "113", "114"],
		"CDFG" : "conv_via_tiling",
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
				"SubConnect" : [
					{"ID" : "71", "SubInstance" : "grp_conv_via_tiling_Pipeline_VITIS_LOOP_75_1_VITIS_LOOP_76_2_VITIS_LOOP_77_3_fu_419", "Port" : "mem1", "Inst_start_state" : "37", "Inst_end_state" : "38"},
					{"ID" : "88", "SubInstance" : "grp_conv_via_tiling_Pipeline_VITIS_LOOP_150_17_VITIS_LOOP_151_18_VITIS_LOOP_152_19_V_fu_481", "Port" : "mem1", "Inst_start_state" : "50", "Inst_end_state" : "51"},
					{"ID" : "90", "SubInstance" : "grp_conv_via_tiling_Pipeline_VITIS_LOOP_174_24_VITIS_LOOP_175_25_VITIS_LOOP_176_26_fu_502", "Port" : "mem1", "Inst_start_state" : "52", "Inst_end_state" : "53"}]},
			{"Name" : "mem2", "Type" : "MAXI", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "71", "SubInstance" : "grp_conv_via_tiling_Pipeline_VITIS_LOOP_75_1_VITIS_LOOP_76_2_VITIS_LOOP_77_3_fu_419", "Port" : "mem2", "Inst_start_state" : "37", "Inst_end_state" : "38"},
					{"ID" : "75", "SubInstance" : "grp_conv_via_tiling_Pipeline_VITIS_LOOP_119_10_VITIS_LOOP_120_11_fu_440", "Port" : "mem2", "Inst_start_state" : "46", "Inst_end_state" : "47"},
					{"ID" : "97", "SubInstance" : "grp_conv_via_tiling_Pipeline_VITIS_LOOP_192_27_VITIS_LOOP_193_28_VITIS_LOOP_194_29_fu_527", "Port" : "mem2", "Inst_start_state" : "56", "Inst_end_state" : "57"}]},
			{"Name" : "in_ch", "Type" : "None", "Direction" : "I"},
			{"Name" : "out_ch", "Type" : "None", "Direction" : "I"},
			{"Name" : "H", "Type" : "None", "Direction" : "I"},
			{"Name" : "W", "Type" : "None", "Direction" : "I"},
			{"Name" : "input_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "weight", "Type" : "None", "Direction" : "I"},
			{"Name" : "bias", "Type" : "None", "Direction" : "I"},
			{"Name" : "output_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "stride", "Type" : "None", "Direction" : "I"},
			{"Name" : "pad", "Type" : "None", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_129_12", "PipelineType" : "no",
				"LoopDec" : {"FSMBitwidth" : "57", "FirstState" : "ap_ST_fsm_state48", "LastState" : ["ap_ST_fsm_state55"], "QuitState" : ["ap_ST_fsm_state48"], "PreState" : ["ap_ST_fsm_state47"], "PostState" : ["ap_ST_fsm_state56"], "OneDepthLoop" : "0", "OneStateBlock": ""}},
			{"Name" : "VITIS_LOOP_102_6", "PipelineType" : "no",
				"LoopDec" : {"FSMBitwidth" : "57", "FirstState" : "ap_ST_fsm_state44", "LastState" : ["ap_ST_fsm_state57"], "QuitState" : ["ap_ST_fsm_state44"], "PreState" : ["ap_ST_fsm_state43"], "PostState" : ["ap_ST_fsm_state41"], "OneDepthLoop" : "0", "OneStateBlock": ""}},
			{"Name" : "VITIS_LOOP_91_5", "PipelineType" : "no",
				"LoopDec" : {"FSMBitwidth" : "57", "FirstState" : "ap_ST_fsm_state41", "LastState" : ["ap_ST_fsm_state44"], "QuitState" : ["ap_ST_fsm_state41"], "PreState" : ["ap_ST_fsm_state40"], "PostState" : ["ap_ST_fsm_state39"], "OneDepthLoop" : "0", "OneStateBlock": ""}},
			{"Name" : "VITIS_LOOP_90_4", "PipelineType" : "no",
				"LoopDec" : {"FSMBitwidth" : "57", "FirstState" : "ap_ST_fsm_state39", "LastState" : ["ap_ST_fsm_state41"], "QuitState" : ["ap_ST_fsm_state39"], "PreState" : ["ap_ST_fsm_state38"], "PostState" : ["ap_ST_fsm_state1"], "OneDepthLoop" : "0", "OneStateBlock": ""}}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.localOut_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.localW_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.localW_1_U", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.localW_2_U", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.localW_3_U", "Parent" : "0"},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.localW_4_U", "Parent" : "0"},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.localW_5_U", "Parent" : "0"},
	{"ID" : "8", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.localW_6_U", "Parent" : "0"},
	{"ID" : "9", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.localW_7_U", "Parent" : "0"},
	{"ID" : "10", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.localW_8_U", "Parent" : "0"},
	{"ID" : "11", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.localIn_U", "Parent" : "0"},
	{"ID" : "12", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.localIn_1_U", "Parent" : "0"},
	{"ID" : "13", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.localIn_2_U", "Parent" : "0"},
	{"ID" : "14", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.localIn_3_U", "Parent" : "0"},
	{"ID" : "15", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.localIn_4_U", "Parent" : "0"},
	{"ID" : "16", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.localIn_5_U", "Parent" : "0"},
	{"ID" : "17", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.localIn_6_U", "Parent" : "0"},
	{"ID" : "18", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.localIn_7_U", "Parent" : "0"},
	{"ID" : "19", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.localIn_8_U", "Parent" : "0"},
	{"ID" : "20", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364", "Parent" : "0", "Child" : ["21", "70"],
		"CDFG" : "conv_kernel",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "257", "EstimateLatencyMax" : "208193",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "inData_0_0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "21", "SubInstance" : "grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94", "Port" : "inData_0_0", "Inst_start_state" : "2", "Inst_end_state" : "3"}]},
			{"Name" : "inData_0_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "21", "SubInstance" : "grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94", "Port" : "inData_0_1", "Inst_start_state" : "2", "Inst_end_state" : "3"}]},
			{"Name" : "inData_0_2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "21", "SubInstance" : "grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94", "Port" : "inData_0_2", "Inst_start_state" : "2", "Inst_end_state" : "3"}]},
			{"Name" : "inData_1_0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "21", "SubInstance" : "grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94", "Port" : "inData_1_0", "Inst_start_state" : "2", "Inst_end_state" : "3"}]},
			{"Name" : "inData_1_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "21", "SubInstance" : "grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94", "Port" : "inData_1_1", "Inst_start_state" : "2", "Inst_end_state" : "3"}]},
			{"Name" : "inData_1_2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "21", "SubInstance" : "grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94", "Port" : "inData_1_2", "Inst_start_state" : "2", "Inst_end_state" : "3"}]},
			{"Name" : "inData_2_0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "21", "SubInstance" : "grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94", "Port" : "inData_2_0", "Inst_start_state" : "2", "Inst_end_state" : "3"}]},
			{"Name" : "inData_2_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "21", "SubInstance" : "grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94", "Port" : "inData_2_1", "Inst_start_state" : "2", "Inst_end_state" : "3"}]},
			{"Name" : "inData_2_2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "21", "SubInstance" : "grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94", "Port" : "inData_2_2", "Inst_start_state" : "2", "Inst_end_state" : "3"}]},
			{"Name" : "weight_0_0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "21", "SubInstance" : "grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94", "Port" : "weight_0_0", "Inst_start_state" : "2", "Inst_end_state" : "3"}]},
			{"Name" : "weight_0_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "21", "SubInstance" : "grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94", "Port" : "weight_0_1", "Inst_start_state" : "2", "Inst_end_state" : "3"}]},
			{"Name" : "weight_0_2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "21", "SubInstance" : "grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94", "Port" : "weight_0_2", "Inst_start_state" : "2", "Inst_end_state" : "3"}]},
			{"Name" : "weight_1_0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "21", "SubInstance" : "grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94", "Port" : "weight_1_0", "Inst_start_state" : "2", "Inst_end_state" : "3"}]},
			{"Name" : "weight_1_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "21", "SubInstance" : "grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94", "Port" : "weight_1_1", "Inst_start_state" : "2", "Inst_end_state" : "3"}]},
			{"Name" : "weight_1_2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "21", "SubInstance" : "grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94", "Port" : "weight_1_2", "Inst_start_state" : "2", "Inst_end_state" : "3"}]},
			{"Name" : "weight_2_0", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "21", "SubInstance" : "grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94", "Port" : "weight_2_0", "Inst_start_state" : "2", "Inst_end_state" : "3"}]},
			{"Name" : "weight_2_1", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "21", "SubInstance" : "grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94", "Port" : "weight_2_1", "Inst_start_state" : "2", "Inst_end_state" : "3"}]},
			{"Name" : "weight_2_2", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "21", "SubInstance" : "grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94", "Port" : "weight_2_2", "Inst_start_state" : "2", "Inst_end_state" : "3"}]},
			{"Name" : "outData", "Type" : "Memory", "Direction" : "IO",
				"SubConnect" : [
					{"ID" : "21", "SubInstance" : "grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94", "Port" : "outData", "Inst_start_state" : "2", "Inst_end_state" : "3"}]},
			{"Name" : "outTileH", "Type" : "None", "Direction" : "I"},
			{"Name" : "outTileW", "Type" : "None", "Direction" : "I"},
			{"Name" : "stride", "Type" : "None", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_25_1", "PipelineType" : "no",
				"LoopDec" : {"FSMBitwidth" : "3", "FirstState" : "ap_ST_fsm_state2", "LastState" : ["ap_ST_fsm_state3"], "QuitState" : ["ap_ST_fsm_state2"], "PreState" : ["ap_ST_fsm_state1"], "PostState" : ["ap_ST_fsm_state1"], "OneDepthLoop" : "0", "OneStateBlock": ""}}]},
	{"ID" : "21", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94", "Parent" : "20", "Child" : ["22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69"],
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
	{"ID" : "22", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.fadd_32ns_32ns_32_4_full_dsp_1_U96", "Parent" : "21"},
	{"ID" : "23", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.fadd_32ns_32ns_32_4_full_dsp_1_U97", "Parent" : "21"},
	{"ID" : "24", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.fadd_32ns_32ns_32_4_full_dsp_1_U98", "Parent" : "21"},
	{"ID" : "25", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.fadd_32ns_32ns_32_4_full_dsp_1_U99", "Parent" : "21"},
	{"ID" : "26", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.fadd_32ns_32ns_32_4_full_dsp_1_U100", "Parent" : "21"},
	{"ID" : "27", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.fadd_32ns_32ns_32_4_full_dsp_1_U101", "Parent" : "21"},
	{"ID" : "28", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.fadd_32ns_32ns_32_4_full_dsp_1_U102", "Parent" : "21"},
	{"ID" : "29", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.fadd_32ns_32ns_32_4_full_dsp_1_U103", "Parent" : "21"},
	{"ID" : "30", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.fadd_32ns_32ns_32_4_full_dsp_1_U104", "Parent" : "21"},
	{"ID" : "31", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.fadd_32ns_32ns_32_4_full_dsp_1_U105", "Parent" : "21"},
	{"ID" : "32", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.fmul_32ns_32ns_32_3_max_dsp_1_U106", "Parent" : "21"},
	{"ID" : "33", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.fmul_32ns_32ns_32_3_max_dsp_1_U107", "Parent" : "21"},
	{"ID" : "34", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.fmul_32ns_32ns_32_3_max_dsp_1_U108", "Parent" : "21"},
	{"ID" : "35", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.fmul_32ns_32ns_32_3_max_dsp_1_U109", "Parent" : "21"},
	{"ID" : "36", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.fmul_32ns_32ns_32_3_max_dsp_1_U110", "Parent" : "21"},
	{"ID" : "37", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.fmul_32ns_32ns_32_3_max_dsp_1_U111", "Parent" : "21"},
	{"ID" : "38", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.fmul_32ns_32ns_32_3_max_dsp_1_U112", "Parent" : "21"},
	{"ID" : "39", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.fmul_32ns_32ns_32_3_max_dsp_1_U113", "Parent" : "21"},
	{"ID" : "40", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.fmul_32ns_32ns_32_3_max_dsp_1_U114", "Parent" : "21"},
	{"ID" : "41", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.mul_64ns_66ns_129_1_1_U115", "Parent" : "21"},
	{"ID" : "42", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.mul_64ns_66ns_129_1_1_U116", "Parent" : "21"},
	{"ID" : "43", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.mul_64ns_66ns_129_1_1_U117", "Parent" : "21"},
	{"ID" : "44", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.mul_64ns_66ns_129_1_1_U118", "Parent" : "21"},
	{"ID" : "45", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.mul_64ns_66ns_129_1_1_U119", "Parent" : "21"},
	{"ID" : "46", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.mul_64ns_66ns_129_1_1_U120", "Parent" : "21"},
	{"ID" : "47", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.sparsemux_7_2_32_1_1_U121", "Parent" : "21"},
	{"ID" : "48", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.sparsemux_7_2_32_1_1_U122", "Parent" : "21"},
	{"ID" : "49", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.sparsemux_7_2_32_1_1_U123", "Parent" : "21"},
	{"ID" : "50", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.sparsemux_7_2_32_1_1_U124", "Parent" : "21"},
	{"ID" : "51", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.sparsemux_7_2_32_1_1_U125", "Parent" : "21"},
	{"ID" : "52", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.sparsemux_7_2_32_1_1_U126", "Parent" : "21"},
	{"ID" : "53", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.sparsemux_7_2_32_1_1_U127", "Parent" : "21"},
	{"ID" : "54", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.sparsemux_7_2_32_1_1_U128", "Parent" : "21"},
	{"ID" : "55", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.sparsemux_7_2_32_1_1_U129", "Parent" : "21"},
	{"ID" : "56", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.mul_3ns_32s_35_1_1_U130", "Parent" : "21"},
	{"ID" : "57", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.urem_64s_3ns_2_68_1_U131", "Parent" : "21"},
	{"ID" : "58", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.mul_3ns_32s_35_1_1_U132", "Parent" : "21"},
	{"ID" : "59", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.urem_64s_3ns_2_68_1_U133", "Parent" : "21"},
	{"ID" : "60", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.sparsemux_7_2_32_1_1_U134", "Parent" : "21"},
	{"ID" : "61", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.sparsemux_7_2_32_1_1_U135", "Parent" : "21"},
	{"ID" : "62", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.sparsemux_7_2_32_1_1_U136", "Parent" : "21"},
	{"ID" : "63", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.sparsemux_7_2_32_1_1_U137", "Parent" : "21"},
	{"ID" : "64", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.sparsemux_7_2_32_1_1_U138", "Parent" : "21"},
	{"ID" : "65", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.sparsemux_7_2_32_1_1_U139", "Parent" : "21"},
	{"ID" : "66", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.sparsemux_7_2_32_1_1_U140", "Parent" : "21"},
	{"ID" : "67", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.sparsemux_7_2_32_1_1_U141", "Parent" : "21"},
	{"ID" : "68", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.sparsemux_7_2_32_1_1_U142", "Parent" : "21"},
	{"ID" : "69", "Level" : "3", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94.flow_control_loop_pipe_sequential_init_U", "Parent" : "21"},
	{"ID" : "70", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_kernel_fu_364.mul_32ns_32ns_64_1_1_U176", "Parent" : "20"},
	{"ID" : "71", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_conv_via_tiling_Pipeline_VITIS_LOOP_75_1_VITIS_LOOP_76_2_VITIS_LOOP_77_3_fu_419", "Parent" : "0", "Child" : ["72"],
		"CDFG" : "conv_via_tiling_Pipeline_VITIS_LOOP_75_1_VITIS_LOOP_76_2_VITIS_LOOP_77_3",
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
			{"Name" : "mem1", "Type" : "MAXI", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "mem1_blk_n_AR", "Type" : "RtlSignal"},
					{"Name" : "mem1_blk_n_R", "Type" : "RtlSignal"}]},
			{"Name" : "out_w", "Type" : "None", "Direction" : "I"},
			{"Name" : "mul_ln45_1", "Type" : "None", "Direction" : "I"},
			{"Name" : "out_h", "Type" : "None", "Direction" : "I"},
			{"Name" : "zext_ln75_1", "Type" : "None", "Direction" : "I"},
			{"Name" : "mul_ln45", "Type" : "None", "Direction" : "I"},
			{"Name" : "output_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "cmp764", "Type" : "None", "Direction" : "I"},
			{"Name" : "bias", "Type" : "None", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_75_1_VITIS_LOOP_76_2_VITIS_LOOP_77_3", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter16", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter16", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "72", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_via_tiling_Pipeline_VITIS_LOOP_75_1_VITIS_LOOP_76_2_VITIS_LOOP_77_3_fu_419.flow_control_loop_pipe_sequential_init_U", "Parent" : "71"},
	{"ID" : "73", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_conv_via_tiling_Pipeline_VITIS_LOOP_109_7_VITIS_LOOP_111_8_VITIS_LOOP_113_9_fu_435", "Parent" : "0", "Child" : ["74"],
		"CDFG" : "conv_via_tiling_Pipeline_VITIS_LOOP_109_7_VITIS_LOOP_111_8_VITIS_LOOP_113_9",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "3138", "EstimateLatencyMax" : "3138",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "localOut", "Type" : "Memory", "Direction" : "O"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_109_7_VITIS_LOOP_111_8_VITIS_LOOP_113_9", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_state1", "FirstStateIter" : "", "FirstStateBlock" : "ap_ST_fsm_state1_blk", "LastState" : "ap_ST_fsm_state1", "LastStateIter" : "", "LastStateBlock" : "ap_ST_fsm_state1_blk", "QuitState" : "ap_ST_fsm_state1", "QuitStateIter" : "", "QuitStateBlock" : "ap_ST_fsm_state1_blk", "OneDepthLoop" : "1", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "74", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_via_tiling_Pipeline_VITIS_LOOP_109_7_VITIS_LOOP_111_8_VITIS_LOOP_113_9_fu_435.flow_control_loop_pipe_sequential_init_U", "Parent" : "73"},
	{"ID" : "75", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_conv_via_tiling_Pipeline_VITIS_LOOP_119_10_VITIS_LOOP_120_11_fu_440", "Parent" : "0", "Child" : ["76", "77", "78", "79"],
		"CDFG" : "conv_via_tiling_Pipeline_VITIS_LOOP_119_10_VITIS_LOOP_120_11",
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
			{"Name" : "mem2", "Type" : "MAXI", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "mem2_blk_n_AR", "Type" : "RtlSignal"},
					{"Name" : "mem2_blk_n_R", "Type" : "RtlSignal"}]},
			{"Name" : "mul83", "Type" : "None", "Direction" : "I"},
			{"Name" : "mul_ln104", "Type" : "None", "Direction" : "I"},
			{"Name" : "zext_ln104", "Type" : "None", "Direction" : "I"},
			{"Name" : "tileW", "Type" : "None", "Direction" : "I"},
			{"Name" : "mul_ln90", "Type" : "None", "Direction" : "I"},
			{"Name" : "out_w", "Type" : "None", "Direction" : "I"},
			{"Name" : "zext_ln91_3", "Type" : "None", "Direction" : "I"},
			{"Name" : "output_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "localOut", "Type" : "Memory", "Direction" : "O"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_119_10_VITIS_LOOP_120_11", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter47", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter47", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "76", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_via_tiling_Pipeline_VITIS_LOOP_119_10_VITIS_LOOP_120_11_fu_440.mul_32s_32s_32_1_1_U12", "Parent" : "75"},
	{"ID" : "77", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_via_tiling_Pipeline_VITIS_LOOP_119_10_VITIS_LOOP_120_11_fu_440.sdiv_32ns_32ns_32_36_1_U13", "Parent" : "75"},
	{"ID" : "78", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_via_tiling_Pipeline_VITIS_LOOP_119_10_VITIS_LOOP_120_11_fu_440.srem_32ns_32ns_32_36_1_U14", "Parent" : "75"},
	{"ID" : "79", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_via_tiling_Pipeline_VITIS_LOOP_119_10_VITIS_LOOP_120_11_fu_440.flow_control_loop_pipe_sequential_init_U", "Parent" : "75"},
	{"ID" : "80", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_conv_via_tiling_Pipeline_VITIS_LOOP_136_13_VITIS_LOOP_138_14_VITIS_LOOP_140_15_V_fu_455", "Parent" : "0", "Child" : ["81"],
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
	{"ID" : "81", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_via_tiling_Pipeline_VITIS_LOOP_136_13_VITIS_LOOP_138_14_VITIS_LOOP_140_15_V_fu_455.flow_control_loop_pipe_sequential_init_U", "Parent" : "80"},
	{"ID" : "82", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_conv_via_tiling_Pipeline_VITIS_LOOP_164_21_VITIS_LOOP_166_22_VITIS_LOOP_168_23_fu_468", "Parent" : "0", "Child" : ["83", "84", "85", "86", "87"],
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
	{"ID" : "83", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_via_tiling_Pipeline_VITIS_LOOP_164_21_VITIS_LOOP_166_22_VITIS_LOOP_168_23_fu_468.mul_5ns_7ns_11_1_1_U53", "Parent" : "82"},
	{"ID" : "84", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_via_tiling_Pipeline_VITIS_LOOP_164_21_VITIS_LOOP_166_22_VITIS_LOOP_168_23_fu_468.urem_5ns_3ns_2_9_1_U54", "Parent" : "82"},
	{"ID" : "85", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_via_tiling_Pipeline_VITIS_LOOP_164_21_VITIS_LOOP_166_22_VITIS_LOOP_168_23_fu_468.mul_5ns_7ns_11_1_1_U55", "Parent" : "82"},
	{"ID" : "86", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_via_tiling_Pipeline_VITIS_LOOP_164_21_VITIS_LOOP_166_22_VITIS_LOOP_168_23_fu_468.urem_5ns_3ns_2_9_1_U56", "Parent" : "82"},
	{"ID" : "87", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_via_tiling_Pipeline_VITIS_LOOP_164_21_VITIS_LOOP_166_22_VITIS_LOOP_168_23_fu_468.flow_control_loop_pipe_sequential_init_U", "Parent" : "82"},
	{"ID" : "88", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_conv_via_tiling_Pipeline_VITIS_LOOP_150_17_VITIS_LOOP_151_18_VITIS_LOOP_152_19_V_fu_481", "Parent" : "0", "Child" : ["89"],
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
	{"ID" : "89", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_via_tiling_Pipeline_VITIS_LOOP_150_17_VITIS_LOOP_151_18_VITIS_LOOP_152_19_V_fu_481.flow_control_loop_pipe_sequential_init_U", "Parent" : "88"},
	{"ID" : "90", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_conv_via_tiling_Pipeline_VITIS_LOOP_174_24_VITIS_LOOP_175_25_VITIS_LOOP_176_26_fu_502", "Parent" : "0", "Child" : ["91", "92", "93", "94", "95", "96"],
		"CDFG" : "conv_via_tiling_Pipeline_VITIS_LOOP_174_24_VITIS_LOOP_175_25_VITIS_LOOP_176_26",
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
					{"Name" : "mem1_blk_n_AR", "Type" : "RtlSignal"},
					{"Name" : "mem1_blk_n_R", "Type" : "RtlSignal"}]},
			{"Name" : "add_ln91_1", "Type" : "None", "Direction" : "I"},
			{"Name" : "mul_ln174", "Type" : "None", "Direction" : "I"},
			{"Name" : "mul_ln91_2", "Type" : "None", "Direction" : "I"},
			{"Name" : "zext_ln174", "Type" : "None", "Direction" : "I"},
			{"Name" : "sext_ln91", "Type" : "None", "Direction" : "I"},
			{"Name" : "sext_ln90_2", "Type" : "None", "Direction" : "I"},
			{"Name" : "sext_ln90_3", "Type" : "None", "Direction" : "I"},
			{"Name" : "localIn", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localIn_1", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localIn_2", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localIn_3", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localIn_4", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localIn_5", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localIn_6", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localIn_7", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "localIn_8", "Type" : "Memory", "Direction" : "O"},
			{"Name" : "p_cast33", "Type" : "None", "Direction" : "I"},
			{"Name" : "sext_ln90_1", "Type" : "None", "Direction" : "I"},
			{"Name" : "input_r", "Type" : "None", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_174_24_VITIS_LOOP_175_25_VITIS_LOOP_176_26", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter35", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter35", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "91", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_via_tiling_Pipeline_VITIS_LOOP_174_24_VITIS_LOOP_175_25_VITIS_LOOP_176_26_fu_502.mul_31ns_33ns_63_1_1_U68", "Parent" : "90"},
	{"ID" : "92", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_via_tiling_Pipeline_VITIS_LOOP_174_24_VITIS_LOOP_175_25_VITIS_LOOP_176_26_fu_502.mul_31ns_33ns_63_1_1_U69", "Parent" : "90"},
	{"ID" : "93", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_via_tiling_Pipeline_VITIS_LOOP_174_24_VITIS_LOOP_175_25_VITIS_LOOP_176_26_fu_502.mul_33s_32s_62_1_1_U70", "Parent" : "90"},
	{"ID" : "94", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_via_tiling_Pipeline_VITIS_LOOP_174_24_VITIS_LOOP_175_25_VITIS_LOOP_176_26_fu_502.urem_31ns_3ns_2_35_1_U71", "Parent" : "90"},
	{"ID" : "95", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_via_tiling_Pipeline_VITIS_LOOP_174_24_VITIS_LOOP_175_25_VITIS_LOOP_176_26_fu_502.urem_31ns_3ns_2_35_1_U72", "Parent" : "90"},
	{"ID" : "96", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_via_tiling_Pipeline_VITIS_LOOP_174_24_VITIS_LOOP_175_25_VITIS_LOOP_176_26_fu_502.flow_control_loop_pipe_sequential_init_U", "Parent" : "90"},
	{"ID" : "97", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_conv_via_tiling_Pipeline_VITIS_LOOP_192_27_VITIS_LOOP_193_28_VITIS_LOOP_194_29_fu_527", "Parent" : "0", "Child" : ["98", "99"],
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
	{"ID" : "98", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_via_tiling_Pipeline_VITIS_LOOP_192_27_VITIS_LOOP_193_28_VITIS_LOOP_194_29_fu_527.mul_31ns_32s_62_1_1_U200", "Parent" : "97"},
	{"ID" : "99", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_conv_via_tiling_Pipeline_VITIS_LOOP_192_27_VITIS_LOOP_193_28_VITIS_LOOP_194_29_fu_527.flow_control_loop_pipe_sequential_init_U", "Parent" : "97"},
	{"ID" : "100", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.control_s_axi_U", "Parent" : "0"},
	{"ID" : "101", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mem1_m_axi_U", "Parent" : "0"},
	{"ID" : "102", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mem2_m_axi_U", "Parent" : "0"},
	{"ID" : "103", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_31ns_32s_32_1_1_U215", "Parent" : "0"},
	{"ID" : "104", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_32ns_32ns_63_1_1_U216", "Parent" : "0"},
	{"ID" : "105", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_32ns_32ns_64_1_1_U217", "Parent" : "0"},
	{"ID" : "106", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_32ns_32ns_64_1_1_U218", "Parent" : "0"},
	{"ID" : "107", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_32ns_36ns_68_1_1_U219", "Parent" : "0"},
	{"ID" : "108", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_32ns_64ns_96_1_1_U220", "Parent" : "0"},
	{"ID" : "109", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_32s_32s_32_1_1_U221", "Parent" : "0"},
	{"ID" : "110", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_32s_32s_32_1_1_U222", "Parent" : "0"},
	{"ID" : "111", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_32s_34ns_65_1_1_U223", "Parent" : "0"},
	{"ID" : "112", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mul_32s_34ns_65_1_1_U224", "Parent" : "0"},
	{"ID" : "113", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sdiv_32ns_32s_32_36_seq_1_U225", "Parent" : "0"},
	{"ID" : "114", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sdiv_32ns_32s_32_36_seq_1_U226", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	conv_via_tiling {
		mem1 {Type I LastRead 11 FirstWrite -1}
		mem2 {Type IO LastRead 46 FirstWrite -1}
		in_ch {Type I LastRead 0 FirstWrite -1}
		out_ch {Type I LastRead 0 FirstWrite -1}
		H {Type I LastRead 0 FirstWrite -1}
		W {Type I LastRead 0 FirstWrite -1}
		input_r {Type I LastRead 0 FirstWrite -1}
		weight {Type I LastRead 0 FirstWrite -1}
		bias {Type I LastRead 0 FirstWrite -1}
		output_r {Type I LastRead 0 FirstWrite -1}
		stride {Type I LastRead 0 FirstWrite -1}
		pad {Type I LastRead 0 FirstWrite -1}}
	conv_kernel {
		inData_0_0 {Type I LastRead 71 FirstWrite -1}
		inData_0_1 {Type I LastRead 71 FirstWrite -1}
		inData_0_2 {Type I LastRead 71 FirstWrite -1}
		inData_1_0 {Type I LastRead 71 FirstWrite -1}
		inData_1_1 {Type I LastRead 71 FirstWrite -1}
		inData_1_2 {Type I LastRead 71 FirstWrite -1}
		inData_2_0 {Type I LastRead 71 FirstWrite -1}
		inData_2_1 {Type I LastRead 71 FirstWrite -1}
		inData_2_2 {Type I LastRead 71 FirstWrite -1}
		weight_0_0 {Type I LastRead 1 FirstWrite -1}
		weight_0_1 {Type I LastRead 1 FirstWrite -1}
		weight_0_2 {Type I LastRead 1 FirstWrite -1}
		weight_1_0 {Type I LastRead 1 FirstWrite -1}
		weight_1_1 {Type I LastRead 1 FirstWrite -1}
		weight_1_2 {Type I LastRead 1 FirstWrite -1}
		weight_2_0 {Type I LastRead 1 FirstWrite -1}
		weight_2_1 {Type I LastRead 1 FirstWrite -1}
		weight_2_2 {Type I LastRead 1 FirstWrite -1}
		outData {Type IO LastRead 1 FirstWrite 115}
		outTileH {Type I LastRead 0 FirstWrite -1}
		outTileW {Type I LastRead 0 FirstWrite -1}
		stride {Type I LastRead 0 FirstWrite -1}}
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
		inData_2_2 {Type I LastRead 71 FirstWrite -1}}
	conv_via_tiling_Pipeline_VITIS_LOOP_75_1_VITIS_LOOP_76_2_VITIS_LOOP_77_3 {
		mem2 {Type O LastRead 12 FirstWrite 11}
		mem1 {Type I LastRead 10 FirstWrite -1}
		out_w {Type I LastRead 0 FirstWrite -1}
		mul_ln45_1 {Type I LastRead 0 FirstWrite -1}
		out_h {Type I LastRead 0 FirstWrite -1}
		zext_ln75_1 {Type I LastRead 0 FirstWrite -1}
		mul_ln45 {Type I LastRead 0 FirstWrite -1}
		output_r {Type I LastRead 0 FirstWrite -1}
		cmp764 {Type I LastRead 0 FirstWrite -1}
		bias {Type I LastRead 0 FirstWrite -1}}
	conv_via_tiling_Pipeline_VITIS_LOOP_109_7_VITIS_LOOP_111_8_VITIS_LOOP_113_9 {
		localOut {Type O LastRead -1 FirstWrite 0}}
	conv_via_tiling_Pipeline_VITIS_LOOP_119_10_VITIS_LOOP_120_11 {
		mem2 {Type I LastRead 46 FirstWrite -1}
		mul83 {Type I LastRead 0 FirstWrite -1}
		mul_ln104 {Type I LastRead 0 FirstWrite -1}
		zext_ln104 {Type I LastRead 0 FirstWrite -1}
		tileW {Type I LastRead 0 FirstWrite -1}
		mul_ln90 {Type I LastRead 0 FirstWrite -1}
		out_w {Type I LastRead 0 FirstWrite -1}
		zext_ln91_3 {Type I LastRead 0 FirstWrite -1}
		output_r {Type I LastRead 0 FirstWrite -1}
		localOut {Type O LastRead -1 FirstWrite 47}}
	conv_via_tiling_Pipeline_VITIS_LOOP_136_13_VITIS_LOOP_138_14_VITIS_LOOP_140_15_V {
		localW {Type O LastRead -1 FirstWrite 1}
		localW_1 {Type O LastRead -1 FirstWrite 1}
		localW_2 {Type O LastRead -1 FirstWrite 1}
		localW_3 {Type O LastRead -1 FirstWrite 1}
		localW_4 {Type O LastRead -1 FirstWrite 1}
		localW_5 {Type O LastRead -1 FirstWrite 1}
		localW_6 {Type O LastRead -1 FirstWrite 1}
		localW_7 {Type O LastRead -1 FirstWrite 1}
		localW_8 {Type O LastRead -1 FirstWrite 1}}
	conv_via_tiling_Pipeline_VITIS_LOOP_164_21_VITIS_LOOP_166_22_VITIS_LOOP_168_23 {
		localIn {Type O LastRead -1 FirstWrite 9}
		localIn_1 {Type O LastRead -1 FirstWrite 9}
		localIn_2 {Type O LastRead -1 FirstWrite 9}
		localIn_3 {Type O LastRead -1 FirstWrite 9}
		localIn_4 {Type O LastRead -1 FirstWrite 9}
		localIn_5 {Type O LastRead -1 FirstWrite 9}
		localIn_6 {Type O LastRead -1 FirstWrite 9}
		localIn_7 {Type O LastRead -1 FirstWrite 9}
		localIn_8 {Type O LastRead -1 FirstWrite 9}}
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
		p_cast44 {Type I LastRead 0 FirstWrite -1}}
	conv_via_tiling_Pipeline_VITIS_LOOP_174_24_VITIS_LOOP_175_25_VITIS_LOOP_176_26 {
		mem1 {Type I LastRead 11 FirstWrite -1}
		add_ln91_1 {Type I LastRead 0 FirstWrite -1}
		mul_ln174 {Type I LastRead 0 FirstWrite -1}
		mul_ln91_2 {Type I LastRead 0 FirstWrite -1}
		zext_ln174 {Type I LastRead 0 FirstWrite -1}
		sext_ln91 {Type I LastRead 0 FirstWrite -1}
		sext_ln90_2 {Type I LastRead 0 FirstWrite -1}
		sext_ln90_3 {Type I LastRead 0 FirstWrite -1}
		localIn {Type O LastRead -1 FirstWrite 35}
		localIn_1 {Type O LastRead -1 FirstWrite 35}
		localIn_2 {Type O LastRead -1 FirstWrite 35}
		localIn_3 {Type O LastRead -1 FirstWrite 35}
		localIn_4 {Type O LastRead -1 FirstWrite 35}
		localIn_5 {Type O LastRead -1 FirstWrite 35}
		localIn_6 {Type O LastRead -1 FirstWrite 35}
		localIn_7 {Type O LastRead -1 FirstWrite 35}
		localIn_8 {Type O LastRead -1 FirstWrite 35}
		p_cast33 {Type I LastRead 0 FirstWrite -1}
		sext_ln90_1 {Type I LastRead 0 FirstWrite -1}
		input_r {Type I LastRead 0 FirstWrite -1}}
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
	, {"Name" : "Interval", "Min" : "0", "Max" : "0"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	mem1 { m_axi {  { m_axi_mem1_AWVALID VALID 1 1 }  { m_axi_mem1_AWREADY READY 0 1 }  { m_axi_mem1_AWADDR ADDR 1 64 }  { m_axi_mem1_AWID ID 1 1 }  { m_axi_mem1_AWLEN SIZE 1 8 }  { m_axi_mem1_AWSIZE BURST 1 3 }  { m_axi_mem1_AWBURST LOCK 1 2 }  { m_axi_mem1_AWLOCK CACHE 1 2 }  { m_axi_mem1_AWCACHE PROT 1 4 }  { m_axi_mem1_AWPROT QOS 1 3 }  { m_axi_mem1_AWQOS REGION 1 4 }  { m_axi_mem1_AWREGION USER 1 4 }  { m_axi_mem1_AWUSER DATA 1 1 }  { m_axi_mem1_WVALID VALID 1 1 }  { m_axi_mem1_WREADY READY 0 1 }  { m_axi_mem1_WDATA FIFONUM 1 32 }  { m_axi_mem1_WSTRB STRB 1 4 }  { m_axi_mem1_WLAST LAST 1 1 }  { m_axi_mem1_WID ID 1 1 }  { m_axi_mem1_WUSER DATA 1 1 }  { m_axi_mem1_ARVALID VALID 1 1 }  { m_axi_mem1_ARREADY READY 0 1 }  { m_axi_mem1_ARADDR ADDR 1 64 }  { m_axi_mem1_ARID ID 1 1 }  { m_axi_mem1_ARLEN SIZE 1 8 }  { m_axi_mem1_ARSIZE BURST 1 3 }  { m_axi_mem1_ARBURST LOCK 1 2 }  { m_axi_mem1_ARLOCK CACHE 1 2 }  { m_axi_mem1_ARCACHE PROT 1 4 }  { m_axi_mem1_ARPROT QOS 1 3 }  { m_axi_mem1_ARQOS REGION 1 4 }  { m_axi_mem1_ARREGION USER 1 4 }  { m_axi_mem1_ARUSER DATA 1 1 }  { m_axi_mem1_RVALID VALID 0 1 }  { m_axi_mem1_RREADY READY 1 1 }  { m_axi_mem1_RDATA FIFONUM 0 32 }  { m_axi_mem1_RLAST LAST 0 1 }  { m_axi_mem1_RID ID 0 1 }  { m_axi_mem1_RUSER DATA 0 1 }  { m_axi_mem1_RRESP RESP 0 2 }  { m_axi_mem1_BVALID VALID 0 1 }  { m_axi_mem1_BREADY READY 1 1 }  { m_axi_mem1_BRESP RESP 0 2 }  { m_axi_mem1_BID ID 0 1 }  { m_axi_mem1_BUSER DATA 0 1 } } }
	mem2 { m_axi {  { m_axi_mem2_AWVALID VALID 1 1 }  { m_axi_mem2_AWREADY READY 0 1 }  { m_axi_mem2_AWADDR ADDR 1 64 }  { m_axi_mem2_AWID ID 1 1 }  { m_axi_mem2_AWLEN SIZE 1 8 }  { m_axi_mem2_AWSIZE BURST 1 3 }  { m_axi_mem2_AWBURST LOCK 1 2 }  { m_axi_mem2_AWLOCK CACHE 1 2 }  { m_axi_mem2_AWCACHE PROT 1 4 }  { m_axi_mem2_AWPROT QOS 1 3 }  { m_axi_mem2_AWQOS REGION 1 4 }  { m_axi_mem2_AWREGION USER 1 4 }  { m_axi_mem2_AWUSER DATA 1 1 }  { m_axi_mem2_WVALID VALID 1 1 }  { m_axi_mem2_WREADY READY 0 1 }  { m_axi_mem2_WDATA FIFONUM 1 32 }  { m_axi_mem2_WSTRB STRB 1 4 }  { m_axi_mem2_WLAST LAST 1 1 }  { m_axi_mem2_WID ID 1 1 }  { m_axi_mem2_WUSER DATA 1 1 }  { m_axi_mem2_ARVALID VALID 1 1 }  { m_axi_mem2_ARREADY READY 0 1 }  { m_axi_mem2_ARADDR ADDR 1 64 }  { m_axi_mem2_ARID ID 1 1 }  { m_axi_mem2_ARLEN SIZE 1 8 }  { m_axi_mem2_ARSIZE BURST 1 3 }  { m_axi_mem2_ARBURST LOCK 1 2 }  { m_axi_mem2_ARLOCK CACHE 1 2 }  { m_axi_mem2_ARCACHE PROT 1 4 }  { m_axi_mem2_ARPROT QOS 1 3 }  { m_axi_mem2_ARQOS REGION 1 4 }  { m_axi_mem2_ARREGION USER 1 4 }  { m_axi_mem2_ARUSER DATA 1 1 }  { m_axi_mem2_RVALID VALID 0 1 }  { m_axi_mem2_RREADY READY 1 1 }  { m_axi_mem2_RDATA FIFONUM 0 32 }  { m_axi_mem2_RLAST LAST 0 1 }  { m_axi_mem2_RID ID 0 1 }  { m_axi_mem2_RUSER DATA 0 1 }  { m_axi_mem2_RRESP RESP 0 2 }  { m_axi_mem2_BVALID VALID 0 1 }  { m_axi_mem2_BREADY READY 1 1 }  { m_axi_mem2_BRESP RESP 0 2 }  { m_axi_mem2_BID ID 0 1 }  { m_axi_mem2_BUSER DATA 0 1 } } }
}

set maxi_interface_dict [dict create]
dict set maxi_interface_dict mem1 { CHANNEL_NUM 0 BUNDLE mem1 NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 READ_WRITE_MODE READ_ONLY}
dict set maxi_interface_dict mem2 { CHANNEL_NUM 0 BUNDLE mem2 NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 READ_WRITE_MODE READ_WRITE}

# RTL port scheduling information:
set fifoSchedulingInfoList { 
}

# RTL bus port read request latency information:
set busReadReqLatencyList { 
	{ mem1 1 }
	{ mem2 1 }
}

# RTL bus port write response latency information:
set busWriteResLatencyList { 
	{ mem1 1 }
	{ mem2 1 }
}

# RTL array port load latency information:
set memoryLoadLatencyList { 
}
