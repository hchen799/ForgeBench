# This script segment is generated automatically by AutoPilot

set name conv_via_tiling_mul_31ns_32s_62_1_1
if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler $name BINDTYPE {op} TYPE {mul} IMPL {auto} LATENCY 0 ALLOW_PRAGMA 1
}


# clear list
if {${::AESL::PGuard_autoexp_gen}} {
    cg_default_interface_gen_dc_begin
    cg_default_interface_gen_bundle_begin
    AESL_LIB_XILADAPTER::native_axis_begin
}

# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 210 \
    name localOut \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename localOut \
    op interface \
    ports { localOut_address0 { O 12 vector } localOut_ce0 { O 1 bit } localOut_q0 { I 32 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'localOut'"
}
}


# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 202 \
    name mem2 \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_mem2 \
    op interface \
    ports { m_axi_mem2_AWVALID { O 1 bit } m_axi_mem2_AWREADY { I 1 bit } m_axi_mem2_AWADDR { O 64 vector } m_axi_mem2_AWID { O 1 vector } m_axi_mem2_AWLEN { O 32 vector } m_axi_mem2_AWSIZE { O 3 vector } m_axi_mem2_AWBURST { O 2 vector } m_axi_mem2_AWLOCK { O 2 vector } m_axi_mem2_AWCACHE { O 4 vector } m_axi_mem2_AWPROT { O 3 vector } m_axi_mem2_AWQOS { O 4 vector } m_axi_mem2_AWREGION { O 4 vector } m_axi_mem2_AWUSER { O 1 vector } m_axi_mem2_WVALID { O 1 bit } m_axi_mem2_WREADY { I 1 bit } m_axi_mem2_WDATA { O 32 vector } m_axi_mem2_WSTRB { O 4 vector } m_axi_mem2_WLAST { O 1 bit } m_axi_mem2_WID { O 1 vector } m_axi_mem2_WUSER { O 1 vector } m_axi_mem2_ARVALID { O 1 bit } m_axi_mem2_ARREADY { I 1 bit } m_axi_mem2_ARADDR { O 64 vector } m_axi_mem2_ARID { O 1 vector } m_axi_mem2_ARLEN { O 32 vector } m_axi_mem2_ARSIZE { O 3 vector } m_axi_mem2_ARBURST { O 2 vector } m_axi_mem2_ARLOCK { O 2 vector } m_axi_mem2_ARCACHE { O 4 vector } m_axi_mem2_ARPROT { O 3 vector } m_axi_mem2_ARQOS { O 4 vector } m_axi_mem2_ARREGION { O 4 vector } m_axi_mem2_ARUSER { O 1 vector } m_axi_mem2_RVALID { I 1 bit } m_axi_mem2_RREADY { O 1 bit } m_axi_mem2_RDATA { I 32 vector } m_axi_mem2_RLAST { I 1 bit } m_axi_mem2_RID { I 1 vector } m_axi_mem2_RFIFONUM { I 9 vector } m_axi_mem2_RUSER { I 1 vector } m_axi_mem2_RRESP { I 2 vector } m_axi_mem2_BVALID { I 1 bit } m_axi_mem2_BREADY { O 1 bit } m_axi_mem2_BRESP { I 2 vector } m_axi_mem2_BID { I 1 vector } m_axi_mem2_BUSER { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 203 \
    name zext_ln104 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_zext_ln104 \
    op interface \
    ports { zext_ln104 { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 204 \
    name mul_ln90_2 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_mul_ln90_2 \
    op interface \
    ports { mul_ln90_2 { I 31 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 205 \
    name sext_ln75 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln75 \
    op interface \
    ports { sext_ln75 { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 206 \
    name zext_ln91_2 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_zext_ln91_2 \
    op interface \
    ports { zext_ln91_2 { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 207 \
    name output_r \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_output_r \
    op interface \
    ports { output_r { I 64 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 208 \
    name xor_ln91 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_xor_ln91 \
    op interface \
    ports { xor_ln91 { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 209 \
    name mul_ln192 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_mul_ln192 \
    op interface \
    ports { mul_ln192 { I 96 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 211 \
    name zext_ln102 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_zext_ln102 \
    op interface \
    ports { zext_ln102 { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 212 \
    name mul_ln91_3 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_mul_ln91_3 \
    op interface \
    ports { mul_ln91_3 { I 64 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 213 \
    name tmp_cast_mid1257 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_tmp_cast_mid1257 \
    op interface \
    ports { tmp_cast_mid1257 { I 31 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 214 \
    name select_ln91 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_select_ln91 \
    op interface \
    ports { select_ln91 { I 32 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id -1 \
    name ap_ctrl \
    type ap_ctrl \
    reset_level 1 \
    sync_rst true \
    corename ap_ctrl \
    op interface \
    ports { ap_start { I 1 bit } ap_ready { O 1 bit } ap_done { O 1 bit } ap_idle { O 1 bit } } \
} "
}


# Adapter definition:
set PortName ap_clk
set DataWd 1 
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc cg_default_interface_gen_clock] == "cg_default_interface_gen_clock"} {
eval "cg_default_interface_gen_clock { \
    id -2 \
    name ${PortName} \
    reset_level 1 \
    sync_rst true \
    corename apif_ap_clk \
    data_wd ${DataWd} \
    op interface \
}"
} else {
puts "@W \[IMPL-113\] Cannot find bus interface model in the library. Ignored generation of bus interface for '${PortName}'"
}
}


# Adapter definition:
set PortName ap_rst
set DataWd 1 
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc cg_default_interface_gen_reset] == "cg_default_interface_gen_reset"} {
eval "cg_default_interface_gen_reset { \
    id -3 \
    name ${PortName} \
    reset_level 1 \
    sync_rst true \
    corename apif_ap_rst \
    data_wd ${DataWd} \
    op interface \
}"
} else {
puts "@W \[IMPL-114\] Cannot find bus interface model in the library. Ignored generation of bus interface for '${PortName}'"
}
}



# merge
if {${::AESL::PGuard_autoexp_gen}} {
    cg_default_interface_gen_dc_end
    cg_default_interface_gen_bundle_end
    AESL_LIB_XILADAPTER::native_axis_end
}


# flow_control definition:
set InstName conv_via_tiling_flow_control_loop_pipe_sequential_init_U
set CompName conv_via_tiling_flow_control_loop_pipe_sequential_init
set name flow_control_loop_pipe_sequential_init
if {${::AESL::PGuard_autocg_gen} && ${::AESL::PGuard_autocg_ipmgen}} {
if {[info proc ::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control] == "::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control"} {
eval "::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control { \
    name ${name} \
    prefix conv_via_tiling_ \
}"
} else {
puts "@W \[IMPL-107\] Cannot find ::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control, check your platform lib"
}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler $CompName BINDTYPE interface TYPE internal_upc_flow_control INSTNAME $InstName
}


