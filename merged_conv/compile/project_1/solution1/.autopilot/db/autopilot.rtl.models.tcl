set SynModuleInfo {
  {SRCNAME conv_via_tiling_Pipeline_VITIS_LOOP_75_1_VITIS_LOOP_76_2_VITIS_LOOP_77_3 MODELNAME conv_via_tiling_Pipeline_VITIS_LOOP_75_1_VITIS_LOOP_76_2_VITIS_LOOP_77_3 RTLNAME conv_via_tiling_conv_via_tiling_Pipeline_VITIS_LOOP_75_1_VITIS_LOOP_76_2_VITIS_LOOP_77_3
    SUBMODULES {
      {MODELNAME conv_via_tiling_flow_control_loop_pipe_sequential_init RTLNAME conv_via_tiling_flow_control_loop_pipe_sequential_init BINDTYPE interface TYPE internal_upc_flow_control INSTNAME conv_via_tiling_flow_control_loop_pipe_sequential_init_U}
    }
  }
  {SRCNAME conv_via_tiling_Pipeline_VITIS_LOOP_109_7_VITIS_LOOP_111_8_VITIS_LOOP_113_9 MODELNAME conv_via_tiling_Pipeline_VITIS_LOOP_109_7_VITIS_LOOP_111_8_VITIS_LOOP_113_9 RTLNAME conv_via_tiling_conv_via_tiling_Pipeline_VITIS_LOOP_109_7_VITIS_LOOP_111_8_VITIS_LOOP_113_9}
  {SRCNAME conv_via_tiling_Pipeline_VITIS_LOOP_119_10_VITIS_LOOP_120_11 MODELNAME conv_via_tiling_Pipeline_VITIS_LOOP_119_10_VITIS_LOOP_120_11 RTLNAME conv_via_tiling_conv_via_tiling_Pipeline_VITIS_LOOP_119_10_VITIS_LOOP_120_11
    SUBMODULES {
      {MODELNAME conv_via_tiling_mul_32s_32s_32_1_1 RTLNAME conv_via_tiling_mul_32s_32s_32_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME conv_via_tiling_sdiv_32ns_32ns_32_36_1 RTLNAME conv_via_tiling_sdiv_32ns_32ns_32_36_1 BINDTYPE op TYPE sdiv IMPL auto LATENCY 35 ALLOW_PRAGMA 1}
      {MODELNAME conv_via_tiling_srem_32ns_32ns_32_36_1 RTLNAME conv_via_tiling_srem_32ns_32ns_32_36_1 BINDTYPE op TYPE srem IMPL auto LATENCY 35 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME conv_via_tiling_Pipeline_VITIS_LOOP_136_13_VITIS_LOOP_138_14_VITIS_LOOP_140_15_V MODELNAME conv_via_tiling_Pipeline_VITIS_LOOP_136_13_VITIS_LOOP_138_14_VITIS_LOOP_140_15_V RTLNAME conv_via_tiling_conv_via_tiling_Pipeline_VITIS_LOOP_136_13_VITIS_LOOP_138_14_VITIS_LOOP_140_15_V}
  {SRCNAME conv_via_tiling_Pipeline_VITIS_LOOP_150_17_VITIS_LOOP_151_18_VITIS_LOOP_152_19_V MODELNAME conv_via_tiling_Pipeline_VITIS_LOOP_150_17_VITIS_LOOP_151_18_VITIS_LOOP_152_19_V RTLNAME conv_via_tiling_conv_via_tiling_Pipeline_VITIS_LOOP_150_17_VITIS_LOOP_151_18_VITIS_LOOP_152_19_V}
  {SRCNAME conv_via_tiling_Pipeline_VITIS_LOOP_164_21_VITIS_LOOP_166_22_VITIS_LOOP_168_23 MODELNAME conv_via_tiling_Pipeline_VITIS_LOOP_164_21_VITIS_LOOP_166_22_VITIS_LOOP_168_23 RTLNAME conv_via_tiling_conv_via_tiling_Pipeline_VITIS_LOOP_164_21_VITIS_LOOP_166_22_VITIS_LOOP_168_23
    SUBMODULES {
      {MODELNAME conv_via_tiling_mul_5ns_7ns_11_1_1 RTLNAME conv_via_tiling_mul_5ns_7ns_11_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME conv_via_tiling_urem_5ns_3ns_2_9_1 RTLNAME conv_via_tiling_urem_5ns_3ns_2_9_1 BINDTYPE op TYPE urem IMPL auto LATENCY 8 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME conv_via_tiling_Pipeline_VITIS_LOOP_174_24_VITIS_LOOP_175_25_VITIS_LOOP_176_26 MODELNAME conv_via_tiling_Pipeline_VITIS_LOOP_174_24_VITIS_LOOP_175_25_VITIS_LOOP_176_26 RTLNAME conv_via_tiling_conv_via_tiling_Pipeline_VITIS_LOOP_174_24_VITIS_LOOP_175_25_VITIS_LOOP_176_26
    SUBMODULES {
      {MODELNAME conv_via_tiling_mul_31ns_33ns_63_1_1 RTLNAME conv_via_tiling_mul_31ns_33ns_63_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME conv_via_tiling_mul_33s_32s_62_1_1 RTLNAME conv_via_tiling_mul_33s_32s_62_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME conv_via_tiling_urem_31ns_3ns_2_35_1 RTLNAME conv_via_tiling_urem_31ns_3ns_2_35_1 BINDTYPE op TYPE urem IMPL auto LATENCY 34 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4 MODELNAME conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4 RTLNAME conv_via_tiling_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4
    SUBMODULES {
      {MODELNAME conv_via_tiling_fadd_32ns_32ns_32_4_full_dsp_1 RTLNAME conv_via_tiling_fadd_32ns_32ns_32_4_full_dsp_1 BINDTYPE op TYPE fadd IMPL fulldsp LATENCY 3 ALLOW_PRAGMA 1}
      {MODELNAME conv_via_tiling_fmul_32ns_32ns_32_3_max_dsp_1 RTLNAME conv_via_tiling_fmul_32ns_32ns_32_3_max_dsp_1 BINDTYPE op TYPE fmul IMPL maxdsp LATENCY 2 ALLOW_PRAGMA 1}
      {MODELNAME conv_via_tiling_mul_64ns_66ns_129_1_1 RTLNAME conv_via_tiling_mul_64ns_66ns_129_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME conv_via_tiling_sparsemux_7_2_32_1_1 RTLNAME conv_via_tiling_sparsemux_7_2_32_1_1 BINDTYPE op TYPE sparsemux IMPL auto}
      {MODELNAME conv_via_tiling_mul_3ns_32s_35_1_1 RTLNAME conv_via_tiling_mul_3ns_32s_35_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME conv_via_tiling_urem_64s_3ns_2_68_1 RTLNAME conv_via_tiling_urem_64s_3ns_2_68_1 BINDTYPE op TYPE urem IMPL auto LATENCY 67 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME conv_kernel MODELNAME conv_kernel RTLNAME conv_via_tiling_conv_kernel
    SUBMODULES {
      {MODELNAME conv_via_tiling_mul_32ns_32ns_64_1_1 RTLNAME conv_via_tiling_mul_32ns_32ns_64_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME conv_via_tiling_Pipeline_VITIS_LOOP_192_27_VITIS_LOOP_193_28_VITIS_LOOP_194_29 MODELNAME conv_via_tiling_Pipeline_VITIS_LOOP_192_27_VITIS_LOOP_193_28_VITIS_LOOP_194_29 RTLNAME conv_via_tiling_conv_via_tiling_Pipeline_VITIS_LOOP_192_27_VITIS_LOOP_193_28_VITIS_LOOP_194_29
    SUBMODULES {
      {MODELNAME conv_via_tiling_mul_31ns_32s_62_1_1 RTLNAME conv_via_tiling_mul_31ns_32s_62_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
    }
  }
  {SRCNAME conv_via_tiling MODELNAME conv_via_tiling RTLNAME conv_via_tiling IS_TOP 1
    SUBMODULES {
      {MODELNAME conv_via_tiling_mul_31ns_32s_32_1_1 RTLNAME conv_via_tiling_mul_31ns_32s_32_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME conv_via_tiling_mul_32ns_32ns_63_1_1 RTLNAME conv_via_tiling_mul_32ns_32ns_63_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME conv_via_tiling_mul_32ns_36ns_68_1_1 RTLNAME conv_via_tiling_mul_32ns_36ns_68_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME conv_via_tiling_mul_32ns_64ns_96_1_1 RTLNAME conv_via_tiling_mul_32ns_64ns_96_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME conv_via_tiling_mul_32s_34ns_65_1_1 RTLNAME conv_via_tiling_mul_32s_34ns_65_1_1 BINDTYPE op TYPE mul IMPL auto LATENCY 0 ALLOW_PRAGMA 1}
      {MODELNAME conv_via_tiling_sdiv_32ns_32s_32_36_seq_1 RTLNAME conv_via_tiling_sdiv_32ns_32s_32_36_seq_1 BINDTYPE op TYPE sdiv IMPL auto_seq LATENCY 35 ALLOW_PRAGMA 1}
      {MODELNAME conv_via_tiling_localOut_RAM_AUTO_1R1W RTLNAME conv_via_tiling_localOut_RAM_AUTO_1R1W BINDTYPE storage TYPE ram IMPL auto LATENCY 2 ALLOW_PRAGMA 1}
      {MODELNAME conv_via_tiling_localW_RAM_AUTO_1R1W RTLNAME conv_via_tiling_localW_RAM_AUTO_1R1W BINDTYPE storage TYPE ram IMPL auto LATENCY 2 ALLOW_PRAGMA 1}
      {MODELNAME conv_via_tiling_mem1_m_axi RTLNAME conv_via_tiling_mem1_m_axi BINDTYPE interface TYPE adapter IMPL m_axi}
      {MODELNAME conv_via_tiling_mem2_m_axi RTLNAME conv_via_tiling_mem2_m_axi BINDTYPE interface TYPE adapter IMPL m_axi}
      {MODELNAME conv_via_tiling_control_s_axi RTLNAME conv_via_tiling_control_s_axi BINDTYPE interface TYPE interface_s_axilite}
    }
  }
}
