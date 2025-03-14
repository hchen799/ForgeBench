set ModuleHierarchy {[{
"Name" : "conv_via_tiling","ID" : "0","Type" : "sequential",
"SubInsts" : [
	{"Name" : "grp_conv_via_tiling_Pipeline_VITIS_LOOP_75_1_VITIS_LOOP_76_2_VITIS_LOOP_77_3_fu_419","ID" : "1","Type" : "sequential",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_75_1_VITIS_LOOP_76_2_VITIS_LOOP_77_3","ID" : "2","Type" : "pipeline"},]},],
"SubLoops" : [
	{"Name" : "VITIS_LOOP_90_4","ID" : "3","Type" : "no",
	"SubLoops" : [
	{"Name" : "VITIS_LOOP_91_5","ID" : "4","Type" : "no",
		"SubLoops" : [
		{"Name" : "VITIS_LOOP_102_6","ID" : "5","Type" : "no",
			"SubInsts" : [
			{"Name" : "grp_conv_via_tiling_Pipeline_VITIS_LOOP_109_7_VITIS_LOOP_111_8_VITIS_LOOP_113_9_fu_435","ID" : "6","Type" : "sequential",
					"SubLoops" : [
					{"Name" : "VITIS_LOOP_109_7_VITIS_LOOP_111_8_VITIS_LOOP_113_9","ID" : "7","Type" : "pipeline"},]},
			{"Name" : "grp_conv_via_tiling_Pipeline_VITIS_LOOP_119_10_VITIS_LOOP_120_11_fu_440","ID" : "8","Type" : "sequential",
					"SubLoops" : [
					{"Name" : "VITIS_LOOP_119_10_VITIS_LOOP_120_11","ID" : "9","Type" : "pipeline"},]},
			{"Name" : "grp_conv_via_tiling_Pipeline_VITIS_LOOP_192_27_VITIS_LOOP_193_28_VITIS_LOOP_194_29_fu_527","ID" : "10","Type" : "sequential",
					"SubLoops" : [
					{"Name" : "VITIS_LOOP_192_27_VITIS_LOOP_193_28_VITIS_LOOP_194_29","ID" : "11","Type" : "pipeline"},]},],
			"SubLoops" : [
			{"Name" : "VITIS_LOOP_129_12","ID" : "12","Type" : "no",
				"SubInsts" : [
				{"Name" : "grp_conv_kernel_fu_364","ID" : "13","Type" : "sequential",
						"SubLoops" : [
						{"Name" : "VITIS_LOOP_25_1","ID" : "14","Type" : "no",
						"SubInsts" : [
						{"Name" : "grp_conv_kernel_Pipeline_VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4_fu_94","ID" : "15","Type" : "sequential",
								"SubLoops" : [
								{"Name" : "VITIS_LOOP_26_2_VITIS_LOOP_27_3_VITIS_LOOP_28_4","ID" : "16","Type" : "pipeline"},]},]},]},
				{"Name" : "grp_conv_via_tiling_Pipeline_VITIS_LOOP_136_13_VITIS_LOOP_138_14_VITIS_LOOP_140_15_V_fu_455","ID" : "17","Type" : "sequential",
						"SubLoops" : [
						{"Name" : "VITIS_LOOP_136_13_VITIS_LOOP_138_14_VITIS_LOOP_140_15_VITIS_LOOP_142_16","ID" : "18","Type" : "pipeline"},]},
				{"Name" : "grp_conv_via_tiling_Pipeline_VITIS_LOOP_164_21_VITIS_LOOP_166_22_VITIS_LOOP_168_23_fu_468","ID" : "19","Type" : "sequential",
						"SubLoops" : [
						{"Name" : "VITIS_LOOP_164_21_VITIS_LOOP_166_22_VITIS_LOOP_168_23","ID" : "20","Type" : "pipeline"},]},
				{"Name" : "grp_conv_via_tiling_Pipeline_VITIS_LOOP_150_17_VITIS_LOOP_151_18_VITIS_LOOP_152_19_V_fu_481","ID" : "21","Type" : "sequential",
						"SubLoops" : [
						{"Name" : "VITIS_LOOP_150_17_VITIS_LOOP_151_18_VITIS_LOOP_152_19_VITIS_LOOP_153_20","ID" : "22","Type" : "pipeline"},]},
				{"Name" : "grp_conv_via_tiling_Pipeline_VITIS_LOOP_174_24_VITIS_LOOP_175_25_VITIS_LOOP_176_26_fu_502","ID" : "23","Type" : "sequential",
						"SubLoops" : [
						{"Name" : "VITIS_LOOP_174_24_VITIS_LOOP_175_25_VITIS_LOOP_176_26","ID" : "24","Type" : "pipeline"},]},]},]},]},]},]
}]}