// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.1.2 (64-bit)
// Tool Version Limit: 2024.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#ifdef SDT
#include "xparameters.h"
#endif
#include "xconv_via_tiling.h"

extern XConv_via_tiling_Config XConv_via_tiling_ConfigTable[];

#ifdef SDT
XConv_via_tiling_Config *XConv_via_tiling_LookupConfig(UINTPTR BaseAddress) {
	XConv_via_tiling_Config *ConfigPtr = NULL;

	int Index;

	for (Index = (u32)0x0; XConv_via_tiling_ConfigTable[Index].Name != NULL; Index++) {
		if (!BaseAddress || XConv_via_tiling_ConfigTable[Index].Control_BaseAddress == BaseAddress) {
			ConfigPtr = &XConv_via_tiling_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XConv_via_tiling_Initialize(XConv_via_tiling *InstancePtr, UINTPTR BaseAddress) {
	XConv_via_tiling_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XConv_via_tiling_LookupConfig(BaseAddress);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XConv_via_tiling_CfgInitialize(InstancePtr, ConfigPtr);
}
#else
XConv_via_tiling_Config *XConv_via_tiling_LookupConfig(u16 DeviceId) {
	XConv_via_tiling_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XCONV_VIA_TILING_NUM_INSTANCES; Index++) {
		if (XConv_via_tiling_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XConv_via_tiling_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XConv_via_tiling_Initialize(XConv_via_tiling *InstancePtr, u16 DeviceId) {
	XConv_via_tiling_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XConv_via_tiling_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XConv_via_tiling_CfgInitialize(InstancePtr, ConfigPtr);
}
#endif

#endif

