// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.1.2 (64-bit)
// Tool Version Limit: 2024.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
/***************************** Include Files *********************************/
#include "xconv_via_tiling.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XConv_via_tiling_CfgInitialize(XConv_via_tiling *InstancePtr, XConv_via_tiling_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XConv_via_tiling_Set_in_ch(XConv_via_tiling *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv_via_tiling_WriteReg(InstancePtr->Control_BaseAddress, XCONV_VIA_TILING_CONTROL_ADDR_IN_CH_DATA, Data);
}

u32 XConv_via_tiling_Get_in_ch(XConv_via_tiling *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv_via_tiling_ReadReg(InstancePtr->Control_BaseAddress, XCONV_VIA_TILING_CONTROL_ADDR_IN_CH_DATA);
    return Data;
}

void XConv_via_tiling_Set_out_ch(XConv_via_tiling *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv_via_tiling_WriteReg(InstancePtr->Control_BaseAddress, XCONV_VIA_TILING_CONTROL_ADDR_OUT_CH_DATA, Data);
}

u32 XConv_via_tiling_Get_out_ch(XConv_via_tiling *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv_via_tiling_ReadReg(InstancePtr->Control_BaseAddress, XCONV_VIA_TILING_CONTROL_ADDR_OUT_CH_DATA);
    return Data;
}

void XConv_via_tiling_Set_H(XConv_via_tiling *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv_via_tiling_WriteReg(InstancePtr->Control_BaseAddress, XCONV_VIA_TILING_CONTROL_ADDR_H_DATA, Data);
}

u32 XConv_via_tiling_Get_H(XConv_via_tiling *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv_via_tiling_ReadReg(InstancePtr->Control_BaseAddress, XCONV_VIA_TILING_CONTROL_ADDR_H_DATA);
    return Data;
}

void XConv_via_tiling_Set_W(XConv_via_tiling *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv_via_tiling_WriteReg(InstancePtr->Control_BaseAddress, XCONV_VIA_TILING_CONTROL_ADDR_W_DATA, Data);
}

u32 XConv_via_tiling_Get_W(XConv_via_tiling *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv_via_tiling_ReadReg(InstancePtr->Control_BaseAddress, XCONV_VIA_TILING_CONTROL_ADDR_W_DATA);
    return Data;
}

void XConv_via_tiling_Set_input_r(XConv_via_tiling *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv_via_tiling_WriteReg(InstancePtr->Control_BaseAddress, XCONV_VIA_TILING_CONTROL_ADDR_INPUT_R_DATA, (u32)(Data));
    XConv_via_tiling_WriteReg(InstancePtr->Control_BaseAddress, XCONV_VIA_TILING_CONTROL_ADDR_INPUT_R_DATA + 4, (u32)(Data >> 32));
}

u64 XConv_via_tiling_Get_input_r(XConv_via_tiling *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv_via_tiling_ReadReg(InstancePtr->Control_BaseAddress, XCONV_VIA_TILING_CONTROL_ADDR_INPUT_R_DATA);
    Data += (u64)XConv_via_tiling_ReadReg(InstancePtr->Control_BaseAddress, XCONV_VIA_TILING_CONTROL_ADDR_INPUT_R_DATA + 4) << 32;
    return Data;
}

void XConv_via_tiling_Set_weight(XConv_via_tiling *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv_via_tiling_WriteReg(InstancePtr->Control_BaseAddress, XCONV_VIA_TILING_CONTROL_ADDR_WEIGHT_DATA, (u32)(Data));
    XConv_via_tiling_WriteReg(InstancePtr->Control_BaseAddress, XCONV_VIA_TILING_CONTROL_ADDR_WEIGHT_DATA + 4, (u32)(Data >> 32));
}

u64 XConv_via_tiling_Get_weight(XConv_via_tiling *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv_via_tiling_ReadReg(InstancePtr->Control_BaseAddress, XCONV_VIA_TILING_CONTROL_ADDR_WEIGHT_DATA);
    Data += (u64)XConv_via_tiling_ReadReg(InstancePtr->Control_BaseAddress, XCONV_VIA_TILING_CONTROL_ADDR_WEIGHT_DATA + 4) << 32;
    return Data;
}

void XConv_via_tiling_Set_bias(XConv_via_tiling *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv_via_tiling_WriteReg(InstancePtr->Control_BaseAddress, XCONV_VIA_TILING_CONTROL_ADDR_BIAS_DATA, (u32)(Data));
    XConv_via_tiling_WriteReg(InstancePtr->Control_BaseAddress, XCONV_VIA_TILING_CONTROL_ADDR_BIAS_DATA + 4, (u32)(Data >> 32));
}

u64 XConv_via_tiling_Get_bias(XConv_via_tiling *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv_via_tiling_ReadReg(InstancePtr->Control_BaseAddress, XCONV_VIA_TILING_CONTROL_ADDR_BIAS_DATA);
    Data += (u64)XConv_via_tiling_ReadReg(InstancePtr->Control_BaseAddress, XCONV_VIA_TILING_CONTROL_ADDR_BIAS_DATA + 4) << 32;
    return Data;
}

void XConv_via_tiling_Set_output_r(XConv_via_tiling *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv_via_tiling_WriteReg(InstancePtr->Control_BaseAddress, XCONV_VIA_TILING_CONTROL_ADDR_OUTPUT_R_DATA, (u32)(Data));
    XConv_via_tiling_WriteReg(InstancePtr->Control_BaseAddress, XCONV_VIA_TILING_CONTROL_ADDR_OUTPUT_R_DATA + 4, (u32)(Data >> 32));
}

u64 XConv_via_tiling_Get_output_r(XConv_via_tiling *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv_via_tiling_ReadReg(InstancePtr->Control_BaseAddress, XCONV_VIA_TILING_CONTROL_ADDR_OUTPUT_R_DATA);
    Data += (u64)XConv_via_tiling_ReadReg(InstancePtr->Control_BaseAddress, XCONV_VIA_TILING_CONTROL_ADDR_OUTPUT_R_DATA + 4) << 32;
    return Data;
}

void XConv_via_tiling_Set_stride(XConv_via_tiling *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv_via_tiling_WriteReg(InstancePtr->Control_BaseAddress, XCONV_VIA_TILING_CONTROL_ADDR_STRIDE_DATA, Data);
}

u32 XConv_via_tiling_Get_stride(XConv_via_tiling *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv_via_tiling_ReadReg(InstancePtr->Control_BaseAddress, XCONV_VIA_TILING_CONTROL_ADDR_STRIDE_DATA);
    return Data;
}

void XConv_via_tiling_Set_pad(XConv_via_tiling *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv_via_tiling_WriteReg(InstancePtr->Control_BaseAddress, XCONV_VIA_TILING_CONTROL_ADDR_PAD_DATA, Data);
}

u32 XConv_via_tiling_Get_pad(XConv_via_tiling *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv_via_tiling_ReadReg(InstancePtr->Control_BaseAddress, XCONV_VIA_TILING_CONTROL_ADDR_PAD_DATA);
    return Data;
}

