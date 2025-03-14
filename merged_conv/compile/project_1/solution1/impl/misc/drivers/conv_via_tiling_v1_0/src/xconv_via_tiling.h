// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.1.2 (64-bit)
// Tool Version Limit: 2024.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef XCONV_VIA_TILING_H
#define XCONV_VIA_TILING_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xconv_via_tiling_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#else
typedef struct {
#ifdef SDT
    char *Name;
#else
    u16 DeviceId;
#endif
    u64 Control_BaseAddress;
} XConv_via_tiling_Config;
#endif

typedef struct {
    u64 Control_BaseAddress;
    u32 IsReady;
} XConv_via_tiling;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XConv_via_tiling_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XConv_via_tiling_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XConv_via_tiling_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XConv_via_tiling_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
#ifdef SDT
int XConv_via_tiling_Initialize(XConv_via_tiling *InstancePtr, UINTPTR BaseAddress);
XConv_via_tiling_Config* XConv_via_tiling_LookupConfig(UINTPTR BaseAddress);
#else
int XConv_via_tiling_Initialize(XConv_via_tiling *InstancePtr, u16 DeviceId);
XConv_via_tiling_Config* XConv_via_tiling_LookupConfig(u16 DeviceId);
#endif
int XConv_via_tiling_CfgInitialize(XConv_via_tiling *InstancePtr, XConv_via_tiling_Config *ConfigPtr);
#else
int XConv_via_tiling_Initialize(XConv_via_tiling *InstancePtr, const char* InstanceName);
int XConv_via_tiling_Release(XConv_via_tiling *InstancePtr);
#endif


void XConv_via_tiling_Set_in_ch(XConv_via_tiling *InstancePtr, u32 Data);
u32 XConv_via_tiling_Get_in_ch(XConv_via_tiling *InstancePtr);
void XConv_via_tiling_Set_out_ch(XConv_via_tiling *InstancePtr, u32 Data);
u32 XConv_via_tiling_Get_out_ch(XConv_via_tiling *InstancePtr);
void XConv_via_tiling_Set_H(XConv_via_tiling *InstancePtr, u32 Data);
u32 XConv_via_tiling_Get_H(XConv_via_tiling *InstancePtr);
void XConv_via_tiling_Set_W(XConv_via_tiling *InstancePtr, u32 Data);
u32 XConv_via_tiling_Get_W(XConv_via_tiling *InstancePtr);
void XConv_via_tiling_Set_input_r(XConv_via_tiling *InstancePtr, u64 Data);
u64 XConv_via_tiling_Get_input_r(XConv_via_tiling *InstancePtr);
void XConv_via_tiling_Set_weight(XConv_via_tiling *InstancePtr, u64 Data);
u64 XConv_via_tiling_Get_weight(XConv_via_tiling *InstancePtr);
void XConv_via_tiling_Set_bias(XConv_via_tiling *InstancePtr, u64 Data);
u64 XConv_via_tiling_Get_bias(XConv_via_tiling *InstancePtr);
void XConv_via_tiling_Set_output_r(XConv_via_tiling *InstancePtr, u64 Data);
u64 XConv_via_tiling_Get_output_r(XConv_via_tiling *InstancePtr);
void XConv_via_tiling_Set_stride(XConv_via_tiling *InstancePtr, u32 Data);
u32 XConv_via_tiling_Get_stride(XConv_via_tiling *InstancePtr);
void XConv_via_tiling_Set_pad(XConv_via_tiling *InstancePtr, u32 Data);
u32 XConv_via_tiling_Get_pad(XConv_via_tiling *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
