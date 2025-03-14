# ==============================================================
# Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.1.2 (64-bit)
# Tool Version Limit: 2024.05
# Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
# Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
# 
# ==============================================================
cmake_minimum_required(VERSION 3.15)
project(conv_via_tiling)
find_package(common)
collector_create (PROJECT_LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}")
collect (PROJECT_LIB_SOURCES xconv_via_tiling.c)
collect (PROJECT_LIB_SOURCES xconv_via_tiling_sinit.c)
collect (PROJECT_LIB_SOURCES xconv_via_tiling_linux.c)
collect (PROJECT_LIB_SOURCES xconv_via_tiling_g.c)
collector_list (_sources PROJECT_LIB_SOURCES)
collector_create (PROJECT_LIB_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}")
collect (PROJECT_LIB_HEADERS xconv_via_tiling.h)
collect (PROJECT_LIB_HEADERS xconv_via_tiling_hw.h)
collector_list (_headers PROJECT_LIB_HEADERS)
file(COPY ${_headers} DESTINATION ${CMAKE_BINARY_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)
add_library(conv_via_tiling STATIC ${_sources})
set_target_properties(conv_via_tiling PROPERTIES LINKER_LANGUAGE C)

