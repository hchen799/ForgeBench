// 0x00 : reserved
// 0x04 : reserved
// 0x08 : reserved
// 0x0c : reserved
// 0x10 : Data signal of in_ch
//        bit 31~0 - in_ch[31:0] (Read/Write)
// 0x14 : reserved
// 0x18 : Data signal of out_ch
//        bit 31~0 - out_ch[31:0] (Read/Write)
// 0x1c : reserved
// 0x20 : Data signal of H
//        bit 31~0 - H[31:0] (Read/Write)
// 0x24 : reserved
// 0x28 : Data signal of W
//        bit 31~0 - W[31:0] (Read/Write)
// 0x2c : reserved
// 0x30 : Data signal of input_r
//        bit 31~0 - input_r[31:0] (Read/Write)
// 0x34 : Data signal of input_r
//        bit 31~0 - input_r[63:32] (Read/Write)
// 0x38 : reserved
// 0x3c : Data signal of weight
//        bit 31~0 - weight[31:0] (Read/Write)
// 0x40 : Data signal of weight
//        bit 31~0 - weight[63:32] (Read/Write)
// 0x44 : reserved
// 0x48 : Data signal of bias
//        bit 31~0 - bias[31:0] (Read/Write)
// 0x4c : Data signal of bias
//        bit 31~0 - bias[63:32] (Read/Write)
// 0x50 : reserved
// 0x54 : Data signal of output_r
//        bit 31~0 - output_r[31:0] (Read/Write)
// 0x58 : Data signal of output_r
//        bit 31~0 - output_r[63:32] (Read/Write)
// 0x5c : reserved
// 0x60 : Data signal of stride
//        bit 31~0 - stride[31:0] (Read/Write)
// 0x64 : reserved
// 0x68 : Data signal of pad
//        bit 31~0 - pad[31:0] (Read/Write)
// 0x6c : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define CONTROL_ADDR_IN_CH_DATA    0x10
#define CONTROL_BITS_IN_CH_DATA    32
#define CONTROL_ADDR_OUT_CH_DATA   0x18
#define CONTROL_BITS_OUT_CH_DATA   32
#define CONTROL_ADDR_H_DATA        0x20
#define CONTROL_BITS_H_DATA        32
#define CONTROL_ADDR_W_DATA        0x28
#define CONTROL_BITS_W_DATA        32
#define CONTROL_ADDR_INPUT_R_DATA  0x30
#define CONTROL_BITS_INPUT_R_DATA  64
#define CONTROL_ADDR_WEIGHT_DATA   0x3c
#define CONTROL_BITS_WEIGHT_DATA   64
#define CONTROL_ADDR_BIAS_DATA     0x48
#define CONTROL_BITS_BIAS_DATA     64
#define CONTROL_ADDR_OUTPUT_R_DATA 0x54
#define CONTROL_BITS_OUTPUT_R_DATA 64
#define CONTROL_ADDR_STRIDE_DATA   0x60
#define CONTROL_BITS_STRIDE_DATA   32
#define CONTROL_ADDR_PAD_DATA      0x68
#define CONTROL_BITS_PAD_DATA      32
