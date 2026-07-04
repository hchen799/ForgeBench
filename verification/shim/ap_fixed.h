// Local shim so ForgeBench float designs compile with a stock g++ (no Vitis).
// In float verification mode data_type is "float" and the ap_fixed<> template
// is never instantiated in generated code; this alias just satisfies the include.
#ifndef FORGEBENCH_SHIM_AP_FIXED_H
#define FORGEBENCH_SHIM_AP_FIXED_H

template <int W = 32, int I = 16> using ap_fixed = float;
template <int W = 32, int I = 16> using ap_ufixed = float;
template <int W = 32> using ap_int = int;
template <int W = 32> using ap_uint = unsigned int;

#endif  // FORGEBENCH_SHIM_AP_FIXED_H
