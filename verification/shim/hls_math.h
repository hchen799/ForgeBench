// Local shim mapping hls::<fn> math calls to <cmath> so ForgeBench float
// designs compile and run with a stock g++ (no Vitis). Used only for the
// local CSIM-free self-check; the real server run uses Vitis' hls_math.h.
#ifndef FORGEBENCH_SHIM_HLS_MATH_H
#define FORGEBENCH_SHIM_HLS_MATH_H

#include <cmath>

namespace hls {
using std::exp;
using std::expf;
using std::tanh;
using std::tanhf;
using std::sqrt;
using std::sqrtf;
using std::log;
using std::logf;
using std::fabs;
using std::pow;
using std::powf;
using std::sin;
using std::sinf;
using std::cos;
using std::cosf;
using std::erf;
using std::erff;
using std::floor;
using std::ceil;
using std::fmax;
using std::fmin;
}  // namespace hls

#endif  // FORGEBENCH_SHIM_HLS_MATH_H
