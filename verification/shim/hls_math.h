// Local shim mapping hls::<fn> math calls to <cmath> so ForgeBench float
// designs compile and run with a stock g++ (no Vitis). Used only for the
// local CSIM-free self-check; the real server run uses Vitis' hls_math.h.
#ifndef FORGEBENCH_SHIM_HLS_MATH_H
#define FORGEBENCH_SHIM_HLS_MATH_H

#include <cmath>
#include <math.h>

namespace hls {
// Base names have proper float/double overloads in <cmath>.
using std::exp;
using std::tanh;
using std::sqrt;
using std::log;
using std::fabs;
using std::pow;
using std::sin;
using std::cos;
using std::erf;
using std::floor;
using std::ceil;
using std::fmax;
using std::fmin;
// The f-suffixed C variants are only guaranteed in the global namespace
// (<math.h>); libstdc++ (e.g. GCC 11) does not re-export them under std::.
using ::expf;
using ::tanhf;
using ::sqrtf;
using ::logf;
using ::powf;
using ::sinf;
using ::cosf;
using ::erff;
}  // namespace hls

#endif  // FORGEBENCH_SHIM_HLS_MATH_H
