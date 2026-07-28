"""Golden reference for the conv domain.

Independent textbook oracle for the conv op vocabulary:
    load, store, conv, batchnorm, activation, maxpool, adaptive_avgpool,
    matrix_add.

The *math* is the canonical/textbook definition, parameterized by each op's
genuine knobs (conv stride/padding/kernel, batchnorm epsilon, pooling
kernel/stride, ...). In addition we model two C memory semantics faithfully
(these are how the generated C actually behaves, not math changes):

  - BRAM arrays are C file-scope globals, so they start zero-initialized. Arrays
    read before being written contribute zeros.
  - load/store copy exactly prod(op_dims) elements row-major; compute ops read
    the first prod(shape) flattened elements reshaped and write results into the
    first N flattened elements of the output array.

Where an op's declared logical shape is LARGER than the underlying array's flat
storage (this happens, e.g., in vgg19 maxpool where C is declared 256 but the
source buffer only holds 128 channels), we read with the op's declared strides
and treat any read that runs past the source array's real storage as 0. The
generated C reads out of bounds there too; only the in-bounds region is ever
stored to the output DRAM and compared, so this faithfully models the compared
result.

Verified against the emitted C templates (conv_template.cpp,
batch_norm_template.cpp, maxpool_template.cpp, adaptive_avgpool_template.cpp,
matrix_add_template.cpp, activations_template.cpp) and the dispatch in
conv/generate_code.py generate_conv_function / generate_*_code.

Op / arg conventions (see conv/generate_code.py generate_func_def):
  load/store   args=[src, dst]                     copy prod(dims) elems src->dst
  conv         args=[in, kernel, (bias?), out]
               dims=[C_IN,C_OUT,H_IN,W_IN,H_OUT,W_OUT,K,PAD,STRIDE, <pragma...>]
               func_info=[template, "conv2d"|"group_conv2d", with_bias]
               in[C_IN][H_IN][W_IN], kernel[C_OUT][C_IN][K][K],
               bias[C_OUT], out[C_OUT][H_OUT][W_OUT]
  batchnorm    args=[in, weights, out]  dims=[C,H,W]  func_info=[template, eps]
               weights[4][C]: 0=gamma 1=beta 2=mean 3=variance
  activation   args=[in, out]           dims=[C,H,W]  func_info=[template, name]
  maxpool      args=[in, out]
               dims=[C,H_IN,W_IN,H_OUT,W_OUT,K_H,K_W,STRIDE_H,STRIDE_W]
  adaptive_avgpool args=[in, out]  dims=[C,H_IN,W_IN,H_OUT,W_OUT]
  matrix_add   args=[in1, in2, out]     dims=[C,H,W]
"""
import numpy as np

from verification.activations import apply_activation

CONV_OPS = {"conv", "batchnorm", "maxpool", "adaptive_avgpool"}


def _prod(dims):
    out = 1
    for d in dims:
        out *= d
    return out


def _read(arrays, name, shape):
    """Read the first prod(shape) elements of `name`, row-major, reshaped.

    If the array's real storage is smaller than prod(shape) (the op declares a
    logically larger tensor than the buffer holds), zero-pad the tail; the C
    reads OOB there but only the in-bounds region is ever stored/compared.
    """
    flat = arrays[name].reshape(-1)
    n = _prod(shape)
    if flat.size < n:
        buf = np.zeros(n, dtype=np.float32)
        buf[: flat.size] = flat.astype(np.float32)
    else:
        buf = flat[:n].astype(np.float32)
    return buf.reshape(shape)


def _write(arrays, name, value):
    """Write `value` into the first value.size elements of `name` in place.

    If value is larger than the destination storage, only the leading portion
    that fits is written (mirrors the C writing OOB past a too-small buffer:
    the excess never reaches the compared output DRAM)."""
    flat = arrays[name].reshape(-1)
    v = np.asarray(value, dtype=np.float32).reshape(-1)
    m = min(flat.size, v.size)
    flat[:m] = v[:m]


def _op_copy(op, arrays):  # load and store
    src, dst = op["args"][0], op["args"][1]
    n = _prod(op["dims"])
    _write(arrays, dst, arrays[src].reshape(-1)[:n])


def _op_conv(op, arrays):
    d = op["dims"]
    C_IN, C_OUT, H_IN, W_IN, H_OUT, W_OUT, K, PAD, STRIDE = d[:9]
    _template, func_type, with_bias = op["func_info"]

    # For group_conv2d the generator appends the groups count as the final arg
    # (matching conv/auto_generate_json.py: conv_args.append(groups)). The emitted
    # C still declares the kernel as [C_OUT][C_IN][K][K] (full C_IN) and only reads
    # the block-diagonal slice per group, so the weight shape is unchanged.
    args = list(op["args"])
    groups = 1
    if func_type == "group_conv2d":
        groups = int(args.pop())  # trailing groups count

    if with_bias:
        in_name, w_name, b_name, out_name = args
    else:
        in_name, w_name, out_name = args
        b_name = None

    x = _read(arrays, in_name, (C_IN, H_IN, W_IN))
    w = _read(arrays, w_name, (C_OUT, C_IN, K, K))

    # Initialize output to bias (per out-channel) or zero, matching the C.
    if with_bias:
        bias = _read(arrays, b_name, (C_OUT,))
        out = np.repeat(bias, H_OUT * W_OUT).reshape(C_OUT, H_OUT, W_OUT).astype(np.float32)
    else:
        out = np.zeros((C_OUT, H_OUT, W_OUT), dtype=np.float32)

    ci_per_g = C_IN // groups
    co_per_g = C_OUT // groups

    # Standard cross-correlation with padding/stride, dilation=1, square kernel.
    # For grouped conv, output channels in group g only see input channels in the
    # same group; the kernel is indexed by absolute (co, ci) exactly as the C.
    for i in range(H_OUT):
        for j in range(W_OUT):
            for kh in range(K):
                in_row = i * STRIDE - PAD + kh
                if in_row < 0 or in_row >= H_IN:
                    continue
                for kw in range(K):
                    in_col = j * STRIDE - PAD + kw
                    if in_col < 0 or in_col >= W_IN:
                        continue
                    if groups == 1:
                        out[:, i, j] += w[:, :, kh, kw] @ x[:, in_row, in_col]
                    else:
                        for g in range(groups):
                            ci0, ci1 = g * ci_per_g, (g + 1) * ci_per_g
                            co0, co1 = g * co_per_g, (g + 1) * co_per_g
                            out[co0:co1, i, j] += (
                                w[co0:co1, ci0:ci1, kh, kw]
                                @ x[ci0:ci1, in_row, in_col]
                            )
    _write(arrays, out_name, out)


def _op_batchnorm(op, arrays):
    C, H, W = op["dims"]
    _template, eps = op["func_info"]
    in_name, w_name, out_name = op["args"]

    x = _read(arrays, in_name, (C, H, W))
    weights = _read(arrays, w_name, (4, C))  # gamma, beta, mean, variance
    gamma = weights[0].reshape(C, 1, 1)
    beta = weights[1].reshape(C, 1, 1)
    mean = weights[2].reshape(C, 1, 1)
    var = weights[3].reshape(C, 1, 1)

    norm = (x - mean) / np.sqrt(var + np.float32(eps))
    out = (gamma * norm + beta).astype(np.float32)
    _write(arrays, out_name, out)


def _op_activation(op, arrays):
    C, H, W = op["dims"]
    act_name = op["func_info"][1]
    params = op["func_info"][2:]  # extra activation parameters, e.g. elu alpha
    in_name, out_name = op["args"]
    x = _read(arrays, in_name, (C, H, W))
    _write(arrays, out_name, _apply_conv_activation(act_name, x, params))


def _apply_conv_activation(name, x, params=()):
    """Conv-domain activations. Softmax here is over the CHANNEL axis (axis 0),
    which differs from the shared row-wise softmax; the rest reuse the shared
    textbook oracles."""
    key = name.lower()
    if key == "softmax":
        x = x.astype(np.float32)
        shifted = x - np.max(x, axis=0, keepdims=True)
        e = np.exp(shifted)
        return (e / np.sum(e, axis=0, keepdims=True)).astype(np.float32)
    if key == "hardsigmoid":
        return np.clip((x + 3.0) / 6.0, 0.0, 1.0).astype(np.float32)
    if key == "hardswish":
        return (x * np.clip((x + 3.0) / 6.0, 0.0, 1.0)).astype(np.float32)
    return apply_activation(name, x, params)


def _op_maxpool(op, arrays):
    C, H_IN, W_IN, H_OUT, W_OUT, K_H, K_W, S_H, S_W = op["dims"]
    in_name, out_name = op["args"]
    x = _read(arrays, in_name, (C, H_IN, W_IN))
    out = np.empty((C, H_OUT, W_OUT), dtype=np.float32)
    for c in range(C):
        for i in range(H_OUT):
            for j in range(W_OUT):
                r0, c0 = i * S_H, j * S_W
                window = x[c, r0:r0 + K_H, c0:c0 + K_W]
                out[c, i, j] = window.max()
    _write(arrays, out_name, out)


def _op_adaptive_avgpool(op, arrays):
    C, H_IN, W_IN, H_OUT, W_OUT = op["dims"]
    in_name, out_name = op["args"]
    x = _read(arrays, in_name, (C, H_IN, W_IN))
    out = np.zeros((C, H_OUT, W_OUT), dtype=np.float32)
    for oh in range(H_OUT):
        h_start = int(np.floor(oh * H_IN / H_OUT))
        h_end = min(int(np.ceil((oh + 1) * H_IN / H_OUT)), H_IN)
        for ow in range(W_OUT):
            w_start = int(np.floor(ow * W_IN / W_OUT))
            w_end = min(int(np.ceil((ow + 1) * W_IN / W_OUT)), W_IN)
            region = x[:, h_start:h_end, w_start:w_end]
            count = (h_end - h_start) * (w_end - w_start)
            if count > 0:
                out[:, oh, ow] = region.reshape(C, -1).sum(axis=1) / np.float32(count)
    _write(arrays, out_name, out)


def _op_matrix_add(op, arrays):
    C, H, W = op["dims"]
    in1, in2, out_name = op["args"]
    a = _read(arrays, in1, (C, H, W))
    b = _read(arrays, in2, (C, H, W))
    _write(arrays, out_name, (a + b).astype(np.float32))


_DISPATCH = {
    "load": _op_copy,
    "store": _op_copy,
    "conv": _op_conv,
    "batchnorm": _op_batchnorm,
    "activation": _op_activation,
    "maxpool": _op_maxpool,
    "adaptive_avgpool": _op_adaptive_avgpool,
    "matrix_add": _op_matrix_add,
}


def run(ops, arrays):
    """Walk ops in insertion order, mutating `arrays` (name -> np.ndarray) in place."""
    for op_name, op in ops.items():
        fn = op["func_name"]
        if fn not in _DISPATCH:
            raise NotImplementedError(f"conv golden: unsupported op '{fn}' ({op_name})")
        _DISPATCH[fn](op, arrays)
