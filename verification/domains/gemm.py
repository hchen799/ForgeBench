"""Golden reference for the gemm domain.

Textbook oracle for the gemm op vocabulary: load, store, gemm, vmm, mmv,
dot_product, activation. The *math* is the canonical definition; we additionally
model two C memory semantics faithfully (these are not math changes, they are how
the generated C actually behaves):

  - BRAM arrays are C globals, so they start zero-initialized. Arrays that are
    read before being written therefore contribute zeros.
  - load/store copy exactly prod(op_dims) elements row-major (the copy function is
    sized by the op's own `dims`, which can be smaller than the declared array).

Verified against the emitted C in gemm/generate_code.py.

Op / arg conventions:
  load/store  args=[src, dst]              copy prod(dims) elems src->dst
  gemm        args=[A,B,bias,out] dims=[M,N,K]  out=A@B (+bias if with_bias)
  vmm         args=[A,B,bias,out] dims=[M,N]     out[N]=sum_i A[i][j]*B[i]  (B is [M])
  mmv         args=[A,B,bias,out] dims=[M,N]     out[M]=sum_j A[i][j]*B[j]  (B is [N])
  dot_product args=[A,B,bias,out] dims=[M]       out[0]=sum(A*B)
  activation  args=[in,out]       dims=[H,W]     out=act(in)
"""
import numpy as np

from verification.activations import apply_activation

GEMM_OPS = {"gemm", "vmm", "mmv", "dot_product"}


def _prod(dims):
    out = 1
    for d in dims:
        out *= d
    return out


def _read(arrays, name, shape):
    """Read the first prod(shape) elements of `name`, row-major, reshaped."""
    return arrays[name].reshape(-1)[: _prod(shape)].reshape(shape).astype(np.float32)


def _write(arrays, name, value):
    """Write `value` into the first value.size elements of `name` in place."""
    flat = arrays[name].reshape(-1)
    v = np.asarray(value, dtype=np.float32).reshape(-1)
    flat[: v.size] = v


def _op_copy(op, arrays):  # load and store
    src, dst = op["args"][0], op["args"][1]
    n = _prod(op["dims"])
    _write(arrays, dst, arrays[src].reshape(-1)[:n])


def _op_gemm(op, arrays):
    M, N, K = op["dims"]
    _order, _unroll, with_bias, _inline = op["func_info"]
    a, b, bias, out = op["args"]
    res = (_read(arrays, a, (M, N)) @ _read(arrays, b, (N, K))).astype(np.float32)
    if with_bias:
        res = (res + _read(arrays, bias, (M, K))).astype(np.float32)
    _write(arrays, out, res)


def _op_mmv(op, arrays):
    M, N = op["dims"]
    _order, _unroll, with_bias, _inline = op["func_info"]
    a, b, bias, out = op["args"]
    res = (_read(arrays, a, (M, N)) @ _read(arrays, b, (N,))).astype(np.float32)
    if with_bias:
        res = (res + _read(arrays, bias, (M,))).astype(np.float32)
    _write(arrays, out, res)


def _op_vmm(op, arrays):
    M, N = op["dims"]
    _order, _unroll, with_bias, _inline = op["func_info"]
    a, b, bias, out = op["args"]
    res = (_read(arrays, a, (M, N)).T @ _read(arrays, b, (M,))).astype(np.float32)
    if with_bias:
        res = (res + _read(arrays, bias, (N,))).astype(np.float32)
    _write(arrays, out, res)


def _op_dot(op, arrays):
    M = op["dims"][0]
    _unroll, with_bias, _inline = op["func_info"]
    a, b, bias, out = op["args"]
    res = np.array([np.dot(_read(arrays, a, (M,)), _read(arrays, b, (M,)))], dtype=np.float32)
    if with_bias:
        res = (res + _read(arrays, bias, (1,))).astype(np.float32)
    _write(arrays, out, res)


def _op_activation(op, arrays):
    H, W = op["dims"]
    act_name = op["func_info"][1]
    params = op["func_info"][2:]  # extra activation parameters, e.g. elu alpha
    in_name, out_name = op["args"]
    _write(arrays, out_name, apply_activation(act_name, _read(arrays, in_name, (H, W)), params))


_DISPATCH = {
    "load": _op_copy,
    "store": _op_copy,
    "gemm": _op_gemm,
    "mmv": _op_mmv,
    "vmm": _op_vmm,
    "dot_product": _op_dot,
    "activation": _op_activation,
}


def run(ops, arrays):
    """Walk ops in insertion order, mutating `arrays` (name -> np.ndarray) in place."""
    for op_name, op in ops.items():
        fn = op["func_name"]
        if fn not in _DISPATCH:
            raise NotImplementedError(f"gemm golden: unsupported op '{fn}' ({op_name})")
        _DISPATCH[fn](op, arrays)
