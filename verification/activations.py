"""Textbook activation oracles shared across ForgeBench domains.

Golden-reference policy (see plan / feedback-golden-oracle): implement the
canonical/textbook definition. Deviate to match the emitted C **only** where the
C deliberately implements a widely-used *variant*; each such deviation is noted
inline so it can be reported in the verification README.

Deliberate variants matched here:
  - gelu: tanh approximation (matches gemm/2D_activations_template.cpp), NOT the
    exact erf-based gelu.
  - softmax: row-wise over the last axis (matches the C). We subtract the row max
    for numerical stability; softmax is shift-invariant so this is mathematically
    identical to the C's raw-exp form and stays within float tolerance.

Note: several activations in the C template take extra scalar parameters
(leaky_relu alpha, elu alpha, selu alpha/lambda, relu6 cap, prelu, rrelu,
thresholded_relu). The generator's activation call site passes only (input,
output), so those variants are not instantiable in valid configs today; they are
implemented with standard defaults for completeness but are not exercised.
"""
import numpy as np

_SQRT_2_OVER_PI = np.float32(np.sqrt(2.0 / np.pi))


def relu(x):
    return np.maximum(x, 0.0).astype(np.float32)


def relu6(x, cap=6.0):
    return np.clip(x, 0.0, cap).astype(np.float32)


def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x).astype(np.float32)


def prelu(x, alpha=0.25):
    return np.where(x >= 0, x, alpha * x).astype(np.float32)


def rrelu(x, lower=1.0 / 8, upper=1.0 / 3):
    ralpha = (lower + upper) / 2.0
    return np.where(x >= 0, x, ralpha * x).astype(np.float32)


def thresholded_relu(x, theta=1.0):
    return np.where(x > theta, x, 0.0).astype(np.float32)


def sigmoid(x):
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def tanh_act(x):
    return np.tanh(x).astype(np.float32)


def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1.0)).astype(np.float32)


def selu(x, alpha=1.6732632423543772, lam=1.0507009873554805):
    return np.where(x >= 0, lam * x, lam * alpha * (np.exp(x) - 1.0)).astype(np.float32)


def gelu(x):
    # Deliberate variant: tanh approximation, matching the emitted C.
    x = x.astype(np.float32)
    inner = _SQRT_2_OVER_PI * (x + np.float32(0.044715) * x * x * x)
    return (np.float32(0.5) * x * (1.0 + np.tanh(inner))).astype(np.float32)


def swish(x):
    return (x * sigmoid(x)).astype(np.float32)


def softmax(x):
    # Row-wise over the last axis (matches the C), stabilized by max-subtraction.
    x = x.astype(np.float32)
    shifted = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(shifted)
    return (e / np.sum(e, axis=-1, keepdims=True)).astype(np.float32)


# Map config activation names -> implementation. "tanh" and "tanh_act" both accepted.
_DISPATCH = {
    "relu": relu,
    "relu6": relu6,
    "leaky_relu": leaky_relu,
    "prelu": prelu,
    "rrelu": rrelu,
    "thresholded_relu": thresholded_relu,
    "sigmoid": sigmoid,
    "tanh": tanh_act,
    "tanh_act": tanh_act,
    "elu": elu,
    "selu": selu,
    "gelu": gelu,
    "swish": swish,
    "softmax": softmax,
}


def apply_activation(name, x):
    key = name.lower()
    if key not in _DISPATCH:
        raise NotImplementedError(f"activation '{name}' not implemented in golden reference")
    return _DISPATCH[key](x)
