"""FP64 and AP_TRN/AP_WRAP PyTorch models for production ResNet-18.

The intended graph is shared; full-buffer and tiled versions have separately
named reference contracts. Both saved versions accumulate each conv product
into data_t, unlike the dedicated accelerator's wide/shared-exponent scheme.
"""
import torch
import torch.nn.functional as F

from .arithmetic import DATA, product_sum, sqrt_codes, trunc_div, wrap


class FP64Ops:
    def conv(self, x, w, stride=1, padding=0):
        return F.conv2d(x[None], w, stride=stride, padding=padding)[0]

    def bn(self, x, p):
        gamma, beta, mean, variance = [v[:, None, None] for v in p]
        return gamma * (x - mean) / torch.sqrt(variance + 1e-5) + beta

    def add(self, a, b):
        return a + b

    def gap(self, x):
        return x.mean((1, 2))

    def fc(self, x, w):
        return F.linear(x, w)


class FixedOps:
    def conv(self, x, w, stride=1, padding=0):
        kh, kw = w.shape[-2:]
        h = (x.shape[1] + 2 * padding - kh) // stride + 1
        width = (x.shape[2] + 2 * padding - kw) // stride + 1
        patches = F.unfold(x[None].double(), (kh, kw), padding=padding, stride=stride)[
            0
        ].long()
        return product_sum(patches, w.reshape(w.shape[0], -1)).reshape(
            w.shape[0], h, width
        )

    def bn(self, x, p):
        gamma, beta, mean, variance = [v[:, None, None] for v in p]
        # variance + data_t(eps) has W=17,I=6; eps quantizes to zero.
        denom = sqrt_codes(variance)
        norm = wrap(trunc_div((x - mean) * DATA.scale, denom))
        return wrap((gamma * norm >> DATA.frac) + beta)

    def add(self, a, b):
        return wrap(a + b)

    def gap(self, x):
        # The saved designs cast the area (49) to data_t: it becomes -15.
        total = wrap(x.sum((1, 2)))
        area = int(DATA.quantize(x.shape[1] * x.shape[2]))
        return wrap(trunc_div(total * DATA.scale, area))

    def fc(self, x, w):
        return product_sum(x[:, None], w)[:, 0]


@torch.inference_mode()
def forward(tensors, ops, checkpoint):
    """Explicit independent graph: never interpret JSON ops or generated C++."""

    def emit(name, value):
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite reference tensor: " + name)
        checkpoint(name, value)
        return value

    x = ops.conv(tensors["DRAM_input"], tensors["DRAM_w_stem"], 2, 3)
    x = ops.bn(x, tensors["DRAM_bn_stem"]).clamp_min(0)
    emit("stem", x)
    x = F.max_pool2d(F.pad(x[None].double(), (1, 1, 1, 1)), 3, 2)[0].to(x.dtype)
    emit("pool", x)
    for stage in range(1, 5):
        for block in range(2):
            name = f"s{stage}_b{block}"
            stride = 2 if stage > 1 and block == 0 else 1
            skip = x
            y = ops.conv(x, tensors[f"DRAM_w_{name}_1"], stride, 1)
            y = ops.bn(y, tensors[f"DRAM_bn_{name}_1"]).clamp_min(0)
            y = ops.conv(y, tensors[f"DRAM_w_{name}_2"], 1, 1)
            y = ops.bn(y, tensors[f"DRAM_bn_{name}_2"])
            if stride == 2:
                skip = ops.conv(skip, tensors[f"DRAM_w_{name}_down"], 2)
            x = emit(name, ops.add(y, skip).clamp_min(0))
    x = emit("gap", ops.gap(x))
    return emit("logits", ops.fc(x, tensors["DRAM_fc"].reshape(1000, 512)))


class ResNetReference:
    def __init__(self, variant, fixed):
        if variant not in ("resnet18-full", "resnet18-tiled"):
            raise ValueError("unsupported ResNet reference variant")
        self.variant, self.fixed = variant, fixed
        self.ops = FixedOps() if fixed else FP64Ops()

    def run(self, tensors, checkpoint):
        values = {
            name: tensor.long() if self.fixed else tensor.double() / DATA.scale
            for name, tensor in tensors.items()
        }
        return forward(values, self.ops, checkpoint)
