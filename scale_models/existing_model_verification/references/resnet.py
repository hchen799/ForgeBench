"""Independent FP64 and fixed PyTorch graphs for production ResNet family.

Source contracts select legacy v1 arithmetic or repaired v2 wide reductions.
The bottleneck graph preserves the production projection without extra BN.
"""
import torch
import torch.nn.functional as F

from .arithmetic import DATA, product_sum, sqrt_codes, trunc_div, wrap


RESNET_SPECS = {
    18: ((2, 2, 2, 2), False),
    34: ((3, 4, 6, 3), False),
    50: ((3, 4, 6, 3), True),
    101: ((3, 4, 23, 3), True),
    152: ((3, 8, 36, 3), True),
}


def depth_from_variant(variant):
    import re

    match = re.fullmatch(r"resnet(18|34|50|101|152)-(?:full|tiled)", variant)
    if match is None:
        raise ValueError("unsupported ResNet reference variant: " + variant)
    return int(match[1])


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
def forward(tensors, ops, checkpoint, depth=18):
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
    blocks, bottleneck = RESNET_SPECS[depth]
    for stage in range(1, 5):
        for block in range(blocks[stage - 1]):
            name = f"s{stage}_b{block}"
            stride = 2 if stage > 1 and block == 0 else 1
            skip = x
            y = ops.conv(
                x,
                tensors[f"DRAM_w_{name}_1"],
                1 if bottleneck else stride,
                0 if bottleneck else 1,
            )
            y = ops.bn(y, tensors[f"DRAM_bn_{name}_1"]).clamp_min(0)
            y = ops.conv(y, tensors[f"DRAM_w_{name}_2"], stride if bottleneck else 1, 1)
            y = ops.bn(y, tensors[f"DRAM_bn_{name}_2"])
            if bottleneck:
                y = ops.conv(y.clamp_min(0), tensors[f"DRAM_w_{name}_3"])
                y = ops.bn(y, tensors[f"DRAM_bn_{name}_3"])
            if f"DRAM_w_{name}_down" in tensors:
                skip = ops.conv(skip, tensors[f"DRAM_w_{name}_down"], stride)
            x = emit(name, ops.add(y, skip).clamp_min(0))
    x = emit("gap", ops.gap(x))
    return emit("logits", ops.fc(x, tensors["DRAM_fc"].reshape(1000, -1)))


class ResNetReference:
    def __init__(self, variant, fixed, version=1):
        self.variant, self.fixed = variant, fixed
        from .repaired import ResNetOps

        self.ops = (ResNetOps() if version == 2 else FixedOps()) if fixed else FP64Ops()
        self.depth = depth_from_variant(variant)

    def run(self, tensors, checkpoint):
        values = {
            name: tensor.long() if self.fixed else tensor.double() / DATA.scale
            for name, tensor in tensors.items()
        }
        return forward(values, self.ops, checkpoint, self.depth)
