"""Explicit ResNet-18 graph matching the accelerator's documented variant."""
import torch
import torch.nn.functional as F


class FloatOps:
    name = ""

    def conv(self, x, weights, stride=1, padding=0, output_shift=0):
        return F.conv2d(x[None], weights, stride=stride, padding=padding)[0] * (2.0 ** -output_shift)

    def bn(self, x, params):
        gamma, beta, mean, variance = (p[:, None, None] for p in params)
        return gamma * (x - mean) / torch.sqrt(variance + 1e-5) + beta

    def relu(self, x):
        return x.clamp_min(0)

    def pool(self, x):
        return F.max_pool2d(F.pad(x[None], (1, 1, 1, 1)), 3, 2)[0]

    def add(self, x, skip):
        return x + skip

    def gap(self, x):
        return x.mean((1, 2))

    def fc(self, x, weights, output_shift=0):
        return F.linear(x, weights) * (2.0 ** -output_shift)


def forward(tensors, shifts, ops, checkpoint):
    """Graph is explicit, not inferred from the generated C++ call sequence.

    checkpoint(name, value) consumes each tensor immediately, before scratch
    reuse could matter. No in-place Python operations alias the skip branch.
    """
    def run(name, method, *args, **kwargs):
        ops.name = name
        result = getattr(ops, method)(*args, **kwargs)
        if not torch.isfinite(result).all():
            raise ValueError(f"non-finite result at {name}")
        checkpoint(name, result)
        return result

    x = tensors["DRAM_input"]
    if tuple(x.shape) != (3, 224, 224):
        raise ValueError("expected CHW input (3, 224, 224)")
    x = run("stem.conv", "conv", x, tensors["DRAM_w_stem"], stride=2, padding=3, output_shift=shifts["SHIFT_STEM"])
    x = run("stem.bn", "bn", x, tensors["DRAM_bn_stem"])
    x = run("stem.relu", "relu", x)
    x = run("stem.pool", "pool", x)
    for stage, channels, size in [(1, 64, 56), (2, 128, 28), (3, 256, 14), (4, 512, 7)]:
        for block in range(2):
            prefix = f"s{stage}_b{block}"
            down = stage > 1 and block == 0
            shift = f"SHIFT_{prefix.upper()}"
            skip = x
            y = run(prefix + ".conv1", "conv", x, tensors[f"DRAM_w_{prefix}_1"], stride=2 if down else 1,
                    padding=1, output_shift=shifts[shift + "_1"])
            y = run(prefix + ".bn1", "bn", y, tensors[f"DRAM_bn_{prefix}_1"])
            y = run(prefix + ".relu1", "relu", y)
            y = run(prefix + ".conv2", "conv", y, tensors[f"DRAM_w_{prefix}_2"], padding=1,
                    output_shift=shifts[shift + "_2"])
            y = run(prefix + ".bn2", "bn", y, tensors[f"DRAM_bn_{prefix}_2"])
            if down:
                skip = run(prefix + ".down", "conv", x, tensors[f"DRAM_w_{prefix}_down"], stride=2,
                           output_shift=shifts[shift + "_DOWN"])
            x = run(prefix + ".add", "add", y, skip)
            x = run(prefix + ".out", "relu", x)
            if tuple(x.shape) != (channels, size, size):
                raise ValueError(f"unexpected shape at {prefix}: {tuple(x.shape)}")
    x = run("head.gap", "gap", x)
    return run("head.logits", "fc", x, tensors["DRAM_fc"], output_shift=shifts["SHIFT_FC"])
