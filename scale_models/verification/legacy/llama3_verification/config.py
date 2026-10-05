"""Architecture and packed DRAM layout; no numerical framework dependency."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    profile: str = "llama3_8b"
    layers: int = 32
    hidden: int = 4096
    ffn: int = 14336
    q_heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    vocab: int = 128256
    max_ctx: int = 2048
    prefill_tile: int = 16
    tile_in: int = 128
    tile_out: int = 128
    rope_theta: float = 500000.0
    epsilon: float = 1e-5

    def validate(self):
        for field in ("layers", "hidden", "ffn", "q_heads", "kv_heads", "head_dim", "vocab", "max_ctx", "prefill_tile", "tile_in", "tile_out"):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")
        if self.hidden != self.q_heads * self.head_dim or self.q_heads % self.kv_heads:
            raise ValueError("inconsistent grouped-query attention dimensions")
        if self.head_dim % 2 or self.max_ctx > 8192 or self.prefill_tile > self.max_ctx:
            raise ValueError("require even head_dim and prefill_tile <= max_ctx <= 8192")
        if max(self.hidden, self.ffn) > 32768:
            raise ValueError("dimensions exceed the verified accumulator bound")
        if self.rope_theta != 500000.0 or self.epsilon != 1e-5:
            raise ValueError("this implementation targets Llama 3 epsilon and RoPE theta")
        return self

    @property
    def kv_dim(self):
        return self.kv_heads * self.head_dim

    def to_dict(self):
        return asdict(self)


def make_config(profile="llama3_8b", **overrides):
    values = {} if profile == "llama3_8b" else dict(profile="tiny", layers=2, hidden=64, ffn=128,
              q_heads=4, kv_heads=1, head_dim=16, vocab=257, max_ctx=32, prefill_tile=4, tile_in=32, tile_out=32)
    if profile not in ("llama3_8b", "tiny"):
        raise ValueError("unknown profile")
    values.update({k: v for k, v in overrides.items() if v is not None})
    return Config(**values).validate()


def segments(config):
    result, offset = [], 0

    def add(name, shape, kind, fan_in=0):
        nonlocal offset
        import math
        size = math.prod(shape)
        result.append(dict(name=name, shape=list(shape), offset=offset, count=size, kind=kind, fan_in=fan_in))
        offset += size

    c = config
    add("embedding", (c.vocab, c.hidden), "embedding")
    for layer in range(c.layers):
        prefix = f"layers.{layer}."
        add(prefix + "attn_norm", (c.hidden,), "norm")
        add(prefix + "q_proj", (c.hidden, c.hidden), "linear", c.hidden)
        add(prefix + "k_proj", (c.kv_dim, c.hidden), "linear", c.hidden)
        add(prefix + "v_proj", (c.kv_dim, c.hidden), "linear", c.hidden)
        add(prefix + "o_proj", (c.hidden, c.hidden), "residual", c.hidden)
        add(prefix + "ffn_norm", (c.hidden,), "norm")
        add(prefix + "gate_proj", (c.ffn, c.hidden), "linear", c.hidden)
        add(prefix + "up_proj", (c.ffn, c.hidden), "linear", c.hidden)
        add(prefix + "down_proj", (c.hidden, c.ffn), "residual", c.ffn)
    add("final_norm", (c.hidden,), "norm")
    add("lm_head", (c.vocab, c.hidden), "linear", c.hidden)
    return result


def scratch_layout(c):
    dims = [("hidden", c.hidden), ("norm", c.hidden), ("q", c.hidden), ("k", c.kv_dim),
            ("v", c.kv_dim), ("context", c.hidden), ("mid", c.hidden), ("gate", c.ffn),
            ("up", c.ffn), ("ffn_out", c.hidden)]
    offset, layout = 0, {}
    for name, cols in dims:
        layout[name] = offset
        offset += c.prefill_tile * cols
    return layout, offset
