"""Configured ResNet operators tested against real vendor ap_fixed arithmetic."""
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from verification.models.resnet18.codegen import emit_project
from verification.models.resnet18.config import Config
from verification.models.resnet18.fixed import FixedOps, sqrt_codes


@pytest.mark.parametrize("w,i", [(32, 10), (24, 8), (12, 6)])
def test_wide_bn_sqrt_shift_and_saturating_mac(tmp_path, w, i):
    include = Path(os.environ.get("VITIS_HLS_INCLUDE", "/tools/software/xilinx/ARCHIVE/Vitis_HLS/2024.1/include"))
    if not (include / "ap_fixed.h").is_file():
        pytest.skip("real vendor headers unavailable")
    c = Config(word_bits=w, integer_bits=i, tile_c=7, tile_h=3, tile_w=5).validate()
    p = c.precision
    emit_project(tmp_path, c)
    source = tmp_path / "probe.cpp"
    source.write_text(f'''#include "top.cpp"
#include <iostream>
data_t from_raw(long long v) {{ data_t x; x.range({w-1},0)=v; return x; }}
long long code(data_t x) {{ return ap_int<{w}>(x.range({w-1},0)).to_int64(); }}
int main() {{
    char op;
    while(std::cin >> op) {{
        long long v;
        if(op=='B') {{
            long long g,b,m,var; std::cin >> v >> g >> b >> m >> var;
            data_t x[1][1][1],y[1][1][1],params[4][1];
            x[0][0][0]=from_raw(v); params[0][0]=from_raw(g); params[1][0]=from_raw(b);
            params[2][0]=from_raw(m); params[3][0]=from_raw(var);
            batchnorm_tiled_runtime<1,1,1,1>(1,1,x,params,y);
            std::cout << code(y[0][0][0]) << '\\n';
        }} else if(op=='R') {{
            std::cin >> v;
            auto root=verification_sqrt((acc_t)from_raw(v));
            std::cout << ap_int<{c.acc_word_bits+1}>(root.range({c.acc_word_bits},0)).to_int64() << '\\n';
        }} else if(op=='S') {{
            int shift; std::cin >> v >> shift;
            acc_t x; x.range({c.acc_word_bits-1},0)=v;
            auto y=apply_power_of_two_shift(x,shift);
            std::cout << ap_int<{c.acc_word_bits}>(y.range({c.acc_word_bits-1},0)).to_int64() << '\\n';
        }} else if(op=='F') {{
            data_t x[17],w[2][17],y[2];
            for(int j=0;j<17;++j) {{ x[j]=from_raw({p.maximum});
                w[0][j]=from_raw(j<9 ? {p.maximum} : {p.minimum});
                w[1][j]=from_raw(j<9 ? {p.minimum} : {p.maximum}); }}
            fc_tiled_runtime<2,17>(x,w,3,y);
            std::cout << code(y[0]) << '\\n' << code(y[1]) << '\\n';
        }} else return 2;
    }}
}}
''')
    subprocess.run(["g++", "-std=c++14", "-O2", "-DNDEBUG", "-Wno-unknown-pragmas", "-I" + str(include),
                    str(source), "-o", str(tmp_path / "probe")], check=True, capture_output=True, text=True)

    def probe(commands):
        result = subprocess.run([str(tmp_path / "probe")], input="\n".join(commands) + "\n", check=True, capture_output=True, text=True)
        return torch.tensor([int(v) for v in result.stdout.split()], dtype=torch.int64)

    torch.set_num_threads(2)
    rng = np.random.default_rng(17)
    rows = rng.integers(p.minimum, p.maximum + 1, (512, 5), dtype=np.int64)
    rows[:, 4] = rng.integers(0, p.maximum + 1, 512)
    rows[:4, 4] = [0, 1, p.maximum - 1, p.maximum]
    x = torch.from_numpy(rows[:, 0].copy()).reshape(-1, 1, 1)
    params = torch.from_numpy(rows[:, 1:].T.copy())
    ops = FixedOps(config=c)
    assert torch.equal(probe(["B " + " ".join(map(str, row)) for row in rows]), ops.bn(x, params).flatten())
    variance = torch.from_numpy(rows[:, 4].copy())
    assert torch.equal(probe([f"R {v}" for v in variance.tolist()]), sqrt_codes(variance, p))
    pairs = [(v, s) for v in [ops.acc_min, -7, -1, 0, 1, 7, ops.acc_max] for s in [-3, -1, 0, 1, 3]]
    expected = torch.tensor([ops.shift(torch.tensor(v), s).item() for v, s in pairs])
    assert torch.equal(probe([f"S {v} {s}" for v, s in pairs]), expected)
    x = torch.full((17,), p.maximum, dtype=torch.int64)
    weights = torch.tensor([[p.maximum] * 9 + [p.minimum] * 8, [p.minimum] * 9 + [p.maximum] * 8])
    assert torch.equal(probe(["F"]), ops.fc(x, weights, 3))
