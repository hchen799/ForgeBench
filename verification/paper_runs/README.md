# Paper verification: experiments, configs, scripts

Everything the paper reports about functional verification is produced by the scripts in this folder. Each run is one JSON config
(`<column>__<experiment>.json`, resolved through `extends`) given to the engine `verification/functional_verification.py`.
Raw outputs go to `verification/results_paper/<config>/` (untracked, release bundle); `collect_results.sh` copies the compact part to
the tracked `verification/paper_results/` and builds the tables.

## Columns (data formats)

| Column | Data | Accumulator / operator storage | Used for |
|---|---|---|---|
| A `A_16_5` | `<16,5>` | data type | operators |
| B `B_16_5_op24_8_acc32_10` ("mixed") | `<16,5>` I/O | operators: `<24,8>` operator storage + `<32,10>` RND accumulator; Table 7 designs: `<32,10>` RND accumulator (`acc_t`) | operators, Table 7 |
| C `C_32_10` | `<32,10>` | data type | operators, Table 7, full models, CO-SIM |
| D `D_float` | float | float | operators, Table 7, CO-SIM |

`fixed<W,I>` is truncate + wrap unless modes are written (`fixed<W,I,rnd,sat>`); `op_params` (`acc_type`, `acc_rounding`,
`acc_overflow`, `data_type`) are applied to the operator under test (operators) or to every compute operator (whole designs).

## Experiments

| Experiment | Designs | Configs | Script |
|---|---|---|---|
| Operators, largest range + N=100 | 57 Table-2 operator variants (`verification/operators/*/variants/*.json`) | `{A,B,C,D}_*__max_range.json` | `run_all.sh` |
| Operators, +-1, N=100 | same | `{A,B,C,D}_*__pm1.json` | `run_all.sh` |
| Table 7 generated programs | 18 (`manifest/designs/modular.csv`, role=program, construction=generated) | `{B,C,D}_*__table7_{max_range,pm1}.json` | `run_table7.sh` |
| Table 7 hand-written designs | 8 (`verification/manual_designs.py`: activation_op1-3, conv_block_op1-3, attn_breakdown_op1-2) | `{B,C,D}_*__manual_{max_range,pm1}.json` | `run_table7.sh` |
| Full-scale models | ResNet-18/34/50/101/152 (full, tiled), Llama-3-8B ctx2048, `<32,10>` RND/SAT | — (verify.py) | `run_fullmodel.sh` |
| CO-SIM | 57 operator variants, +-1, 10 looped trials | `{C,D}_*__cosim.json` | `run_cosim.sh` |

Tables: `make_tables.py` -> `verif_ops.tex`, `verif_modular.tex`, `verif_fullmodel.tex` + one CSV each.

### Per-experiment design JSON

Every run writes `configs/<design>__<format>.json`: the exact design JSON of the reported trials — `data_type`, patched `op_params`,
every input's range at the reported range (including fixed weight / batchnorm ranges) and `_verification` (`run_range`, `n_trials`,
`seed_base`, `report_seed_offset`, source JSON). Trial *k* of a report uses seed `seed_base + report_seed_offset + k`.

### Input constraints (per design)

- Operators (`fixed_ranges_by_operator`): weights of matmul, mha, swa, conv held at +-0.1; batchnorm parameters in [0.25, 1] (variance floor).
  `range_search.caps` caps the search for tanh (Vitis `hls::tanh` defect at large |x|).
- Table 7 generated programs (`fixed_ranges_by_design`): weights +-sqrt(3/fan_in) (gemm: the second operand), conv bias +-0.1,
  layernorm/rmsnorm parameters [0.25, 1].
- Hand-written designs (`manual_designs.SPECS`): weights +-sqrt(3/fan_in), conv bias +-0.1, batchnorm [0.25, 1].
  `n_trials_by_design`: conv_block_op1 uses N=10 (~35 min per fixed-point trial) and is left out of the range search.
- Full models (`existing_model_verification/references/runner.py`): input +-1, weights +-sqrt(3/fan_in), BN gamma/var [0.9, 1.1],
  beta/mean +-0.05; Llama embedding +-0.5, norm weights [0.9, 1.1]; 4 prefill tokens + 2 decode steps.

## Largest-range search (`range_search.py`, called by the engine)

1. Scan a geometric grid of input magnitudes r (factor sqrt 2; fixed point from 0.001 to 2x the format's integer range, float 0.001-1000),
   20 samples per point; a point passes if every output element of every sample is within the error bound and nothing crashed.
2. The window is the largest contiguous run of passing points; bisect (4 steps, log space) between its top and the first failure.
3. Confirm the top with N=100 samples, stepping down by x1/1.19 until all pass; then back off one more step (safety margin).
4. Report N=100 fresh samples (disjoint seeds) at that range: max abs error, relative L2, RMSE, SQNR; fixed point also a stress
   range (4x the format's range) to record how the design fails.

Error bound (`format_bound`): c * LSB * (L + stages) + peak fraction, with L the longest accumulation and `stages` the dependent
rounding stages (+ #ops - 1 for whole designs). Float: rtol 1e-3, atol 1e-5. Golden: NumPy float64 on the quantized inputs.

## Reproduce

```bash
JOBS=8 bash verification/paper_runs/run_all.sh            # operators, A-D (~1.5 h on 128 cores)
JOBS=6 bash verification/paper_runs/run_table7.sh          # Table 7, B-D (conv_block_op1 dominates: ~6 h)
SOURCES=<full-model projects> OUT=<dir> PY=<python with torch> bash verification/paper_runs/run_fullmodel.sh
JOBS=16 bash verification/paper_runs/run_cosim.sh          # CO-SIM, C and D (idle server; float conv ~8 h)
FULLMODEL=<dir> bash verification/paper_runs/collect_results.sh
```

One design / one format: `python3 -m verification.functional_verification --config verification/paper_runs/C_32_10__pm1.json
--operators ops/gemm/gemm__bias1 --out /tmp/x` (`--n`, `--range`, `--dtypes`, `--sim` override the config).
