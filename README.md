# ForgeBench

## Auto Generation Framework

In the conv/gemm/llm folders, `generate_code.py` generates HLS designs from JSON config files - building the top function design, the testbench, the .tcl script and data files. 

To build test cases, follow these steps:

Step 1: Write the .json file based on your model structure and dataflow, put them under the folder test_case_configs

Step 2: Run the following command to generate the C code for HLS synthesis
``` sh
python generate_configs.py
```
You will see the code generated under hls_files directory

Step 3: Run the following command to lauch the HLS synthesis on all data in hls_files/
``` sh
python run_hls_configs.py
```

## Generated benchmark suites

ForgeBench is a *generator*. The sweep suites it produces (3,840 GEMM, 5,184 DNN/conv, 3,888 LLM = 12,912
configurations) are regenerated from `{gemm,conv,llm}/auto_generate_json.py`; the JSON configs, lean
csynth/impl report archives and logs are distributed as a release bundle (Zenodo DOI: TODO) rather than
stored in git. See `release/build_release.sh` and `release/MANIFEST_FILES.csv` (sizes + SHA-256).
The earlier `ML_testsuite_part_*` archives (5,400 configs) were removed from the branch tip; they remain in
git history under tag `r1-public-snapshot`.

### Modular HLS BenchMarks

The Modular HLS benchmark suite includes 13 testcases across GEMM, DNN and LLM domains. Each testcase has 2--3 input programs generated with ForgeBench with an ideal modularized implementation

Extract with:

```sh
tar -xvzf ModularHLS_testsuite.tar.gz
```

