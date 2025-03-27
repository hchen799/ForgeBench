# ForgeBench

## Auto Generation Framework

In the conv/gemm/llm folders, [generate_code.py] generates HLS designs from JSON config files - building the top function design, the testbench, the .tcl script and data files. 

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

## Ready-To-Go BenchMark Suites

### ML BenchMarks
We provide an ML oriented benchmark suite with 6000+ testcases - 1920 GEMM, 2304 DNN, 1944 LLM. The file is compressed and split for uploading. 
First merge the split files:

```sh
cat ML_testsuite_part_* > ML_testsuite.tar.gz
```

Then extract:

```sh
tar -xvzf ML_testsuite.tar.gz
```

### Modular HLS BenchMarks

The Modular HLS benchmark suite includes 13 testcases across GEMM, DNN and LLM domains. Each testcase has 2--3 input programs generated with ForgeBench with an ideal modularized implementation

Extract with:

```sh
tar -xvzf ModularHLS_testsuite.tar.gz
```

