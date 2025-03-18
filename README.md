# HLS_ML_benchmark

In the conv folder, run the following command to generate all of the code you need to run the HLS, including the top function design, the testbench, the .tcl script and data file. 
``` sh
python generate_code.py
```

Step 1: Write the .json file based on your model structure and dataflow, put them under the folder test_case_configs

Step 2: Run the following command to generate the C code for HLS synthesis
``` sh
python generate_configs.py
```
You will see the code generated under hls_files folder

Step 3: Run the following command to lauch the HLS synthesis
``` sh
python run_hls_configs.py
```
