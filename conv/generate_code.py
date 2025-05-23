import re
import random
import json
import os

def replace_data_type(data_type: str) -> str:
    # Replace all occurrences of <, >, and , with an underscore
    result = re.sub(r'[<>,]', '_', data_type)
    # Remove all spaces
    result = result.replace(" ", "")
    return result


def generate_load_function(
    dims,              # List of dimensions, e.g., [32, 64, 64]
    data_type="float", # Underlying data type
    func_prefix="load" # Function prefix (will be appended with dims and data_type)
):
    """
    Generates an HLS C function that copies an input array to an output array.
    The function is for arrays of dimensions given in the list 'dims'.
    
    The generated function will have a name of the form:
      <func_prefix>_<dim1>_<dim2>_..._<dimN>_data_t
    
    Parameters:
      dims: list of integers, e.g., [32, 64, 64]
      data_type: string, e.g., "float" (used in typedef and function signature)
      func_prefix: prefix for the function name (default "load")
    
    Returns:
      A string with the generated HLS C code.
    """
    # Construct the typedef and include lines
    code_lines = []
    
    
    #code_lines.append(f"typedef {data_type} data_t;\n")
    
    # Build array dimension string, e.g., "[32][64][64]"
    array_dims = "".join(f"[{d}]" for d in dims)
    
    # Construct the function name. E.g., load_32_64_64_data_t
    dim_suffix = "_".join(str(d) for d in dims)
    
    data_type = replace_data_type(data_type)
    func_name = f"{func_prefix}_{dim_suffix}_{data_type}"
    
    # Build function signature
    code_lines.append(f"void {func_name}(data_t input{array_dims}, data_t output{array_dims})")
    code_lines.append("{")
    
    # Generate nested loops.
    # We'll name the loop indices based on the dimension order: idx0, idx1, ..., idxN.
    indent = "    "
    for idx, d in enumerate(dims):
        code_lines.append(f"{indent * (idx+1)}for (int idx{idx} = 0; idx{idx} < {d}; idx{idx}++) {{")
    
    # Generate the assignment line.
    # Build index strings for input and output, e.g., [idx0][idx1][idx2]
    index_str = "".join(f"[idx{idx}]" for idx in range(len(dims)))
    code_lines.append(f"{indent * (len(dims)+1)}output{index_str} = input{index_str};")
    
    # Close all for loops
    for idx in range(len(dims)-1, -1, -1):
        code_lines.append(f"{indent * (idx+1)}}}")
    
    code_lines.append("}")
    
    return "\n".join(code_lines), func_name


def generate_store_function(
    dims,              # List of dimensions, e.g., [32, 64, 64]
    data_type="float", # Underlying data type
    func_prefix="store" # Function prefix (will be appended with dims and data_type)
):
    """
    Generates an HLS C function that copies an input array to an output array.
    The function is for arrays of dimensions given in the list 'dims'.
    
    The generated function will have a name of the form:
      <func_prefix>_<dim1>_<dim2>_..._<dimN>_data_t
    
    Parameters:
      dims: list of integers, e.g., [32, 64, 64]
      data_type: string, e.g., "float" (used in typedef and function signature)
      func_prefix: prefix for the function name (default "load")
    
    Returns:
      A string with the generated HLS C code.
    """
    # Construct the typedef and include lines
    code_lines = []
    #code_lines.append(f"typedef {data_type} data_t;\n")
    
    # Build array dimension string, e.g., "[32][64][64]"
    array_dims = "".join(f"[{d}]" for d in dims)
    
    # Construct the function name. E.g., load_32_64_64_data_t
    dim_suffix = "_".join(str(d) for d in dims)
    data_type = replace_data_type(data_type)
    func_name = f"{func_prefix}_{dim_suffix}_{data_type}"
    
    # Build function signature
    code_lines.append(f"void {func_name}(data_t input{array_dims}, data_t output{array_dims})")
    code_lines.append("{")
    
    # Generate nested loops.
    # We'll name the loop indices based on the dimension order: idx0, idx1, ..., idxN.
    indent = "    "
    for idx, d in enumerate(dims):
        code_lines.append(f"{indent * (idx+1)}for (int idx{idx} = 0; idx{idx} < {d}; idx{idx}++) {{")
    
    # Generate the assignment line.
    # Build index strings for input and output, e.g., [idx0][idx1][idx2]
    index_str = "".join(f"[idx{idx}]" for idx in range(len(dims)))
    code_lines.append(f"{indent * (len(dims)+1)}output{index_str} = input{index_str};")
    
    # Close all for loops
    for idx in range(len(dims)-1, -1, -1):
        code_lines.append(f"{indent * (idx+1)}}}")
    
    code_lines.append("}")
    
    return "\n".join(code_lines), func_name


# def generate_conv_function(
#     template_path,         # Path to conv_template.cpp
#     func_type="conv2d",    # "conv2d" or "group_conv2d"
#     DATA_TYPE="float",
#     C_IN=16,
#     C_OUT=32,
#     H_IN=64,
#     W_IN=64,
#     H_OUT=64,
#     W_OUT=64,
#     K=3,
#     PAD=1,             # Padding size (assumed same for height and width)
#     STRIDE=2,          # Stride (assumed same for height and width)
#     with_bias=False       # Whether bias is included in the function
# ):
#     """
#     Reads the template file, substitutes placeholders, and extracts only the requested
#     function block (either conv2d or group_conv2d), then writes the output to output_path.
#     """
#     # 1) Read the full template file
#     with open(template_path, "r") as f:
#         template_code = f.read()

#     # 2) Set up bias-related placeholders based on with_bias flag.
#     if with_bias:
#         bias_arg = "    data_t bias[{C_OUT}],\n".format(C_OUT=C_OUT)
#         bias_init_expr = "bias[co]"
#     else:
#         bias_arg = ""
#         bias_init_expr = "((data_t)0)"

#     # For the group function, use similar placeholders.
#     group_bias_arg = bias_arg  # same as above
#     group_bias_init_expr = bias_init_expr

#     # 3) First, perform common placeholder substitution for header values.
#     formatted_code = template_code.format(
#         DATA_TYPE=DATA_TYPE,
#         C_IN=C_IN,
#         C_OUT=C_OUT,
#         H_IN=H_IN,
#         W_IN=W_IN,
#         H_OUT=H_OUT,
#         W_OUT=W_OUT,
#         K=K,
#         PAD = PAD,
#         STRIDE = STRIDE,
#         CONV_BIAS_ARG=bias_arg,
#         BIAS_INIT_EXPR=bias_init_expr,
#         GROUP_BIAS_ARG=group_bias_arg,
#         GROUP_BIAS_INIT_EXPR=group_bias_init_expr
#     )

#     # 4) Now extract the requested function block.
#     if func_type == "conv2d":
#         start_marker = "/*==== CONV2D FUNCTION START ====*/"
#         end_marker = "/*==== CONV2D FUNCTION END ====*/"
#     elif func_type == "group_conv2d":
#         start_marker = "/*==== GROUP_CONV2D FUNCTION START ====*/"
#         end_marker = "/*==== GROUP_CONV2D FUNCTION END ====*/"
#     else:
#         raise ValueError("Invalid function type. Choose 'conv2d' or 'group_conv2d'.")

#     start_index = formatted_code.find(start_marker)
#     end_index = formatted_code.find(end_marker)
#     if start_index == -1 or end_index == -1:
#         raise ValueError("Could not find function markers in the template.")


#     # Include the markers if desired or remove them.
#     function_block = formatted_code[start_index + len(start_marker): end_index].strip()
    
#     DATA_TYPE = replace_data_type(DATA_TYPE)
#     if with_bias == False:
#         dim_suffix = f"_{C_IN}_{C_OUT}_{H_IN}_{W_IN}_{H_OUT}_{W_OUT}_{K}_{PAD}_{STRIDE}_{DATA_TYPE}"
#     else:
#         dim_suffix = f"_{C_IN}_{C_OUT}_{H_IN}_{W_IN}_{H_OUT}_{W_OUT}_{K}_{PAD}_{STRIDE}_{DATA_TYPE}_bias"
#     # Use a regex to capture "void" followed by the function name
#     function_block = re.sub(
#         r"(void\s+(\w+))\s*\(",
#         lambda m: m.group(1) + dim_suffix + "(", 
#         function_block,
#         count=1
#     )
#     # 5) Optionally, include the common header (everything before the first function marker).
#     header_end = formatted_code.find("/*====")
#     if header_end != -1:
#         header = formatted_code[:header_end].strip() + "\n\n"
#     else:
#         header = ""

#     # 6) Combine header and the requested function block.
#     output_code = header + function_block
#     function_name = func_type + dim_suffix
    
#     return output_code, function_name

def generate_conv_function(
    template_path,         # Path to conv_template.cpp
    func_type="conv2d",    # "conv2d" or "group_conv2d"
    DATA_TYPE="float",
    C_IN=16,
    C_OUT=32,
    H_IN=64,
    W_IN=64,
    H_OUT=64,
    W_OUT=64,
    K=3,
    PAD=1,             # Padding size (assumed same for height and width)
    STRIDE=2,          # Stride (assumed same for height and width)
    with_bias=False,   # Whether bias is included in the function
    # New parameters for HLS pragmas:
    input_partition_factor=8,
    kernel_partition_factor1=64,
    kernel_partition_factor2=8,
    bias_partition_factor=64,
    output_partition_factor=64,
    unroll_factor_cin=8,
    unroll_factor_cout=64
):
    """
    Reads the template file, substitutes placeholders, and extracts only the requested
    function block (either conv2d or group_conv2d), then writes the output to output_path.
    """

    # 1) Read the full template file
    with open(template_path, "r") as f:
        template_code = f.read()

    # 2) Set up bias-related placeholders based on with_bias flag.
    if with_bias:
        bias_arg = "    data_t bias[{C_OUT}],\n".format(C_OUT=C_OUT)
        bias_init_expr = "bias[co]"
    else:
        bias_arg = ""
        bias_init_expr = "((data_t)0)"

    # For the group function, use similar placeholders.
    group_bias_arg = bias_arg  # same as above
    group_bias_init_expr = bias_init_expr

    # 3) Perform common placeholder substitution for header values.
    formatted_code = template_code.format(
        DATA_TYPE=DATA_TYPE,
        C_IN=C_IN,
        C_OUT=C_OUT,
        H_IN=H_IN,
        W_IN=W_IN,
        H_OUT=H_OUT,
        W_OUT=W_OUT,
        K=K,
        PAD=PAD,
        STRIDE=STRIDE,
        CONV_BIAS_ARG=bias_arg,
        BIAS_INIT_EXPR=bias_init_expr,
        GROUP_BIAS_ARG=group_bias_arg,
        GROUP_BIAS_INIT_EXPR=group_bias_init_expr,
        ARRAY_FACTOR_INPUT=input_partition_factor,
        ARRAY_FACTOR_KERNEL1=kernel_partition_factor1,
        ARRAY_FACTOR_KERNEL2=kernel_partition_factor2,
        ARRAY_FACTOR_BIAS=bias_partition_factor,
        ARRAY_FACTOR_OUTPUT=output_partition_factor,
        UNROLL_FACTOR_C_IN=unroll_factor_cin,
        UNROLL_FACTOR_C_OUT=unroll_factor_cout,
    )

    # 4) Now extract the requested function block.
    if func_type == "conv2d":
        start_marker = "/*==== CONV2D FUNCTION START ====*/"
        end_marker = "/*==== CONV2D FUNCTION END ====*/"
    elif func_type == "group_conv2d":
        start_marker = "/*==== GROUP_CONV2D FUNCTION START ====*/"
        end_marker = "/*==== GROUP_CONV2D FUNCTION END ====*/"
    else:
        raise ValueError("Invalid function type. Choose 'conv2d' or 'group_conv2d'.")

    start_index = formatted_code.find(start_marker)
    end_index = formatted_code.find(end_marker)
    if start_index == -1 or end_index == -1:
        raise ValueError("Could not find function markers in the template.")

    # Include the markers if desired or remove them.
    function_block = formatted_code[start_index + len(start_marker): end_index].strip()

    # Append dimension suffix to the function name using regex substitution.
    DATA_TYPE_modified = replace_data_type(DATA_TYPE) # (or use a conversion function if needed)
    if not with_bias:
        dim_suffix = f"_{C_IN}_{C_OUT}_{H_IN}_{W_IN}_{H_OUT}_{W_OUT}_{K}_{PAD}_{STRIDE}_{DATA_TYPE_modified}"
    else:
        dim_suffix = f"_{C_IN}_{C_OUT}_{H_IN}_{W_IN}_{H_OUT}_{W_OUT}_{K}_{PAD}_{STRIDE}_{DATA_TYPE_modified}_bias"
    function_block = re.sub(
        r"(void\s+(\w+))\s*\(",
        lambda m: m.group(1) + dim_suffix + "(", 
        function_block,
        count=1
    )

    # 5) Optionally, include the common header (everything before the first function marker).
    header_end = formatted_code.find("/*====")
    if header_end != -1:
        header = formatted_code[:header_end].strip() + "\n\n"
    else:
        header = ""

    # 6) Combine header and the requested function block.
    output_code = header + function_block
    function_name = func_type + dim_suffix

    return output_code, function_name

    # # 7) Write to the output file.
    # with open(output_path, "w") as f:
    #     f.write(output_code)

    # print(f"Generated '{output_path}' with function '{func_type}' successfully.")

def generate_batch_norm_code(
    template_path,   # Path to the batch_norm_template.cpp file
    DATA_TYPE="float",
    C_OUT=32,
    H=64,
    W=64,
    EPSILON=0.00001
):
    """
    Reads an external batch norm template and substitutes placeholders to generate
    the final HLS C code for batch normalization.
    
    The expected placeholders in the template are:
      {DATA_TYPE}, {C_OUT}, {H}, {W}, {EPSILON}
    """
    # 1) Read the template file
    with open(template_path, "r") as f:
        template_code = f.read()
    
    # 2) Substitute the placeholders with actual parameters
    generated_code = template_code.format(
        DATA_TYPE=DATA_TYPE,
        C_OUT=C_OUT,
        H=H,
        W=W,
        EPSILON=EPSILON
    )
    DATA_TYPE = replace_data_type(DATA_TYPE)
    dim_suffix = f"_{C_OUT}_{H}_{W}_{DATA_TYPE}"
    func_name = "batch_norm" + dim_suffix
    # Use regex to capture the function signature of maxpool.
    # We assume the template defines the function starting with "void maxpool(".
    new_generated_code = re.sub(
        r"(void\s+batch_norm)\s*\(",
        lambda m: m.group(1) + dim_suffix + "(",
        generated_code,
        count=1
    )
    
    return new_generated_code, func_name


def generate_activation_function(
    template_path,      # Path to activations_template.cpp
    func_name,          # One of: "relu", "leaky_relu", "prelu", "rrelu", "thresholded_relu", "relu6", "sigmoid", "tanh_act", "elu", "selu", "gelu", "swish", "softmax"
    DATA_TYPE="float",
    C=16,
    H=64,
    W=64
):
    """
    Reads an external activation functions template, substitutes common placeholders,
    extracts the specified function block (for a 3D tensor with dimensions [C][H][W]),
    appends the dimension information to the function name, and writes the final HLS C code to output_path.
    
    Expected placeholders in the template:
      {DATA_TYPE}, {C}, {H}, {W}
    """
    # 1) Read the full template file.
    with open(template_path, "r") as f:
        template_code = f.read()
    
    # 2) Substitute common placeholders.
    formatted_code = template_code.format(
        DATA_TYPE=DATA_TYPE,
        C=C,
        H=H,
        W=W
    )
    
    # 3) Set marker strings based on the requested function name.
    func_name = func_name.lower()  # make case-insensitive
    marker_start = ""
    marker_end = ""
    
    if func_name == "relu":
        marker_start = "/*==== RELU FUNCTION START ====*/"
        marker_end = "/*==== RELU FUNCTION END ====*/"
    elif func_name == "leaky_relu":
        marker_start = "/*==== LEAKY_RELU FUNCTION START ====*/"
        marker_end = "/*==== LEAKY_RELU FUNCTION END ====*/"
    elif func_name == "prelu":
        marker_start = "/*==== PRELU FUNCTION START ====*/"
        marker_end = "/*==== PRELU FUNCTION END ====*/"
    elif func_name == "rrelu":
        marker_start = "/*==== RRELU FUNCTION START ====*/"
        marker_end = "/*==== RRELU FUNCTION END ====*/"
    elif func_name == "thresholded_relu":
        marker_start = "/*==== THRESHOLDED_RELU FUNCTION START ====*/"
        marker_end = "/*==== THRESHOLDED_RELU FUNCTION END ====*/"
    elif func_name == "relu6":
        marker_start = "/*==== RELU6 FUNCTION START ====*/"
        marker_end = "/*==== RELU6 FUNCTION END ====*/"
    elif func_name == "sigmoid":
        marker_start = "/*==== SIGMOID FUNCTION START ====*/"
        marker_end = "/*==== SIGMOID FUNCTION END ====*/"
    elif func_name == "tanh":
        marker_start = "/*==== TANH FUNCTION START ====*/"
        marker_end = "/*==== TANH FUNCTION END ====*/"
    elif func_name == "elu":
        marker_start = "/*==== ELU FUNCTION START ====*/"
        marker_end = "/*==== ELU FUNCTION END ====*/"
    elif func_name == "selu":
        marker_start = "/*==== SELU FUNCTION START ====*/"
        marker_end = "/*==== SELU FUNCTION END ====*/"
    elif func_name == "gelu":
        marker_start = "/*==== GELU FUNCTION START ====*/"
        marker_end = "/*==== GELU FUNCTION END ====*/"
    elif func_name == "swish":
        marker_start = "/*==== SWISH FUNCTION START ====*/"
        marker_end = "/*==== SWISH FUNCTION END ====*/"
    elif func_name == "softmax":
        marker_start = "/*==== SOFTMAX FUNCTION START ====*/"
        marker_end = "/*==== SOFTMAX FUNCTION END ====*/"
    elif func_name == "hardsigmoid":
        marker_start = "/*==== HARDSIGMOID FUNCTION START ====*/"
        marker_end = "/*==== HARDSIGMOID FUNCTION END ====*/"
    elif func_name == "hardswish":
        marker_start = "/*==== HARDSWISH FUNCTION START ====*/"
        marker_end = "/*==== HARDSWISH FUNCTION END ====*/"
    else:
        raise ValueError("Invalid function name. Choose from: relu, leaky_relu, prelu, rrelu, thresholded_relu, relu6, sigmoid, tanh_act, elu, selu, gelu, swish, softmax.")
    
    # 4) Extract the function block.
    start_idx = formatted_code.find(marker_start)
    end_idx = formatted_code.find(marker_end)
    if start_idx == -1 or end_idx == -1:
        raise ValueError("Could not find the specified function markers in the template.")
    
    # Skip the marker text.
    function_block = formatted_code[start_idx + len(marker_start): end_idx].strip()
    
    # 5) Append dimension information to the function name.
    # We'll assume the function signature starts with "void <name>(".
    DATA_TYPE = replace_data_type(DATA_TYPE)
    dim_suffix = f"_{C}_{H}_{W}_{DATA_TYPE}"
    # Use a regex to capture "void" followed by the function name
    function_block = re.sub(
        r"(void\s+(\w+))\s*\(",
        lambda m: m.group(1) + dim_suffix + "(", 
        function_block,
        count=1
    )
    
    # 6) Optionally, include a header (everything before the first marker).
    header_end = formatted_code.find("/*====")
    if header_end != -1:
        header = formatted_code[:header_end].strip() + "\n\n"
    else:
        header = ""
    
    output_code = header + function_block
    
    return output_code, func_name + dim_suffix


def generate_maxpool_code(
    template_path,    # Path to maxpool_template.cpp
    DATA_TYPE="float",
    C=16,
    H_IN=64,
    W_IN=64,
    H_OUT=32,
    W_OUT=32,
    K_H=2,
    K_W=2,
    STRIDE_H=2,
    STRIDE_W=2
):
    """
    Reads the maxpool_template.cpp file, substitutes placeholders, and writes
    the final HLS C code for a maxpooling function.
    Expected placeholders:
      {DATA_TYPE}, {C}, {H_IN}, {W_IN}, {H_OUT}, {W_OUT},
      {K_H}, {K_W}, {STRIDE_H}, {STRIDE_W}
    """
    # 1) Read the template
    with open(template_path, "r") as f:
        template_code = f.read()
    
    # 2) Substitute the placeholders with provided parameters
    generated_code = template_code.format(
        DATA_TYPE=DATA_TYPE,
        C=C,
        H_IN=H_IN,
        W_IN=W_IN,
        H_OUT=H_OUT,
        W_OUT=W_OUT,
        K_H=K_H,
        K_W=K_W,
        STRIDE_H=STRIDE_H,
        STRIDE_W=STRIDE_W
    )
    DATA_TYPE = replace_data_type(DATA_TYPE)
    dim_suffix = f"_{C}_{H_IN}_{W_IN}_{H_OUT}_{W_OUT}_{K_H}_{K_W}_{STRIDE_H}_{STRIDE_W}_{DATA_TYPE}"
    
    # Use regex to capture the function signature of maxpool.
    # We assume the template defines the function starting with "void maxpool(".
    new_generated_code = re.sub(
        r"(void\s+maxpool)\s*\(",
        lambda m: m.group(1) + dim_suffix + "(",
        generated_code,
        count=1
    )
    
    func_name = "maxpool" + dim_suffix
    
    return new_generated_code, func_name

def generate_adaptive_avgpool_code(
    template_path,   # Path to adaptive_avgpool_template.cpp
    DATA_TYPE="float",
    C=3,
    H_IN=64,
    W_IN=64,
    H_OUT=1,
    W_OUT=1
):
    """
    Reads the AdaptiveAvgPool template and substitutes placeholders:
      {DATA_TYPE}, {C}, {H_IN}, {W_IN}, {H_OUT}, {W_OUT}.
    
    The generated function has the signature:
      void adaptive_avgpool(data_t input[C][H_IN][W_IN], data_t output[C][H_OUT][W_OUT])
    
    It computes each output element as the average over a corresponding region in the input.
    """
    with open(template_path, "r") as f:
        template_code = f.read()
    
    generated_code = template_code.format(
        DATA_TYPE=DATA_TYPE,
        C=C,
        H_IN=H_IN,
        W_IN=W_IN,
        H_OUT=H_OUT,
        W_OUT=W_OUT
    )
    
    DATA_TYPE = replace_data_type(DATA_TYPE)
    dim_suffix = f"_{C}_{H_IN}_{W_IN}_{H_OUT}_{W_OUT}_{DATA_TYPE}"
    
    # Use regex to capture the function signature of maxpool.
    # We assume the template defines the function starting with "void maxpool(".
    new_generated_code = re.sub(
        r"(void\s+adaptive_avgpool)\s*\(",
        lambda m: m.group(1) + dim_suffix + "(",
        generated_code,
        count=1
    )
    
    func_name = "adaptive_avgpool" + dim_suffix
    
    return new_generated_code, func_name

def generate_matrix_add_code(
    template_path,    # Path to matrix_add_template.cpp
    DATA_TYPE="float",
    C=3,
    H=64,
    W=64
):
    """
    Reads the matrix addition template and substitutes the placeholders with the provided parameters.
    The template uses the following placeholders:
       {DATA_TYPE}, {C}, {H}, {W}
    
    The final function has the following signature:
       void matrix_add(data_t in1[C][H][W], data_t in2[C][H][W], data_t out[C][H][W])
    
    Parameters:
       template_path: path to the matrix_add_template.cpp file.
       output_path: path to output the generated C code.
       DATA_TYPE: the C data type for the operation (e.g., "float").
       C: number of channels.
       H: height.
       W: width.
    """
    # Read the template file.
    with open(template_path, "r") as f:
        template_code = f.read()
    
    # Substitute the placeholders.
    generated_code = template_code.format(
        DATA_TYPE=DATA_TYPE,
        C=C,
        H=H,
        W=W
    )
    
    DATA_TYPE = replace_data_type(DATA_TYPE)
    dim_suffix = f"_{C}_{H}_{W}_{DATA_TYPE}"
    
    # Use regex to capture the function signature of maxpool.
    # We assume the template defines the function starting with "void maxpool(".
    new_generated_code = re.sub(
        r"(void\s+matrix_add)\s*\(",
        lambda m: m.group(1) + dim_suffix + "(",
        generated_code,
        count=1
    )
    
    func_name = "matrix_add" + dim_suffix
    
    return new_generated_code, func_name
    
    


def generate_func_def(op_info, data_type):
    
    if op_info['func_name'] == 'load':
        code_line, full_func_name = generate_load_function(op_info["dims"], data_type, func_prefix="load")
    elif op_info['func_name'] == 'store':
        code_line, full_func_name = generate_store_function(op_info["dims"], data_type, func_prefix="store")
    elif op_info['func_name'] == 'conv':
        code_line, full_func_name = generate_conv_function(op_info["func_info"][0], op_info["func_info"][1], data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3], op_info["dims"][4], op_info["dims"][5], op_info["dims"][6], op_info["dims"][7], op_info["dims"][8], op_info["func_info"][2], op_info["dims"][9], op_info["dims"][10], op_info["dims"][11], op_info["dims"][12], op_info["dims"][13], op_info["dims"][14], op_info["dims"][15])
    elif op_info['func_name'] == 'batchnorm':
        code_line, full_func_name = generate_batch_norm_code(op_info["func_info"][0], data_type,  op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["func_info"][1])
    elif op_info['func_name'] == 'activation':
        code_line, full_func_name = generate_activation_function(op_info["func_info"][0], op_info["func_info"][1], data_type,  op_info["dims"][0], op_info["dims"][1], op_info["dims"][2])
    elif op_info['func_name'] == 'maxpool':
        code_line, full_func_name = generate_maxpool_code(op_info["func_info"][0], data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3], op_info["dims"][4], op_info["dims"][5], op_info["dims"][6], op_info["dims"][7], op_info["dims"][8])
    elif op_info['func_name'] == 'matrix_add':
        code_line, full_func_name = generate_matrix_add_code(op_info["func_info"][0], data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2])
    else:
        print("the operator we do not support!")
        
    return code_line, full_func_name
    

def generate_operator_call(op_info, data_type):
    """
    Generates a single function call string given an operator dictionary.
    
    op_info: dict with keys:
       - "func_name": base function name (e.g., "load", "conv2d", etc.)
       - "dims": list of integers (e.g., [32,64,64] or [32,64,64,32,64,64])
       - "args": list of argument strings for the function call
       
    Returns a string like:
       load_32_64_64_data_t(DRAM_1, BRAM_1);
    """

    # Append _data_t to follow the convention.
    if op_info['func_name'] == 'load':
        code_line, full_func_name = generate_load_function(op_info["dims"], data_type, func_prefix="load")
    elif op_info['func_name'] == 'store':
        code_line, full_func_name = generate_store_function(op_info["dims"], data_type, func_prefix="store")
    elif op_info['func_name'] == 'conv':
        code_line, full_func_name = generate_conv_function(op_info["func_info"][0], op_info["func_info"][1], data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3], op_info["dims"][4], op_info["dims"][5], op_info["dims"][6], op_info["dims"][7], op_info["dims"][8], op_info["func_info"][2], op_info["dims"][9], op_info["dims"][10], op_info["dims"][11], op_info["dims"][12], op_info["dims"][13], op_info["dims"][14], op_info["dims"][15])
    elif op_info['func_name'] == 'batchnorm':
        code_line, full_func_name = generate_batch_norm_code(op_info["func_info"][0], data_type,  op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["func_info"][1])
    elif op_info['func_name'] == 'activation':
        code_line, full_func_name = generate_activation_function(op_info["func_info"][0], op_info["func_info"][1], data_type,  op_info["dims"][0], op_info["dims"][1], op_info["dims"][2])
    elif op_info['func_name'] == 'maxpool':
        code_line, full_func_name = generate_maxpool_code(op_info["func_info"][0], data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3], op_info["dims"][4], op_info["dims"][5], op_info["dims"][6], op_info["dims"][7], op_info["dims"][8])
    elif op_info['func_name'] == 'matrix_add':
        code_line, full_func_name = generate_matrix_add_code(op_info["func_info"][0], data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2])
    else:
        print("the operator we do not support!")
        
    args_str = ", ".join(op_info["args"])
    return f"{full_func_name}({args_str});"


def generate_top_function(brams, drams, ops, data_type="float", top_func_name="top"):
    """
    Generates the complete HLS C top function.
    
    Parameters:
      - brams: list of dicts for BRAM arrays. Each dict has:
           "name": string, array name.
           "dims": list of ints for dimensions.
      - drams: list of dicts for DRAM arrays used in the top function. Each dict has:
           "name": string, array name.
           "dims": list of ints for dimensions.
           "bundle": string, the bundle name for the HLS interface pragma.
      - ops: a dictionary (or ordered dict) where each key is an operator name 
             (for ordering) and the value is a dict with keys "func_name", "dims", "args".
      - data_type: the C data type for data_t.
      - top_func_name: name of the top function.
      
    Returns a string containing the generated HLS C code.
    """
    code_lines = []
    
    code_lines.append("")
    code_lines.append(f"#include <stdio.h>")
    code_lines.append(f"#include <iostream>")
    code_lines.append(f"#include <fstream>")
    code_lines.append(f"#include <cstdlib>")
    code_lines.append(f"#include <ap_fixed.h>")
    code_lines.append(f"#include <hls_math.h>")
    code_lines.append(f"#include <stdlib.h>")
    code_lines.append(f"#include <cstdint>")
    code_lines.append(f"#include <hls_math.h>")
    code_lines.append(f"using namespace std;\n")
    
    # 1. Write typedef.
    code_lines.append(f"typedef {data_type} data_t;\n")
    
    # 2. Declare BRAM arrays.
    for bram in brams:
        dims_str = "".join(f"[{d}]" for d in bram["dims"])
        code_lines.append(f"data_t {bram['name']}{dims_str};")
    
    code_lines.append("")  # blank line
    
    func_name_set = set()
    func_def_name_list = []
   
    for key, op_info in ops.items():
       func_def_code, func_name  = generate_func_def(op_info, data_type)
       if func_name not in func_name_set:
           func_name_set.add(func_name)
           code_lines.append(func_def_code)
           code_lines.append("")
           func_def_name_list.append(func_name)
    
    # 3. Build the top function signature with DRAM parameters.
    dram_params = []
    for dram in drams:
        dims_str = "".join(f"[{d}]" for d in dram["dims"])
        dram_params.append(f"data_t {dram['name']}{dims_str}")
    params_str = ", ".join(dram_params)
    code_lines.append(f"void {top_func_name}({params_str})")
    code_lines.append("{")
    
    # 4. Insert the #pragma HLS interface lines for each DRAM.
    for dram in drams:
        code_lines.append(f"    #pragma HLS interface m_axi port={dram['name']} offset=slave bundle={dram['bundle']}")
    
    code_lines.append("")  # blank line before function calls
    
    # 5. Generate the operator function calls.
    # If ops is a dictionary, we iterate in insertion order.
    for key, op_info in ops.items():
        #print("the value of key is:", key)
        call_str = generate_operator_call(op_info, data_type)
        code_lines.append(f"    {call_str}")
    
    code_lines.append("}")
    
    return "\n".join(code_lines)


def prod(lst):
    """Return the product of all elements in the list."""
    result = 1
    for x in lst:
        result *= x
    return result

def generate_top_h(drams, data_type="float", top_func_name="top"):
    """
    Generates a top.h header file that declares the top function.
    
    Parameters:
      - drams: list of dictionaries for DRAM arrays. Each dict must have:
           "name": string (e.g., "DRAM_1"),
           "dims": list of integers.
      - data_type: string for data type (e.g., "float").
      - top_func_name: name of the top function.
    
    Returns:
      A string containing the header file content.
    """
    lines = []
    lines.append("#include <ap_fixed.h>")
    lines.append("#ifndef TOP_H")
    lines.append("#define TOP_H")
    lines.append("")
    lines.append(f"typedef {data_type} data_t;")
    lines.append("")
    
    # Build function parameter list for DRAM arrays.
    params = []
    for dram in drams:
        dims_str = "".join(f"[{d}]" for d in dram["dims"])
        params.append(f"data_t {dram['name']}{dims_str}")
    param_str = ", ".join(params)
    lines.append(f"void {top_func_name}({param_str});")
    lines.append("")
    lines.append("#endif // TOP_H")
    
    return "\n".join(lines)

def generate_testbench_code(drams, output_dram_names, data_type="float", top_func_name="top"):
    """
    Generates a C test bench for HLS that:
      - Declares DRAM arrays using the specified dimensions.
      - Loads initial values from text files (named "<DRAM_name>.txt").
      - Calls the top function.
      - Writes the output of the DRAM(s) specified in output_dram_names to separate output files.
    
    Parameters:
      - drams: list of dictionaries for DRAM arrays. Each dict must have:
            "name": string (e.g., "DRAM_1"),
            "dims": list of integers,
            "bundle": string (used in the interface pragma).
      - output_dram_names: list of strings; each is the name of a DRAM whose contents should be printed.
      - data_type: string representing the data type (e.g., "float").
      - top_func_name: name of the top function.
      
    Returns:
      A string containing the complete C test bench code.
    """
    code_lines = []
    # Include headers.
    code_lines.append('#include <stdio.h>')
    code_lines.append('#include <stdlib.h>')
    code_lines.append("#include <ap_fixed.h>")
    code_lines.append('#include "top.h"  // Include the top function declaration')
    code_lines.append("")
    
    # Define data type.
    code_lines.append(f"typedef {data_type} data_t;")
    code_lines.append("")
    
    # Declare DRAM arrays.
    for dram in drams:
        dims_str = "".join(f"[{d}]" for d in dram["dims"])
        code_lines.append(f"data_t {dram['name']}{dims_str};")
    code_lines.append("")
    
    # Helper function to load a text file into an array.
    code_lines.append("void load_txt_to_array(const char *filename, data_t *array, int total_size) {")
    code_lines.append("    FILE *fp = fopen(filename, \"r\");")
    code_lines.append("    if (fp == NULL) {")
    code_lines.append("        printf(\"Failed to open %s\\n\", filename);")
    code_lines.append("        exit(1);")
    code_lines.append("    }")
    code_lines.append("    for (int i = 0; i < total_size; i++) {")
    code_lines.append("        float temp;")
    code_lines.append("        fscanf(fp, \"%f\", &temp);")
    code_lines.append("        array[i] = (data_t)temp;")
    code_lines.append("    }")
    code_lines.append("    fclose(fp);")
    code_lines.append("}")
    code_lines.append("")
    
    # Main function.
    code_lines.append("int main() {")
    
    # For each DRAM, generate a load call.
    for dram in drams:
        total_elements = prod(dram["dims"])
        code_lines.append(f"    load_txt_to_array(\"{dram['name']}.txt\", (data_t*){dram['name']}, {total_elements});")
    code_lines.append("")
    
    # Insert top function call. DRAM arguments in the order of the drams list.
    dram_args = ", ".join(d["name"] for d in drams)
    code_lines.append(f"    {top_func_name}({dram_args});")
    code_lines.append("")
    
    # For each output DRAM specified in output_dram_names, generate printing code.
    for out_name in output_dram_names:
        # Find the DRAM in the configuration.
        matched = None
        for dram in drams:
            if dram["name"] == out_name:
                matched = dram
                break
        if matched is None:
            raise ValueError(f"Output DRAM {out_name} not found in DRAM configuration.")
        total_out = prod(matched["dims"])
        code_lines.append(f"    // Write contents of {out_name} to {out_name}_output.txt")
        code_lines.append("    {")
        code_lines.append(f"        FILE *fp = fopen(\"{out_name}_output.txt\", \"w\");")
        code_lines.append("        if (fp != NULL) {")
        code_lines.append(f"            for (int i = 0; i < {total_out}; i++) {{")
        code_lines.append(f"                fprintf(fp, \"%f \", (float)((data_t*){out_name})[i]);")
        code_lines.append("            }")
        code_lines.append("            fclose(fp);")
        code_lines.append("        }")
        code_lines.append("    }")
        code_lines.append("")
    
    code_lines.append("    return 0;")
    code_lines.append("}")
    
    return "\n".join(code_lines)


# def generate_dram_txt_files(drams, path, seed=None):
#     """
#     For each DRAM in the configuration, generate a .txt file containing random numbers
#     between 0 and 1, one per line.
    
#     Parameters:
#       - drams: a list of dictionaries. Each dictionary should have:
#            "name": string, e.g., "DRAM_1"
#            "dims": list of integers, e.g., [2, 4, 4]
#       - seed: optional integer seed for reproducibility.
#     """
#     if seed is not None:
#         random.seed(seed)
    
#     for dram in drams:
#         total_elements = prod(dram["dims"])
#         filename = path + f"{dram['name']}.txt"
#         with open(filename, "w") as f:
#             # Generate random numbers between 0 and 1.
#             numbers = [str(random.random()) for _ in range(total_elements)]
#             # Write each number on a new line.
#             f.write("\n".join(numbers))
#         print(f"Generated {filename} with {total_elements} random numbers.")

def generate_dram_txt_files(drams, seed=None):
    """
    For each DRAM in the configuration, generate a .txt file containing random numbers
    between 0 and 1, one per line.
    
    Parameters:
      - drams: a list of dictionaries. Each dictionary should have:
           "name": string, e.g., "DRAM_1"
           "dims": list of integers, e.g., [2, 4, 4]
      - seed: optional integer seed for reproducibility.
    """
    if seed is not None:
        random.seed(seed)
    
    for dram in drams:
        total_elements = prod(dram["dims"])
        filename = f"{dram['name']}.txt"
        with open(filename, "w") as f:
            # Generate random numbers between 0 and 1.
            #numbers = [str(random.random()) for _ in range(total_elements)]
            numbers = [str(0) for _ in range(total_elements)]
            # Write each number on a new line.
            #f.write("\n".join(numbers))
            f.write("\n".join(numbers))
        print(f"Generated {filename} with {total_elements} random numbers.")
        
def generate_full_tcl_file(drams, FPGA_name, clock_period, task, output_filename="design.tcl"):
    """
    Generates a TCL file for HLS that includes the add_files -tb commands for each DRAM.
    
    Parameters:
      - drams: list of dictionaries, each with a "name" key (e.g., "DRAM_1")
      - output_filename: the name of the TCL file to generate.
      
    The generated TCL file will contain lines like:
        add_files -tb DRAM_1.txt
        add_files -tb DRAM_2.txt
        ...
    """
    # You can add a header if needed.
    lines = []
    lines.append("# Auto-generated TCL file for HLS")
    lines.append("open_project -reset project_1")
    lines.append("")
    lines.append("set_top top")
    lines.append("")
    lines.append("add_files  top.cpp")
    lines.append("add_files -tb tb_top.cpp")
    lines.append("add_files -tb top.h")
    lines.append("")

    # Generate add_files lines for each DRAM based on user configuration.
    for dram in drams:
        lines.append(f"add_files -tb {dram['name']}.txt")
    
    lines.append('open_solution "solution1"')
    lines.append("")
    lines.append(f"set_part {FPGA_name}")
    lines.append("")
    lines.append(f"create_clock -period {clock_period} -name default")
    lines.append("")
    
    if "csim" in task:
        lines.append("csim_design")
        lines.append("")
    if "csynth" in task:
        lines.append("csynth_design")
        lines.append("")
    if "cosim" in task:
        lines.append("cosim_design")
        lines.append("")
    if "export_ip" in task:
        lines.append("export_design -format ip_catalog")
        lines.append("")
        
    lines.append("exit")
    # (Optional) Add other static TCL commands if needed.
    # For example:
    # lines.append("")
    # lines.append("open_project my_project")
    # lines.append("set_part {xc7z020clg484-1}")
    # etc.
    
    # Write the TCL file.
    with open(output_filename, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Generated TCL file '{output_filename}' with the following contents:")
    print("\n".join(lines))



if __name__ == "__main__":
    with open("./test_case_configs/vgg19_block1.json", "r") as f:
        config = json.load(f)
    
    # Extract configuration parameters
    brams = config["brams"]
    drams = config["drams"]
    ops = config["ops"]
    output_dram_names = config["output_dram_names"]
    FPGA_name = config["FPGA_name"]
    clock_period = config["clock_period"]
    task = config["task"]
    data_type = config["data_type"]
    top_func_name = config["top_func_name"]
    output_files = config["output_files"]
    path = output_files["hls_project_path"]
    
    if not os.path.exists(path):
        os.makedirs(path)
    # Generate the top function C++ code
    top_code = generate_top_function(brams, drams, ops, data_type=data_type, top_func_name=top_func_name)
    with open(output_files["top_cpp"], "w") as f:
        f.write(top_code)
    print(f"Generated {output_files['top_cpp']}")
    
    # Generate the top header file
    top_h_code = generate_top_h(drams, data_type=data_type, top_func_name=top_func_name)
    with open(output_files["top_h"], "w") as f:
        f.write(top_h_code)
    print(f"Generated {output_files['top_h']}")
    
    # Generate the testbench C++ code
    tb_code = generate_testbench_code(drams, output_dram_names, data_type=data_type, top_func_name=top_func_name)
    with open(output_files["tb_top_cpp"], "w") as f:
        f.write(tb_code)
    print(f"Generated {output_files['tb_top_cpp']}")
    
    # Generate DRAM initialization files (assumes a seed value is needed)
    generate_dram_txt_files(drams, seed=42)
    print("Generated DRAM initialization files.")
    
    # Generate the TCL file to launch HLS tasks
    generate_full_tcl_file(drams, FPGA_name, clock_period, task, output_filename=output_files["run_hls_tcl"])
    print(f"Generated TCL file: {output_files['run_hls_tcl']}")
