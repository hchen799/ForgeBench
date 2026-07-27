import re
import random

DEFAULT_LEGACY_CONV_CI_FACTOR = 8
DEFAULT_LEGACY_CONV_CO_FACTOR = 64

def replace_data_type(data_type: str) -> str:
    # Replace all occurrences of <, >, and , with an underscore
    result = re.sub(r'[<>,]', '_', data_type)
    # Remove all spaces
    result = result.replace(" ", "")
    return result


def normalize_conv_dims(dims):
    if len(dims) < 16:
        raise ValueError(f"conv expects at least 16 dims, got {len(dims)}: {dims}")
    return dims[:16]


def normalize_conv_tile_dims(dims):
    if len(dims) == 11:
        return dims + [DEFAULT_LEGACY_CONV_CI_FACTOR, DEFAULT_LEGACY_CONV_CO_FACTOR]
    if len(dims) < 12:
        raise ValueError(f"conv_tile expects at least 12 dims, got {len(dims)}: {dims}")
    return dims[:12]


def normalize_attention_tile_dims(dims):
    if len(dims) == 4:
        return dims + [1, 1, 1]
    if len(dims) == 5:
        return dims + [1, 1]
    if len(dims) < 5:
        raise ValueError(f"attention tile expects at least 5 dims, got {len(dims)}: {dims}")
    return dims[:7]


def normalize_clear_matrix_tile_dims(dims):
    if len(dims) == 2:
        return dims + [1]
    if len(dims) < 3:
        raise ValueError(f"clear_matrix_tile expects at least 2 dims, got {len(dims)}: {dims}")
    return dims[:3]


def normalize_init_rowmax_tile_dims(dims):
    if len(dims) == 2:
        return dims + [1]
    if len(dims) < 3:
        raise ValueError(f"init_rowmax_tile expects at least 2 dims, got {len(dims)}: {dims}")
    return dims[:3]


def normalize_matrix_tile_dims(dims):
    if len(dims) == 4:
        return dims + [1]
    if len(dims) < 5:
        raise ValueError(f"matrix tile expects at least 4 dims, got {len(dims)}: {dims}")
    return dims[:5]


def normalize_vector_tile_dims(dims):
    if len(dims) == 2:
        return dims + [1]
    if len(dims) < 3:
        raise ValueError(f"vector tile expects at least 2 dims, got {len(dims)}: {dims}")
    return dims[:3]


def normalize_layer_vector_tile_dims(dims):
    if len(dims) == 3:
        return dims + [1]
    if len(dims) < 4:
        raise ValueError(f"layer vector tile expects at least 3 dims, got {len(dims)}: {dims}")
    return dims[:4]


def normalize_weight_tile_layered_dims(dims):
    if len(dims) == 5:
        return dims + [1, 1]
    if len(dims) < 7:
        raise ValueError(f"layered weight tile expects at least 5 dims, got {len(dims)}: {dims}")
    return dims[:7]


def normalize_weight_tile_2d_dims(dims):
    if len(dims) == 4:
        return dims + [1, 1]
    if len(dims) < 6:
        raise ValueError(f"2d weight tile expects at least 4 dims, got {len(dims)}: {dims}")
    return dims[:6]


def normalize_linear_tile_dims(dims):
    if len(dims) == 3:
        return dims + [1, 1]
    if len(dims) < 5:
        raise ValueError(f"linear_tile expects at least 3 dims, got {len(dims)}: {dims}")
    return dims[:5]


def normalize_matrix_2d_op_dims(dims):
    if len(dims) == 2:
        return dims + [1]
    if len(dims) < 3:
        raise ValueError(f"2d matrix op expects at least 2 dims, got {len(dims)}: {dims}")
    return dims[:3]


def normalize_embedding_lookup_chunk_dims(dims):
    if len(dims) == 5:
        return dims + [1]
    if len(dims) < 6:
        raise ValueError(f"embedding_lookup_chunk expects at least 5 dims, got {len(dims)}: {dims}")
    return dims[:6]


def normalize_rmsnorm_tile_dims(dims):
    if len(dims) == 2:
        return dims + [1]
    if len(dims) < 3:
        raise ValueError(f"rmsnorm tile expects at least 2 dims, got {len(dims)}: {dims}")
    return dims[:3]


def normalize_kv_cache_load_group_tile_dims(dims):
    if len(dims) == 5:
        return dims + [1]
    if len(dims) < 6:
        raise ValueError(f"kv_cache_load_group_tile expects at least 5 dims, got {len(dims)}: {dims}")
    return dims[:6]


def normalize_rope_tile_dims(dims):
    if len(dims) == 3:
        return dims + [1, 1]
    if len(dims) < 5:
        raise ValueError(f"apply_rope_tile expects at least 3 dims, got {len(dims)}: {dims}")
    return dims[:5]


def normalize_attention_rowmax_dims(dims):
    if len(dims) == 3:
        return dims + [1]
    if len(dims) < 4:
        raise ValueError(f"attention_rowmax_tile expects at least 3 dims, got {len(dims)}: {dims}")
    return dims[:4]


def normalize_attention_finalize_dims(dims):
    if len(dims) == 3:
        return dims + [1, 1]
    if len(dims) < 5:
        raise ValueError(f"attention_finalize_tile expects at least 3 dims, got {len(dims)}: {dims}")
    return dims[:5]


def effective_pragma_factor(requested_factor, concrete_dim):
    requested_factor = int(requested_factor)
    concrete_dim = int(concrete_dim)
    if requested_factor <= 1 or concrete_dim <= 1:
        return 1
    return min(requested_factor, concrete_dim)


def emit_array_partition(code_lines, variable, requested_factor, concrete_dim, dim, indent="    "):
    factor = effective_pragma_factor(requested_factor, concrete_dim)
    if factor > 1:
        code_lines.append(
            f"{indent}#pragma HLS array_partition variable={variable} type=cyclic factor={factor} dim={dim}"
        )
    return factor


def emit_unroll(code_lines, requested_factor, concrete_dim, indent):
    factor = effective_pragma_factor(requested_factor, concrete_dim)
    if factor > 1:
        code_lines.append(f"{indent}#pragma HLS unroll factor={factor}")
    return factor


def generate_activation_function(
    template_path,   # Path to the activation template file (e.g., activations_template_2d.cpp)
    func_name,       # The activation function to generate (e.g., "relu6", "sigmoid", "gelu", etc.)
    DATA_TYPE="float",
    SEQ_LENGTH=128,
    HIDDEN_DIM=512
):
    """
    Reads the 2D activation function template and substitutes common placeholders:
       {DATA_TYPE}, {SEQ_LENGTH}, and {HIDDEN_DIM}.
    Then extracts the function block corresponding to the specified activation (func_name)
    and writes it to output_path.
    
    Supported func_name values (case-insensitive):
      "relu", "leaky_relu", "prelu", "rrelu", "thresholded_relu", "relu6",
      "sigmoid", "tanh_act", "elu", "selu", "gelu", "swish", "softmax".
    """
    # 1) Read the full template file.
    with open(template_path, "r") as f:
        template_code = f.read()
    
    # 2) Substitute common placeholders.
    formatted_code = template_code.format(
        DATA_TYPE=DATA_TYPE,
        SEQ_LENGTH=SEQ_LENGTH,
        HIDDEN_DIM=HIDDEN_DIM
    )
    
    # 3) Determine marker strings based on the specified activation function.
    func_name = func_name.lower()  # Normalize to lower-case
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
    elif func_name == "tanh_act":
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
    else:
        raise ValueError("Invalid function name. Choose one of: relu, leaky_relu, prelu, rrelu, thresholded_relu, relu6, sigmoid, tanh_act, elu, selu, gelu, swish, softmax.")
    
    # 4) Extract the specified function block.
    start_idx = formatted_code.find(marker_start)
    end_idx = formatted_code.find(marker_end)
    if start_idx == -1 or end_idx == -1:
        raise ValueError("Could not find the specified function markers in the template.")
    
    # Skip marker text.
    function_block = formatted_code[start_idx + len(marker_start): end_idx].strip()
    
    DATA_TYPE = replace_data_type(DATA_TYPE)
    dim_suffix = f"_{SEQ_LENGTH}_{HIDDEN_DIM}_{DATA_TYPE}"
    # Use a regex to capture "void" followed by the function name
    function_block = re.sub(
        r"(void\s+(\w+))\s*\(",
        lambda m: m.group(1) + dim_suffix + "(", 
        function_block,
        count=1
    )
    
    # 5) Optionally, include a header (everything before the first marker).
    header_end = formatted_code.find("/*====")
    header = ""
    if header_end != -1:
        header = formatted_code[:header_end].strip() + "\n\n"
    
    output_code = header + function_block
    
    return output_code, func_name + dim_suffix

def generate_activation_function_conv(
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

def generate_conv_tiled_function(
    template_path,
    func_type="conv2d_tiled",
    DATA_TYPE="float",
    C_IN=16,
    C_OUT=32,
    H_IN=64,
    W_IN=64,
    H_OUT=64,
    W_OUT=64,
    K=3,
    PAD=1,
    STRIDE=1,
    TILE_C=128,
    TILE_H=14,
    TILE_W=14,
    IC_CHUNK=32,
    OC_CHUNK=128,
    RENORM_GUARD=8
):
    with open(template_path, "r") as f:
        template_code = f.read()

    patch_h = (TILE_H - 1) * STRIDE + K
    patch_w = (TILE_W - 1) * STRIDE + K

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
        TILE_C=TILE_C,
        TILE_H=TILE_H,
        TILE_W=TILE_W,
        IC_CHUNK=IC_CHUNK,
        OC_CHUNK=OC_CHUNK,
        RENORM_GUARD=RENORM_GUARD,
        PATCH_H=patch_h,
        PATCH_W=patch_w,
    )

    if func_type == "conv2d_tiled":
        start_marker = "/*==== CONV2D_TILED FUNCTION START ====*/"
        end_marker = "/*==== CONV2D_TILED FUNCTION END ====*/"
    else:
        raise ValueError("Invalid tiled function type. Choose 'conv2d_tiled'.")

    start_index = formatted_code.find(start_marker)
    end_index = formatted_code.find(end_marker)
    if start_index == -1 or end_index == -1:
        raise ValueError("Could not find tiled convolution function markers in the template.")

    function_block = formatted_code[start_index + len(start_marker): end_index].strip()

    DATA_TYPE_modified = replace_data_type(DATA_TYPE)
    dim_suffix = (
        f"_{C_IN}_{C_OUT}_{H_IN}_{W_IN}_{H_OUT}_{W_OUT}_{K}_{PAD}_{STRIDE}"
        f"_{TILE_C}_{TILE_H}_{TILE_W}_{IC_CHUNK}_{OC_CHUNK}_{RENORM_GUARD}_{DATA_TYPE_modified}"
    )
    function_block = re.sub(
        r"(void\s+(\w+))\s*\(",
        lambda m: m.group(1) + dim_suffix + "(",
        function_block,
        count=1
    )

    header_end = formatted_code.find("/*====")
    if header_end != -1:
        header = formatted_code[:header_end].strip() + "\n\n"
    else:
        header = ""

    output_code = header + function_block
    function_name = func_type + dim_suffix

    return output_code, function_name

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


def generate_clear_tile_function(dims, data_type="float", func_prefix="clear_tile"):
    c_dim, h_dim, w_dim = dims
    data_type_tag = replace_data_type(data_type)
    func_name = f"{func_prefix}_{c_dim}_{h_dim}_{w_dim}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(data_t tile[{c_dim}][{h_dim}][{w_dim}])",
        "{",
        f"    for (int c = 0; c < {c_dim}; ++c) {{",
        f"        for (int h = 0; h < {h_dim}; ++h) {{",
        f"            for (int w = 0; w < {w_dim}; ++w) {{",
        "                tile[c][h][w] = (data_t)0;",
        "            }",
        "        }",
        "    }",
        "}",
    ]
    return "\n".join(code_lines), func_name


def generate_clear_matrix_tile_function(rows, cols, data_type="float", func_prefix="clear_matrix_tile", col_factor=1):
    data_type_tag = replace_data_type(data_type)
    func_name = f"{func_prefix}_{rows}_{cols}_{col_factor}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(data_t tile[{rows}][{cols}])",
        "{",
    ]
    emit_array_partition(code_lines, "tile", col_factor, cols, 2)
    code_lines.extend([
        f"    for (int r = 0; r < {rows}; ++r) {{",
        f"        for (int c = 0; c < {cols}; ++c) {{",
    ])
    emit_unroll(code_lines, col_factor, cols, "            ")
    code_lines.extend([
        "            tile[r][c] = (data_t)0;",
        "        }",
        "    }",
        "}",
    ])
    return "\n".join(code_lines), func_name


def generate_init_rowmax_tile_function(rows, cols, data_type="float", col_factor=1):
    data_type_tag = replace_data_type(data_type)
    func_name = f"init_rowmax_tile_{rows}_{cols}_{col_factor}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(data_t tile[{rows}][{cols}])",
        "{",
    ]
    emit_array_partition(code_lines, "tile", col_factor, cols, 2)
    code_lines.extend([
        f"    for (int r = 0; r < {rows}; ++r) {{",
        f"        for (int c = 0; c < {cols}; ++c) {{",
    ])
    emit_unroll(code_lines, col_factor, cols, "            ")
    code_lines.extend([
        "            tile[r][c] = (data_t)(-8);",
        "        }",
        "    }",
        "}",
    ])
    return "\n".join(code_lines), func_name


def generate_load_matrix_tile_function(data_type="float", rows=2048, cols=4096, tile_rows=16, tile_cols=128, col_factor=1):
    data_type_tag = replace_data_type(data_type)
    func_name = f"load_matrix_tile_{rows}_{cols}_{tile_rows}_{tile_cols}_{col_factor}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t input[{rows}][{cols}],",
        f"    data_t output[{tile_rows}][{tile_cols}],",
        "    int row_base,",
        "    int col_base",
        ")",
        "{",
    ]
    emit_array_partition(code_lines, "output", col_factor, tile_cols, 2)
    code_lines.extend([
        f"    for (int r = 0; r < {tile_rows}; ++r) {{",
        f"        for (int c = 0; c < {tile_cols}; ++c) {{",
    ])
    emit_unroll(code_lines, col_factor, tile_cols, "            ")
    code_lines.extend([
        "            int src_r = row_base + r;",
        "            int src_c = col_base + c;",
        f"            if (src_r < {rows} && src_c < {cols}) {{",
        "                output[r][c] = input[src_r][src_c];",
        "            } else {",
        "                output[r][c] = (data_t)0;",
        "            }",
        "        }",
        "    }",
        "}",
    ])
    return "\n".join(code_lines), func_name


def generate_store_matrix_tile_function(data_type="float", rows=2048, cols=4096, tile_rows=16, tile_cols=128, col_factor=1):
    data_type_tag = replace_data_type(data_type)
    func_name = f"store_matrix_tile_{rows}_{cols}_{tile_rows}_{tile_cols}_{col_factor}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t input[{tile_rows}][{tile_cols}],",
        f"    data_t output[{rows}][{cols}],",
        "    int row_base,",
        "    int col_base,",
        "    int valid_rows,",
        "    int valid_cols",
        ")",
        "{",
    ]
    emit_array_partition(code_lines, "input", col_factor, tile_cols, 2)
    code_lines.extend([
        "    for (int r = 0; r < valid_rows; ++r) {",
        "        for (int c = 0; c < valid_cols; ++c) {",
    ])
    emit_unroll(code_lines, col_factor, tile_cols, "            ")
    code_lines.extend([
        "            output[row_base + r][col_base + c] = input[r][c];",
        "        }",
        "    }",
        "}",
    ])
    return "\n".join(code_lines), func_name


def generate_load_layer_vector_function(data_type="float", layers=32, dim=4096):
    data_type_tag = replace_data_type(data_type)
    func_name = f"load_layer_vector_{layers}_{dim}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t input[{layers}][{dim}],",
        f"    data_t output[{dim}],",
        "    int layer_idx",
        ")",
        "{",
        f"    for (int c = 0; c < {dim}; ++c) {{",
        "        output[c] = input[layer_idx][c];",
        "    }",
        "}",
    ]
    return "\n".join(code_lines), func_name


def generate_clear_vector_tile_function(rows=16, data_type="float"):
    data_type_tag = replace_data_type(data_type)
    func_name = f"clear_vector_tile_{rows}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(data_t output[{rows}])",
        "{",
        f"    for (int r = 0; r < {rows}; ++r) {{",
        "        output[r] = (data_t)0;",
        "    }",
        "}",
    ]
    return "\n".join(code_lines), func_name


def generate_load_vector_tile_function(data_type="float", dim=4096, tile_cols=128, col_factor=1):
    data_type_tag = replace_data_type(data_type)
    func_name = f"load_vector_tile_{dim}_{tile_cols}_{col_factor}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t input[{dim}],",
        f"    data_t output[{tile_cols}],",
        "    int col_base",
        ")",
        "{",
    ]
    emit_array_partition(code_lines, "output", col_factor, tile_cols, 1)
    code_lines.extend([
        f"    for (int c = 0; c < {tile_cols}; ++c) {{",
    ])
    emit_unroll(code_lines, col_factor, tile_cols, "        ")
    code_lines.extend([
        "        int src_c = col_base + c;",
        f"        if (src_c < {dim}) {{",
        "            output[c] = input[src_c];",
        "        } else {",
        "            output[c] = (data_t)0;",
        "        }",
        "    }",
        "}",
    ])
    return "\n".join(code_lines), func_name


def generate_load_layer_vector_tile_function(data_type="float", layers=32, dim=4096, tile_cols=128, col_factor=1):
    data_type_tag = replace_data_type(data_type)
    func_name = f"load_layer_vector_tile_{layers}_{dim}_{tile_cols}_{col_factor}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t input[{layers}][{dim}],",
        f"    data_t output[{tile_cols}],",
        "    int layer_idx,",
        "    int col_base",
        ")",
        "{",
    ]
    emit_array_partition(code_lines, "output", col_factor, tile_cols, 1)
    code_lines.extend([
        f"    for (int c = 0; c < {tile_cols}; ++c) {{",
    ])
    emit_unroll(code_lines, col_factor, tile_cols, "        ")
    code_lines.extend([
        "        int src_c = col_base + c;",
        f"        if (src_c < {dim}) {{",
        "            output[c] = input[layer_idx][src_c];",
        "        } else {",
        "            output[c] = (data_t)0;",
        "        }",
        "    }",
        "}",
    ])
    return "\n".join(code_lines), func_name


def generate_load_weight_tile_layered_function(data_type="float", layers=32, out_dim=4096, in_dim=4096, out_tile=256, in_tile=128, out_factor=1, in_factor=1):
    data_type_tag = replace_data_type(data_type)
    func_name = f"load_weight_tile_layered_{layers}_{out_dim}_{in_dim}_{out_tile}_{in_tile}_{out_factor}_{in_factor}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t input[{layers}][{out_dim}][{in_dim}],",
        f"    data_t output[{out_tile}][{in_tile}],",
        "    int layer_idx,",
        "    int out_base,",
        "    int in_base",
        ")",
        "{",
    ]
    emit_array_partition(code_lines, "output", out_factor, out_tile, 1)
    emit_array_partition(code_lines, "output", in_factor, in_tile, 2)
    code_lines.extend([
        f"    for (int o = 0; o < {out_tile}; ++o) {{",
        f"        for (int i = 0; i < {in_tile}; ++i) {{",
    ])
    emit_unroll(code_lines, in_factor, in_tile, "            ")
    code_lines.extend([
        "            int src_o = out_base + o;",
        "            int src_i = in_base + i;",
        f"            if (src_o < {out_dim} && src_i < {in_dim}) {{",
        "                output[o][i] = input[layer_idx][src_o][src_i];",
        "            } else {",
        "                output[o][i] = (data_t)0;",
        "            }",
        "        }",
        "    }",
        "}",
    ])
    return "\n".join(code_lines), func_name


def generate_load_weight_tile_2d_function(data_type="float", out_dim=128256, in_dim=4096, out_tile=256, in_tile=128, out_factor=1, in_factor=1):
    data_type_tag = replace_data_type(data_type)
    func_name = f"load_weight_tile_2d_{out_dim}_{in_dim}_{out_tile}_{in_tile}_{out_factor}_{in_factor}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t input[{out_dim}][{in_dim}],",
        f"    data_t output[{out_tile}][{in_tile}],",
        "    int out_base,",
        "    int in_base",
        ")",
        "{",
    ]
    emit_array_partition(code_lines, "output", out_factor, out_tile, 1)
    emit_array_partition(code_lines, "output", in_factor, in_tile, 2)
    code_lines.extend([
        f"    for (int o = 0; o < {out_tile}; ++o) {{",
        f"        for (int i = 0; i < {in_tile}; ++i) {{",
    ])
    emit_unroll(code_lines, in_factor, in_tile, "            ")
    code_lines.extend([
        "            int src_o = out_base + o;",
        "            int src_i = in_base + i;",
        f"            if (src_o < {out_dim} && src_i < {in_dim}) {{",
        "                output[o][i] = input[src_o][src_i];",
        "            } else {",
        "                output[o][i] = (data_t)0;",
        "            }",
        "        }",
        "    }",
        "}",
    ])
    return "\n".join(code_lines), func_name


def generate_linear_tile_function(data_type="float", tile_rows=16, in_tile=128, out_tile=256, in_factor=1, out_factor=1):
    data_type_tag = replace_data_type(data_type)
    func_name = f"linear_tile_{tile_rows}_{in_tile}_{out_tile}_{in_factor}_{out_factor}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t input[{tile_rows}][{in_tile}],",
        f"    data_t weight[{out_tile}][{in_tile}],",
        f"    data_t output[{tile_rows}][{out_tile}],",
        "    int valid_rows,",
        "    int valid_out,",
        "    int valid_in",
        ")",
        "{",
    ]
    emit_array_partition(code_lines, "input", in_factor, in_tile, 2)
    emit_array_partition(code_lines, "weight", out_factor, out_tile, 1)
    emit_array_partition(code_lines, "weight", in_factor, in_tile, 2)
    emit_array_partition(code_lines, "output", out_factor, out_tile, 2)
    code_lines.extend([
        "    for (int r = 0; r < valid_rows; ++r) {",
        "        for (int o = 0; o < valid_out; ++o) {",
    ])
    emit_unroll(code_lines, out_factor, out_tile, "            ")
    code_lines.extend([
        "            acc_t sum = output[r][o];",
        "            for (int i = 0; i < valid_in; ++i) {",
    ])
    emit_unroll(code_lines, in_factor, in_tile, "                ")
    code_lines.extend([
        "                sum += (acc_t)input[r][i] * (acc_t)weight[o][i];",
        "            }",
        "            output[r][o] = (data_t)sum;",
        "        }",
        "    }",
        "}",
    ])
    return "\n".join(code_lines), func_name


def generate_rmsnorm_tile_full_function(data_type="float", tile_rows=16, dim=4096):
    data_type_tag = replace_data_type(data_type)
    func_name = f"rmsnorm_tile_full_{tile_rows}_{dim}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t input[{tile_rows}][{dim}],",
        f"    data_t gamma[{dim}],",
        f"    data_t output[{tile_rows}][{dim}],",
        "    int valid_rows",
        ")",
        "{",
        "    const acc_t eps = (acc_t)1e-5;",
        "    for (int r = 0; r < valid_rows; ++r) {",
        "        acc_t sum_sq = (acc_t)0;",
        f"        for (int c = 0; c < {dim}; ++c) {{",
        "            sum_sq += (acc_t)input[r][c] * (acc_t)input[r][c];",
        "        }",
        f"        acc_t inv_rms = (acc_t)1 / hls::sqrt(sum_sq / (acc_t){dim} + eps);",
        f"        for (int c = 0; c < {dim}; ++c) {{",
        "            output[r][c] = (data_t)((acc_t)input[r][c] * (acc_t)gamma[c] * inv_rms);",
        "        }",
        "    }",
        "}",
    ]
    return "\n".join(code_lines), func_name


def generate_matrix_add_tile_2d_function(data_type="float", rows=16, cols=4096, col_factor=1):
    data_type_tag = replace_data_type(data_type)
    func_name = f"matrix_add_tile_2d_{rows}_{cols}_{col_factor}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t lhs[{rows}][{cols}],",
        f"    data_t rhs[{rows}][{cols}],",
        f"    data_t output[{rows}][{cols}],",
        "    int valid_rows,",
        "    int valid_cols",
        ")",
        "{",
    ]
    emit_array_partition(code_lines, "lhs", col_factor, cols, 2)
    emit_array_partition(code_lines, "rhs", col_factor, cols, 2)
    emit_array_partition(code_lines, "output", col_factor, cols, 2)
    code_lines.extend([
        "    for (int r = 0; r < valid_rows; ++r) {",
        "        for (int c = 0; c < valid_cols; ++c) {",
    ])
    emit_unroll(code_lines, col_factor, cols, "            ")
    code_lines.extend([
        "            output[r][c] = lhs[r][c] + rhs[r][c];",
        "        }",
        "    }",
        "}",
    ])
    return "\n".join(code_lines), func_name


def generate_activation_tile_2d_function(data_type="float", rows=16, cols=128, act_type="silu", col_factor=1):
    data_type_tag = replace_data_type(data_type)
    func_name = f"{act_type}_tile_2d_{rows}_{cols}_{col_factor}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t input[{rows}][{cols}],",
        f"    data_t output[{rows}][{cols}],",
        "    int valid_rows,",
        "    int valid_cols",
        ")",
        "{",
    ]
    emit_array_partition(code_lines, "input", col_factor, cols, 2)
    emit_array_partition(code_lines, "output", col_factor, cols, 2)
    code_lines.extend([
        "    for (int r = 0; r < valid_rows; ++r) {",
        "        for (int c = 0; c < valid_cols; ++c) {",
    ])
    emit_unroll(code_lines, col_factor, cols, "            ")
    if act_type == "silu":
        code_lines.append("            output[r][c] = input[r][c] / ((data_t)1 + hls::exp(-input[r][c]));")
    elif act_type == "relu":
        code_lines.append("            output[r][c] = input[r][c] > (data_t)0 ? input[r][c] : (data_t)0;")
    else:
        code_lines.append("            output[r][c] = input[r][c];")
    code_lines.extend([
        "        }",
        "    }",
        "}",
    ])
    return "\n".join(code_lines), func_name


def generate_elementwise_mult_tile_2d_function(data_type="float", rows=16, cols=128, col_factor=1):
    data_type_tag = replace_data_type(data_type)
    func_name = f"elementwise_mult_tile_2d_{rows}_{cols}_{col_factor}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t lhs[{rows}][{cols}],",
        f"    data_t rhs[{rows}][{cols}],",
        f"    data_t output[{rows}][{cols}],",
        "    int valid_rows,",
        "    int valid_cols",
        ")",
        "{",
    ]
    emit_array_partition(code_lines, "lhs", col_factor, cols, 2)
    emit_array_partition(code_lines, "rhs", col_factor, cols, 2)
    emit_array_partition(code_lines, "output", col_factor, cols, 2)
    code_lines.extend([
        "    for (int r = 0; r < valid_rows; ++r) {",
        "        for (int c = 0; c < valid_cols; ++c) {",
    ])
    emit_unroll(code_lines, col_factor, cols, "            ")
    code_lines.extend([
        "            output[r][c] = lhs[r][c] * rhs[r][c];",
        "        }",
        "    }",
        "}",
    ])
    return "\n".join(code_lines), func_name


def generate_embedding_lookup_chunk_function(data_type="float", seq_length=2048, vocab=128256, hidden=4096, tile_rows=16, tile_cols=128, col_factor=1):
    data_type_tag = replace_data_type(data_type)
    func_name = f"embedding_lookup_chunk_{seq_length}_{vocab}_{hidden}_{tile_rows}_{tile_cols}_{col_factor}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t token_ids[{seq_length}],",
        f"    data_t embedding[{vocab}][{hidden}],",
        f"    data_t output[{tile_rows}][{tile_cols}],",
        "    int token_base,",
        "    int hidden_base,",
        "    int valid_rows,",
        "    int valid_cols",
        ")",
        "{",
    ]
    emit_array_partition(code_lines, "output", col_factor, tile_cols, 2)
    code_lines.extend([
        "    for (int r = 0; r < valid_rows; ++r) {",
        "        int token_idx = (int)token_ids[token_base + r];",
        f"        if (token_idx < 0 || token_idx >= {vocab}) token_idx = 0;",
        "        for (int c = 0; c < valid_cols; ++c) {",
    ])
    emit_unroll(code_lines, col_factor, tile_cols, "            ")
    code_lines.extend([
        "            output[r][c] = embedding[token_idx][hidden_base + c];",
        "        }",
        "    }",
        "}",
    ])
    return "\n".join(code_lines), func_name


def generate_rmsnorm_accumulate_tile_function(data_type="float", tile_rows=16, tile_cols=128, col_factor=1):
    data_type_tag = replace_data_type(data_type)
    func_name = f"rmsnorm_accumulate_tile_{tile_rows}_{tile_cols}_{col_factor}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t input[{tile_rows}][{tile_cols}],",
        f"    data_t sumsq[{tile_rows}],",
        "    int valid_rows,",
        "    int valid_cols",
        ")",
        "{",
    ]
    emit_array_partition(code_lines, "input", col_factor, tile_cols, 2)
    code_lines.extend([
        "    for (int r = 0; r < valid_rows; ++r) {",
        "        acc_t sum = (acc_t)sumsq[r];",
        "        for (int c = 0; c < valid_cols; ++c) {",
    ])
    emit_unroll(code_lines, col_factor, tile_cols, "            ")
    code_lines.extend([
        "            sum += (acc_t)input[r][c] * (acc_t)input[r][c];",
        "        }",
        "        sumsq[r] = (data_t)sum;",
        "    }",
        "}",
    ])
    return "\n".join(code_lines), func_name


def generate_rmsnorm_finalize_rows_function(data_type="float", tile_rows=16, full_dim=4096):
    data_type_tag = replace_data_type(data_type)
    func_name = f"rmsnorm_finalize_rows_{tile_rows}_{full_dim}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t sumsq[{tile_rows}],",
        f"    data_t inv_rms[{tile_rows}],",
        "    int valid_rows",
        ")",
        "{",
        "    const acc_t eps = (acc_t)1e-5;",
        "    for (int r = 0; r < valid_rows; ++r) {",
        f"        inv_rms[r] = (data_t)((acc_t)1 / hls::sqrt((acc_t)sumsq[r] / (acc_t){full_dim} + eps));",
        "    }",
        "}",
    ]
    return "\n".join(code_lines), func_name


def generate_rmsnorm_apply_tile_function(data_type="float", tile_rows=16, tile_cols=128, col_factor=1):
    data_type_tag = replace_data_type(data_type)
    func_name = f"rmsnorm_apply_tile_{tile_rows}_{tile_cols}_{col_factor}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t input[{tile_rows}][{tile_cols}],",
        f"    data_t gamma[{tile_cols}],",
        f"    data_t inv_rms[{tile_rows}],",
        f"    data_t output[{tile_rows}][{tile_cols}],",
        "    int valid_rows,",
        "    int valid_cols",
        ")",
        "{",
    ]
    emit_array_partition(code_lines, "input", col_factor, tile_cols, 2)
    emit_array_partition(code_lines, "gamma", col_factor, tile_cols, 1)
    emit_array_partition(code_lines, "output", col_factor, tile_cols, 2)
    code_lines.extend([
        "    for (int r = 0; r < valid_rows; ++r) {",
        "        for (int c = 0; c < valid_cols; ++c) {",
    ])
    emit_unroll(code_lines, col_factor, tile_cols, "            ")
    code_lines.extend([
        "            output[r][c] = (data_t)((acc_t)input[r][c] * (acc_t)gamma[c] * (acc_t)inv_rms[r]);",
        "        }",
        "    }",
        "}",
    ])
    return "\n".join(code_lines), func_name


def generate_embedding_lookup_tile_function(data_type="float", seq_length=2048, vocab=128256, hidden=4096, tile_rows=16):
    data_type_tag = replace_data_type(data_type)
    func_name = f"embedding_lookup_tile_{seq_length}_{vocab}_{hidden}_{tile_rows}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t token_ids[{seq_length}],",
        f"    data_t embedding[{vocab}][{hidden}],",
        f"    data_t output[{tile_rows}][{hidden}],",
        "    int token_base,",
        "    int valid_rows",
        ")",
        "{",
        "    for (int r = 0; r < valid_rows; ++r) {",
        "        int token_idx = (int)token_ids[token_base + r];",
        f"        if (token_idx < 0 || token_idx >= {vocab}) token_idx = 0;",
        f"        for (int c = 0; c < {hidden}; ++c) {{",
        "            output[r][c] = embedding[token_idx][c];",
        "        }",
        "    }",
        "}",
    ]
    return "\n".join(code_lines), func_name


def generate_kv_cache_store_tile_function(data_type="float", src_rows=2048, max_ctx=2048, kv_heads=8, head_dim=128, tile_rows=16, kv_dim=1024):
    data_type_tag = replace_data_type(data_type)
    func_name = f"kv_cache_store_tile_{src_rows}_{max_ctx}_{kv_heads}_{head_dim}_{tile_rows}_{kv_dim}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t input[{src_rows}][{kv_dim}],",
        f"    data_t cache[32][{max_ctx}][{kv_heads}][{head_dim}],",
        "    int layer_idx,",
        "    int src_row_base,",
        "    int cache_row_base,",
        "    int valid_rows",
        ")",
        "{",
        "    for (int r = 0; r < valid_rows; ++r) {",
        f"        for (int h = 0; h < {kv_heads}; ++h) {{",
        f"            for (int d = 0; d < {head_dim}; ++d) {{",
        "                cache[layer_idx][cache_row_base + r][h][d] = input[src_row_base + r][h * 128 + d];",
        "            }",
        "        }",
        "    }",
        "}",
    ]
    return "\n".join(code_lines), func_name


def generate_kv_cache_load_tile_function(data_type="float", max_ctx=2048, kv_heads=8, head_dim=128, tile_rows=128):
    data_type_tag = replace_data_type(data_type)
    func_name = f"kv_cache_load_tile_{max_ctx}_{kv_heads}_{head_dim}_{tile_rows}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t cache[32][{max_ctx}][{kv_heads}][{head_dim}],",
        f"    data_t output[{tile_rows}][{head_dim}],",
        "    int layer_idx,",
        "    int kv_head_idx,",
        "    int cache_row_base,",
        "    int valid_rows",
        ")",
        "{",
        "    for (int r = 0; r < valid_rows; ++r) {",
        f"        for (int d = 0; d < {head_dim}; ++d) {{",
        "            output[r][d] = cache[layer_idx][cache_row_base + r][kv_head_idx][d];",
        "        }",
        "    }",
        "}",
    ]
    return "\n".join(code_lines), func_name


def generate_kv_cache_load_group_tile_function(data_type="float", max_ctx=2048, kv_heads=8, head_dim=128, tile_rows=128, local_kv_heads=1, dim_factor=1):
    data_type_tag = replace_data_type(data_type)
    local_kv_cols = local_kv_heads * head_dim
    func_name = f"kv_cache_load_group_tile_{max_ctx}_{kv_heads}_{head_dim}_{tile_rows}_{local_kv_heads}_{dim_factor}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t cache[32][{max_ctx}][{kv_heads}][{head_dim}],",
        f"    data_t output[{tile_rows}][{local_kv_cols}],",
        "    int layer_idx,",
        "    int kv_head_base,",
        "    int cache_row_base,",
        "    int valid_rows",
        ")",
        "{",
    ]
    emit_array_partition(code_lines, "output", dim_factor, local_kv_cols, 2)
    code_lines.extend([
        "    for (int r = 0; r < valid_rows; ++r) {",
        f"        for (int local_h = 0; local_h < {local_kv_heads}; ++local_h) {{",
        "            int kv_head_idx = kv_head_base + local_h;",
        f"            for (int d = 0; d < {head_dim}; ++d) {{",
    ])
    emit_unroll(code_lines, dim_factor, head_dim, "                ")
    code_lines.extend([
        "                output[r][local_h * 128 + d] = cache[layer_idx][cache_row_base + r][kv_head_idx][d];",
        "            }",
        "        }",
        "    }",
        "}",
    ])
    return "\n".join(code_lines), func_name


def generate_apply_rope_tile_function(data_type="float", tile_rows=16, local_heads=4, head_dim=128, head_factor=1, dim_factor=1):
    data_type_tag = replace_data_type(data_type)
    cols = local_heads * head_dim
    func_name = f"apply_rope_tile_{tile_rows}_{local_heads}_{head_dim}_{head_factor}_{dim_factor}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t input[{tile_rows}][{cols}],",
        f"    data_t output[{tile_rows}][{cols}],",
        "    int pos_base,",
        "    int head_base,",
        "    int valid_rows",
        ")",
        "{",
    ]
    emit_array_partition(code_lines, "input", dim_factor, cols, 2)
    emit_array_partition(code_lines, "output", dim_factor, cols, 2)
    code_lines.extend([
        "    const float rope_theta = 500000.0f;",
        "    for (int r = 0; r < valid_rows; ++r) {",
        f"        for (int h = 0; h < {local_heads}; ++h) {{",
        f"            for (int d = 0; d < {head_dim}; d += 2) {{",
    ])
    emit_unroll(code_lines, dim_factor, max(1, head_dim // 2), "                ")
    code_lines.extend([
        "                int idx = h * 128 + d;",
        "                int global_head = head_base + h;",
        "                float theta = powf(rope_theta, -((float)d) / 128.0f);",
        "                float angle = (float)(pos_base + r) * theta;",
        "                data_t cos_val = (data_t)hls::cos(angle);",
        "                data_t sin_val = (data_t)hls::sin(angle);",
        "                data_t x0 = input[r][idx];",
        "                data_t x1 = input[r][idx + 1];",
        "                output[r][idx] = x0 * cos_val - x1 * sin_val;",
        "                output[r][idx + 1] = x0 * sin_val + x1 * cos_val;",
        "            }",
        "        }",
        "    }",
        "}",
    ])
    return "\n".join(code_lines), func_name


def generate_attention_score_tile_function(data_type="float", q_rows=16, q_heads=4, k_rows=128, head_dim=128, local_kv_heads=1, head_factor=1, dim_factor=1):
    data_type_tag = replace_data_type(data_type)
    q_cols = q_heads * head_dim
    local_kv_cols = local_kv_heads * head_dim
    func_name = f"attention_score_tile_{q_rows}_{q_heads}_{k_rows}_{head_dim}_{local_kv_heads}_{head_factor}_{dim_factor}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t q_tile[{q_rows}][{q_cols}],",
        f"    data_t k_tile[{k_rows}][{local_kv_cols}],",
        f"    data_t score[{q_rows}][{q_heads}][{k_rows}],",
        "    int valid_q,",
        "    int valid_k,",
        "    int q_index_base,",
        "    int k_index_base",
        ")",
        "{",
    ]
    emit_array_partition(code_lines, "q_tile", dim_factor, q_cols, 2)
    emit_array_partition(code_lines, "k_tile", dim_factor, local_kv_cols, 2)
    emit_array_partition(code_lines, "score", head_factor, q_heads, 2)
    code_lines.extend([
        "    const data_t scale = (data_t)1.0 / hls::sqrt((data_t)128);",
        "    for (int qt = 0; qt < valid_q; ++qt) {",
        f"        for (int qh = 0; qh < {q_heads}; ++qh) {{",
        "            int local_kv_head = qh / 4;",
        "            for (int kt = 0; kt < valid_k; ++kt) {",
        "                if ((k_index_base + kt) > (q_index_base + qt)) {",
        "                    score[qt][qh][kt] = (data_t)(-8);",
        "                } else {",
        "                    acc_t sum = (acc_t)0;",
        f"                    for (int d = 0; d < {head_dim}; ++d) {{",
    ])
    emit_unroll(code_lines, dim_factor, head_dim, "                        ")
    code_lines.extend([
        "                        sum += (acc_t)q_tile[qt][qh * 128 + d] * (acc_t)k_tile[kt][local_kv_head * 128 + d];",
        "                    }",
        "                    score[qt][qh][kt] = (data_t)(sum * scale);",
        "                }",
        "            }",
        "        }",
        "    }",
        "}",
    ])
    return "\n".join(code_lines), func_name


def generate_attention_rowmax_tile_function(data_type="float", q_rows=16, q_heads=4, k_rows=128, head_factor=1):
    data_type_tag = replace_data_type(data_type)
    func_name = f"attention_rowmax_tile_{q_rows}_{q_heads}_{k_rows}_{head_factor}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t score[{q_rows}][{q_heads}][{k_rows}],",
        f"    data_t rowmax[{q_rows}][{q_heads}],",
        "    int valid_q,",
        "    int valid_k",
        ")",
        "{",
    ]
    emit_array_partition(code_lines, "score", head_factor, q_heads, 2)
    emit_array_partition(code_lines, "rowmax", head_factor, q_heads, 2)
    code_lines.extend([
        "    for (int qt = 0; qt < valid_q; ++qt) {",
        f"        for (int qh = 0; qh < {q_heads}; ++qh) {{",
    ])
    emit_unroll(code_lines, head_factor, q_heads, "            ")
    code_lines.extend([
        "            for (int kt = 0; kt < valid_k; ++kt) {",
        "                if (score[qt][qh][kt] > rowmax[qt][qh]) {",
        "                    rowmax[qt][qh] = score[qt][qh][kt];",
        "                }",
        "            }",
        "        }",
        "    }",
        "}",
    ])
    return "\n".join(code_lines), func_name


def generate_attention_softmax_context_tile_function(data_type="float", q_rows=16, q_heads=4, k_rows=128, head_dim=128, local_kv_heads=1, head_factor=1, dim_factor=1):
    data_type_tag = replace_data_type(data_type)
    ctx_cols = q_heads * head_dim
    local_kv_cols = local_kv_heads * head_dim
    func_name = f"attention_softmax_context_tile_{q_rows}_{q_heads}_{k_rows}_{head_dim}_{local_kv_heads}_{head_factor}_{dim_factor}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t score[{q_rows}][{q_heads}][{k_rows}],",
        f"    data_t v_tile[{k_rows}][{local_kv_cols}],",
        f"    data_t rowmax[{q_rows}][{q_heads}],",
        f"    data_t rowsum[{q_rows}][{q_heads}],",
        f"    data_t ctx[{q_rows}][{ctx_cols}],",
        "    int valid_q,",
        "    int valid_k",
        ")",
        "{",
    ]
    emit_array_partition(code_lines, "score", head_factor, q_heads, 2)
    emit_array_partition(code_lines, "v_tile", dim_factor, local_kv_cols, 2)
    emit_array_partition(code_lines, "rowmax", head_factor, q_heads, 2)
    emit_array_partition(code_lines, "rowsum", head_factor, q_heads, 2)
    emit_array_partition(code_lines, "ctx", dim_factor, ctx_cols, 2)
    code_lines.extend([
        "    for (int qt = 0; qt < valid_q; ++qt) {",
        f"        for (int qh = 0; qh < {q_heads}; ++qh) {{",
        "            int local_kv_head = qh / 4;",
        "            for (int kt = 0; kt < valid_k; ++kt) {",
        "                data_t weight = hls::exp(score[qt][qh][kt] - rowmax[qt][qh]);",
        "                rowsum[qt][qh] += weight;",
        f"                for (int d = 0; d < {head_dim}; ++d) {{",
    ])
    emit_unroll(code_lines, dim_factor, head_dim, "                    ")
    code_lines.extend([
        "                    ctx[qt][qh * 128 + d] += weight * v_tile[kt][local_kv_head * 128 + d];",
        "                }",
        "            }",
        "        }",
        "    }",
        "}",
    ])
    return "\n".join(code_lines), func_name


def generate_attention_finalize_tile_function(data_type="float", q_rows=16, q_heads=4, head_dim=128, head_factor=1, dim_factor=1):
    data_type_tag = replace_data_type(data_type)
    ctx_cols = q_heads * head_dim
    func_name = f"attention_finalize_tile_{q_rows}_{q_heads}_{head_dim}_{head_factor}_{dim_factor}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t ctx[{q_rows}][{ctx_cols}],",
        f"    data_t rowsum[{q_rows}][{q_heads}],",
        "    int valid_q",
        ")",
        "{",
    ]
    emit_array_partition(code_lines, "ctx", dim_factor, ctx_cols, 2)
    emit_array_partition(code_lines, "rowsum", head_factor, q_heads, 2)
    code_lines.extend([
        "    for (int qt = 0; qt < valid_q; ++qt) {",
        f"        for (int qh = 0; qh < {q_heads}; ++qh) {{",
        "            data_t denom = rowsum[qt][qh];",
        "            if (denom == (data_t)0) denom = (data_t)1;",
        f"            for (int d = 0; d < {head_dim}; ++d) {{",
    ])
    emit_unroll(code_lines, dim_factor, head_dim, "                ")
    code_lines.extend([
        "                ctx[qt][qh * 128 + d] = ctx[qt][qh * 128 + d] / denom;",
        "            }",
        "        }",
        "    }",
        "}",
    ])
    return "\n".join(code_lines), func_name


def generate_load_fmap_patch_function(
    data_type="float",
    c_in=32,
    h_in=56,
    w_in=56,
    c_tile=32,
    patch_h=33,
    patch_w=33,
    stride=1,
    pad=1,
):
    data_type_tag = replace_data_type(data_type)
    func_name = (
        f"load_fmap_patch_{c_in}_{h_in}_{w_in}_{c_tile}_{patch_h}_{patch_w}"
        f"_{stride}_{pad}_{data_type_tag}"
    )
    code_lines = [
        f"void {func_name}(",
        f"    data_t input[{c_in}][{h_in}][{w_in}],",
        f"    data_t output[{c_tile}][{patch_h}][{patch_w}],",
        "    int c_base,",
        "    int oh_base,",
        "    int ow_base",
        ")",
        "{",
        f"    for (int c = 0; c < {c_tile}; ++c) {{",
        f"        for (int h = 0; h < {patch_h}; ++h) {{",
        f"            for (int w = 0; w < {patch_w}; ++w) {{",
        "                int src_c = c_base + c;",
        f"                int src_h = oh_base * {stride} - {pad} + h;",
        f"                int src_w = ow_base * {stride} - {pad} + w;",
        f"                if (src_c < {c_in} && src_h >= 0 && src_h < {h_in} && src_w >= 0 && src_w < {w_in}) {{",
        "                    output[c][h][w] = input[src_c][src_h][src_w];",
        "                } else {",
        "                    output[c][h][w] = (data_t)0;",
        "                }",
        "            }",
        "        }",
        "    }",
        "}",
    ]
    return "\n".join(code_lines), func_name


def generate_load_fmap_tile_function(
    data_type="float",
    c_in=128,
    h_in=56,
    w_in=56,
    c_tile=128,
    tile_h=14,
    tile_w=14,
):
    data_type_tag = replace_data_type(data_type)
    func_name = (
        f"load_fmap_tile_{c_in}_{h_in}_{w_in}_{c_tile}_{tile_h}_{tile_w}_{data_type_tag}"
    )
    code_lines = [
        f"void {func_name}(",
        f"    data_t input[{c_in}][{h_in}][{w_in}],",
        f"    data_t output[{c_tile}][{tile_h}][{tile_w}],",
        "    int c_base,",
        "    int oh_base,",
        "    int ow_base",
        ")",
        "{",
        f"    for (int c = 0; c < {c_tile}; ++c) {{",
        f"        for (int h = 0; h < {tile_h}; ++h) {{",
        f"            for (int w = 0; w < {tile_w}; ++w) {{",
        "                int src_c = c_base + c;",
        "                int src_h = oh_base + h;",
        "                int src_w = ow_base + w;",
        f"                if (src_c < {c_in} && src_h >= 0 && src_h < {h_in} && src_w >= 0 && src_w < {w_in}) {{",
        "                    output[c][h][w] = input[src_c][src_h][src_w];",
        "                } else {",
        "                    output[c][h][w] = (data_t)0;",
        "                }",
        "            }",
        "        }",
        "    }",
        "}",
    ]
    return "\n".join(code_lines), func_name


def generate_store_fmap_tile_function(
    data_type="float",
    c_out=128,
    h_out=56,
    w_out=56,
    c_tile=128,
    tile_h=14,
    tile_w=14,
):
    data_type_tag = replace_data_type(data_type)
    func_name = (
        f"store_fmap_tile_{c_out}_{h_out}_{w_out}_{c_tile}_{tile_h}_{tile_w}_{data_type_tag}"
    )
    code_lines = [
        f"void {func_name}(",
        f"    data_t input[{c_tile}][{tile_h}][{tile_w}],",
        f"    data_t output[{c_out}][{h_out}][{w_out}],",
        "    int c_base,",
        "    int oh_base,",
        "    int ow_base,",
        "    int valid_c,",
        "    int valid_h,",
        "    int valid_w",
        ")",
        "{",
        "    for (int c = 0; c < valid_c; ++c) {",
        "        for (int h = 0; h < valid_h; ++h) {",
        "            for (int w = 0; w < valid_w; ++w) {",
        "                output[c_base + c][oh_base + h][ow_base + w] = input[c][h][w];",
        "            }",
        "        }",
        "    }",
        "}",
    ]
    return "\n".join(code_lines), func_name


def generate_load_weight_tile_function(
    data_type="float",
    c_out=128,
    c_in=32,
    kernel=3,
    oc_tile=128,
    ic_tile=32,
):
    data_type_tag = replace_data_type(data_type)
    func_name = (
        f"load_weight_tile_{c_out}_{c_in}_{kernel}_{oc_tile}_{ic_tile}_{data_type_tag}"
    )
    code_lines = [
        f"void {func_name}(",
        f"    data_t input[{c_out}][{c_in}][{kernel}][{kernel}],",
        f"    data_t output[{oc_tile}][{ic_tile}][{kernel}][{kernel}],",
        "    int co_base,",
        "    int ci_base",
        ")",
        "{",
        f"    for (int co = 0; co < {oc_tile}; ++co) {{",
        f"        for (int ci = 0; ci < {ic_tile}; ++ci) {{",
        f"            for (int kh = 0; kh < {kernel}; ++kh) {{",
        f"                for (int kw = 0; kw < {kernel}; ++kw) {{",
        "                    int src_co = co_base + co;",
        "                    int src_ci = ci_base + ci;",
        f"                    if (src_co < {c_out} && src_ci < {c_in}) {{",
        "                        output[co][ci][kh][kw] = input[src_co][src_ci][kh][kw];",
        "                    } else {",
        "                        output[co][ci][kh][kw] = (data_t)0;",
        "                    }",
        "                }",
        "            }",
        "        }",
        "    }",
        "}",
    ]
    return "\n".join(code_lines), func_name


def generate_load_bn_tile_function(data_type="float", c_out=128, oc_tile=128):
    data_type_tag = replace_data_type(data_type)
    func_name = f"load_bn_tile_{c_out}_{oc_tile}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t input[4][{c_out}],",
        f"    data_t output[4][{oc_tile}],",
        "    int co_base",
        ")",
        "{",
        "    for (int stat = 0; stat < 4; ++stat) {",
        f"        for (int co = 0; co < {oc_tile}; ++co) {{",
        "            int src_co = co_base + co;",
        f"            if (src_co < {c_out}) {{",
        "                output[stat][co] = input[stat][src_co];",
        "            } else {",
        "                output[stat][co] = (data_t)0;",
        "            }",
        "        }",
        "    }",
        "}",
    ]
    return "\n".join(code_lines), func_name


def generate_conv_tile_function(
    data_type="float",
    ic_tile=32,
    patch_h=33,
    patch_w=33,
    oc_tile=128,
    weight_ic=32,
    kernel=3,
    out_c=128,
    tile_h=14,
    tile_w=14,
    stride=1,
    ci_unroll_factor=DEFAULT_LEGACY_CONV_CI_FACTOR,
    co_unroll_factor=DEFAULT_LEGACY_CONV_CO_FACTOR,
):
    data_type_tag = replace_data_type(data_type)
    func_name = (
        f"conv_tile_{ic_tile}_{patch_h}_{patch_w}_{oc_tile}_{weight_ic}_{kernel}"
        f"_{out_c}_{tile_h}_{tile_w}_{stride}_{ci_unroll_factor}_{co_unroll_factor}_{data_type_tag}"
    )
    code_lines = [
        f"void {func_name}(",
        f"    data_t input[{ic_tile}][{patch_h}][{patch_w}],",
        f"    data_t weight[{oc_tile}][{weight_ic}][{kernel}][{kernel}],",
        f"    data_t output[{out_c}][{tile_h}][{tile_w}],",
        "    int valid_oc,",
        "    int valid_ic,",
        "    int valid_h,",
        "    int valid_w",
        ")",
        "{",
        f"    #pragma HLS array_partition variable=input type=cyclic factor={ci_unroll_factor} dim=1",
        f"    #pragma HLS array_partition variable=weight type=cyclic factor={co_unroll_factor} dim=1",
        f"    #pragma HLS array_partition variable=weight type=cyclic factor={ci_unroll_factor} dim=2",
        f"    #pragma HLS array_partition variable=output type=cyclic factor={co_unroll_factor} dim=1",
        "    for (int oh = 0; oh < valid_h; ++oh) {",
        "        for (int ow = 0; ow < valid_w; ++ow) {",
        f"            for (int kh = 0; kh < {kernel}; ++kh) {{",
        f"                for (int kw = 0; kw < {kernel}; ++kw) {{",
        "                    for (int ci = 0; ci < valid_ic; ++ci) {",
        f"                        #pragma HLS unroll factor={ci_unroll_factor}",
        "                        for (int co = 0; co < valid_oc; ++co) {",
        f"                            #pragma HLS unroll factor={co_unroll_factor}",
        f"                            output[co][oh][ow] += input[ci][oh * {stride} + kh][ow * {stride} + kw] * weight[co][ci][kh][kw];",
        "                        }",
        "                    }",
        "                }",
            "            }",
        "        }",
        "    }",
        "}",
    ]
    return "\n".join(code_lines), func_name


def generate_batchnorm_tile_function(data_type="float", tile_c=128, tile_h=14, tile_w=14):
    data_type_tag = replace_data_type(data_type)
    func_name = f"batchnorm_tile_{tile_c}_{tile_h}_{tile_w}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t input[{tile_c}][{tile_h}][{tile_w}],",
        f"    data_t weights[4][{tile_c}],",
        f"    data_t output[{tile_c}][{tile_h}][{tile_w}],",
        "    int valid_c,",
        "    int valid_h,",
        "    int valid_w",
        ")",
        "{",
        "    for (int c = 0; c < valid_c; ++c) {",
        "        for (int h = 0; h < valid_h; ++h) {",
        "            for (int w = 0; w < valid_w; ++w) {",
        "                data_t norm = (input[c][h][w] - weights[2][c]) / hls::sqrt(weights[3][c] + (data_t)0.00001);",
        "                output[c][h][w] = weights[0][c] * norm + weights[1][c];",
        "            }",
        "        }",
        "    }",
        "}",
    ]
    return "\n".join(code_lines), func_name


def generate_activation_tile_function(data_type="float", tile_c=128, tile_h=14, tile_w=14, act_type="relu"):
    data_type_tag = replace_data_type(data_type)
    func_name = f"{act_type}_tile_{tile_c}_{tile_h}_{tile_w}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t input[{tile_c}][{tile_h}][{tile_w}],",
        f"    data_t output[{tile_c}][{tile_h}][{tile_w}],",
        "    int valid_c,",
        "    int valid_h,",
        "    int valid_w",
        ")",
        "{",
        "    for (int c = 0; c < valid_c; ++c) {",
        "        for (int h = 0; h < valid_h; ++h) {",
        "            for (int w = 0; w < valid_w; ++w) {",
    ]
    if act_type == "relu":
        code_lines.append("                output[c][h][w] = input[c][h][w] > (data_t)0 ? input[c][h][w] : (data_t)0;")
    else:
        code_lines.append("                output[c][h][w] = input[c][h][w];")
    code_lines.extend([
        "            }",
        "        }",
        "    }",
        "}",
    ])
    return "\n".join(code_lines), func_name


def generate_matrix_add_tile_function(data_type="float", tile_c=128, tile_h=14, tile_w=14):
    data_type_tag = replace_data_type(data_type)
    func_name = f"matrix_add_tile_{tile_c}_{tile_h}_{tile_w}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t lhs[{tile_c}][{tile_h}][{tile_w}],",
        f"    data_t rhs[{tile_c}][{tile_h}][{tile_w}],",
        f"    data_t output[{tile_c}][{tile_h}][{tile_w}],",
        "    int valid_c,",
        "    int valid_h,",
        "    int valid_w",
        ")",
        "{",
        "    for (int c = 0; c < valid_c; ++c) {",
        "        for (int h = 0; h < valid_h; ++h) {",
        "            for (int w = 0; w < valid_w; ++w) {",
        "                output[c][h][w] = lhs[c][h][w] + rhs[c][h][w];",
        "            }",
        "        }",
        "    }",
        "}",
    ]
    return "\n".join(code_lines), func_name


def generate_maxpool_tile_function(
    data_type="float",
    c_tile=128,
    patch_h=33,
    patch_w=33,
    out_c=128,
    tile_h=14,
    tile_w=14,
    kernel_h=3,
    kernel_w=3,
    stride_h=2,
    stride_w=2,
):
    data_type_tag = replace_data_type(data_type)
    func_name = (
        f"maxpool_tile_{c_tile}_{patch_h}_{patch_w}_{out_c}_{tile_h}_{tile_w}"
        f"_{kernel_h}_{kernel_w}_{stride_h}_{stride_w}_{data_type_tag}"
    )
    code_lines = [
        f"void {func_name}(",
        f"    data_t input[{c_tile}][{patch_h}][{patch_w}],",
        f"    data_t output[{out_c}][{tile_h}][{tile_w}],",
        "    int valid_c,",
        "    int valid_h,",
        "    int valid_w",
        ")",
        "{",
        "    for (int c = 0; c < valid_c; ++c) {",
        "        for (int oh = 0; oh < valid_h; ++oh) {",
        "            for (int ow = 0; ow < valid_w; ++ow) {",
        f"                data_t max_val = input[c][oh * {stride_h}][ow * {stride_w}];",
        f"                for (int kh = 0; kh < {kernel_h}; ++kh) {{",
        f"                    for (int kw = 0; kw < {kernel_w}; ++kw) {{",
        f"                        data_t cand = input[c][oh * {stride_h} + kh][ow * {stride_w} + kw];",
        "                        if (cand > max_val) {",
        "                            max_val = cand;",
        "                        }",
        "                    }",
        "                }",
        "                output[c][oh][ow] = max_val;",
        "            }",
        "        }",
        "    }",
        "}",
    ]
    return "\n".join(code_lines), func_name


def generate_adaptive_avgpool_tile_function(data_type="float", c_tile=128, h_in=7, w_in=7):
    data_type_tag = replace_data_type(data_type)
    func_name = f"adaptive_avgpool_tile_{c_tile}_{h_in}_{w_in}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t input[{c_tile}][{h_in}][{w_in}],",
        f"    data_t output[{c_tile}][1][1],",
        "    int valid_c",
        ")",
        "{",
        "    for (int c = 0; c < valid_c; ++c) {",
        "        data_t sum = (data_t)0;",
        f"        for (int h = 0; h < {h_in}; ++h) {{",
        f"            for (int w = 0; w < {w_in}; ++w) {{",
        "                sum += input[c][h][w];",
        "            }",
        "        }",
        f"        output[c][0][0] = sum / (data_t)({h_in} * {w_in});",
        "    }",
        "}",
    ]
    return "\n".join(code_lines), func_name


def generate_avgpool_accumulate_tile_function(data_type="float", c_tile=128, tile_h=7, tile_w=7):
    data_type_tag = replace_data_type(data_type)
    func_name = f"avgpool_accumulate_tile_{c_tile}_{tile_h}_{tile_w}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t input[{c_tile}][{tile_h}][{tile_w}],",
        f"    data_t output[{c_tile}][1][1],",
        "    int valid_c,",
        "    int valid_h,",
        "    int valid_w",
        ")",
        "{",
        "    for (int c = 0; c < valid_c; ++c) {",
        "        for (int h = 0; h < valid_h; ++h) {",
        "            for (int w = 0; w < valid_w; ++w) {",
        "                output[c][0][0] += input[c][h][w];",
        "            }",
        "        }",
        "    }",
        "}",
    ]
    return "\n".join(code_lines), func_name


def generate_avgpool_finalize_tile_function(data_type="float", c_tile=128, total_area=49):
    data_type_tag = replace_data_type(data_type)
    func_name = f"avgpool_finalize_tile_{c_tile}_{total_area}_{data_type_tag}"
    code_lines = [
        f"void {func_name}(",
        f"    data_t input[{c_tile}][1][1],",
        f"    data_t output[{c_tile}][1][1],",
        "    int valid_c",
        ")",
        "{",
        "    for (int c = 0; c < valid_c; ++c) {",
        f"        output[c][0][0] = input[c][0][0] / (data_t){total_area};",
        "    }",
        "}",
    ]
    return "\n".join(code_lines), func_name


def generate_layer_norm_code(
    template_path,    # Path to layer_norm_template.cpp
    DATA_TYPE="float",
    SEQ_LENGTH=128,   # e.g., sequence length
    DIM=512,          # e.g., feature dimension
    EPSILON=1e-5      # e.g., epsilon value
):
    """
    Reads the layer norm template file and substitutes placeholders with the provided parameters.
    The expected placeholders in the template are:
       {DATA_TYPE}, {SEQ_LENGTH}, {DIM}, and {EPSILON}.
       
    Generates the final HLS C code and writes it to output_path.
    """
    # 1) Read the template file.
    with open(template_path, "r") as f:
        template_code = f.read()
    
    # 2) Substitute placeholders.
    generated_code = template_code.format(
        DATA_TYPE=DATA_TYPE,
        SEQ_LENGTH=SEQ_LENGTH,
        DIM=DIM,
        EPSILON=EPSILON
    )
    
    DATA_TYPE = replace_data_type(DATA_TYPE)
    dim_suffix = f"_{SEQ_LENGTH}_{DIM}_{DATA_TYPE}"
    func_name = "layer_norm" + dim_suffix
    # Use regex to capture the function signature of maxpool.
    # We assume the template defines the function starting with "void maxpool(".
    new_generated_code = re.sub(
        r"(void\s+layer_norm)\s*\(",
        lambda m: m.group(1) + dim_suffix + "(",
        generated_code,
        count=1
    )
    
    return new_generated_code, func_name
    
    
def generate_rms_norm_code(
    template_path,    # Path to rms_norm_template.cpp
    DATA_TYPE="float",
    SEQ_LENGTH=128,   # e.g., sequence length (number of rows)
    DIM=512,          # e.g., feature dimension (number of columns)
    EPSILON=1e-5      # small constant to avoid division by zero
):
    """
    Reads the RMSNorm template file and substitutes placeholders with the provided parameters.
    Expected placeholders in the template:
      {DATA_TYPE}, {SEQ_LENGTH}, {DIM}, {EPSILON}.
      
    Generates the final HLS C code and writes it to output_path.
    """
    # 1) Read the template file.
    with open(template_path, "r") as f:
        template_code = f.read()
    
    # 2) Substitute placeholders.
    generated_code = template_code.format(
        DATA_TYPE=DATA_TYPE,
        SEQ_LENGTH=SEQ_LENGTH,
        DIM=DIM,
        EPSILON=EPSILON
    )
    
    DATA_TYPE = replace_data_type(DATA_TYPE)
    dim_suffix = f"_{SEQ_LENGTH}_{DIM}_{DATA_TYPE}"
    func_name = "rms_norm" + dim_suffix
    # Use regex to capture the function signature of maxpool.
    # We assume the template defines the function starting with "void maxpool(".
    new_generated_code = re.sub(
        r"(void\s+rms_norm)\s*\(",
        lambda m: m.group(1) + dim_suffix + "(",
        generated_code,
        count=1
    )
    
    return new_generated_code, func_name


def generate_matmul_code(
    template_path,   # Path to matmul_template.cpp
    DATA_TYPE="float",
    SEQ_LENGTH=128,
    DIM_IN=512,
    DIM_OUT=256,
    use_bias=False
):
    """
    Reads the matrix multiplication template and substitutes placeholders.
    The template expects the following placeholders:
       {DATA_TYPE}, {SEQ_LENGTH}, {DIM_IN}, {DIM_OUT}, {BIAS_ARG}, and {INIT_VAL}.
    
    If use_bias is True:
       - {BIAS_ARG} becomes: "    data_t bias[{DIM_OUT}],\n"
       - {INIT_VAL} becomes: "bias[j]"
    Otherwise:
       - {BIAS_ARG} becomes an empty string.
       - {INIT_VAL} becomes: "((data_t)0)"
    """
    # Read the template file.
    with open(template_path, "r") as f:
        template_code = f.read()
    
    # Build bias-related placeholders.
    if use_bias:
        bias_arg = "    data_t bias[{DIM_OUT}],\n".format(DIM_OUT=DIM_OUT)
        init_val = "bias[j]"
    else:
        bias_arg = ""
        init_val = "((data_t)0)"
    
    # Substitute placeholders in the template.
    generated_code = template_code.format(
        DATA_TYPE=DATA_TYPE,
        SEQ_LENGTH=SEQ_LENGTH,
        DIM_IN=DIM_IN,
        DIM_OUT=DIM_OUT,
        BIAS_ARG=bias_arg,
        INIT_VAL=init_val
    )
    
    DATA_TYPE = replace_data_type(DATA_TYPE)
    if use_bias == True:
        dim_suffix = f"_{SEQ_LENGTH}_{DIM_IN}_{DIM_OUT}_bias_{DATA_TYPE}"
    else:
        dim_suffix = f"_{SEQ_LENGTH}_{DIM_IN}_{DIM_OUT}_{DATA_TYPE}"
    func_name = "matmul" + dim_suffix
    # Use regex to capture the function signature of maxpool.
    # We assume the template defines the function starting with "void maxpool(".
    new_generated_code = re.sub(
        r"(void\s+matmul)\s*\(",
        lambda m: m.group(1) + dim_suffix + "(",
        generated_code,
        count=1
    )
    
    return new_generated_code, func_name
    

def generate_dropout_code(
    template_path,    # Path to dropout_template.cpp
    DATA_TYPE="float",
    SEQ_LENGTH=128,
    DIM=512
):
    """
    Reads the dropout template and substitutes placeholders with provided parameters.
    
    Expected placeholders in the template:
      {DATA_TYPE}, {SEQ_LENGTH}, {DIM}
      
    Generates the final HLS C code for the dropout function and writes it to output_path.
    """
    # 1) Read the template file.
    with open(template_path, "r") as f:
        template_code = f.read()
    
    # 2) Substitute placeholders.
    generated_code = template_code.format(
        DATA_TYPE=DATA_TYPE,
        SEQ_LENGTH=SEQ_LENGTH,
        DIM=DIM
    )
    
    DATA_TYPE = replace_data_type(DATA_TYPE)
    dim_suffix = f"_{SEQ_LENGTH}_{DIM}_{DATA_TYPE}"
    func_name = "dropout" + dim_suffix
    # Use regex to capture the function signature of maxpool.
    # We assume the template defines the function starting with "void maxpool(".
    new_generated_code = re.sub(
        r"(void\s+dropout)\s*\(",
        lambda m: m.group(1) + dim_suffix + "(",
        generated_code,
        count=1
    )
    
    return new_generated_code, func_name
    
    
def generate_grouped_mha_code(
    template_path,
    DATA_TYPE="float",
    SEQ_LENGTH=128,
    DIM_IN=512,
    NUM_HEADS=8,
    HEAD_DIM=64,
    use_rope=True
):
    DIM_OUT = NUM_HEADS * HEAD_DIM

    if use_rope:
        rope_inline = f"""
    // Inline RoPE logic
    for (int seq = 0; seq < {SEQ_LENGTH}; seq++) {{
        for (int h = 0; h < {NUM_HEADS}; h++) {{
            for (int d = 0; d < {HEAD_DIM}; d += 2) {{
                int idx = h * {HEAD_DIM} + d;
                data_t theta = (data_t)hls::powf(10000.0f, -((float)d) / (float){HEAD_DIM});
                data_t angle = seq * theta;
                data_t cos_val = hls::cos(angle);
                data_t sin_val = hls::sin(angle);

                // Apply RoPE to Q
                data_t q0 = Q[seq][idx];
                data_t q1 = Q[seq][idx + 1];
                Q[seq][idx]     = q0 * cos_val - q1 * sin_val;
                Q[seq][idx + 1] = q0 * sin_val + q1 * cos_val;

                // Apply RoPE to K
                data_t k0 = K[seq][idx];
                data_t k1 = K[seq][idx + 1];
                K[seq][idx]     = k0 * cos_val - k1 * sin_val;
                K[seq][idx + 1] = k0 * sin_val + k1 * cos_val;
            }}
        }}
    }}"""
    else:
        rope_inline = ''

    # Read template
    with open(template_path, "r") as f:
        template_code = f.read()

    generated_code = template_code.format(
        DATA_TYPE=DATA_TYPE,
        SEQ_LENGTH=SEQ_LENGTH,
        DIM_IN=DIM_IN,
        DIM_OUT=DIM_OUT,
        NUM_HEADS=NUM_HEADS,
        HEAD_DIM=HEAD_DIM,
        ROPE_INLINE=rope_inline
    )

    DATA_TYPE = replace_data_type(DATA_TYPE)
    if use_rope == True:
        dim_suffix = f"_{SEQ_LENGTH}_{DIM_IN}_{NUM_HEADS}_{HEAD_DIM}_rope_{DATA_TYPE}"
    else:
        dim_suffix = f"_{SEQ_LENGTH}_{DIM_IN}_{NUM_HEADS}_{HEAD_DIM}_{DATA_TYPE}"
    func_name = "grouped_multihead_attention" + dim_suffix
    # Use regex to capture the function signature of maxpool.
    # We assume the template defines the function starting with "void maxpool(".
    new_generated_code = re.sub(
        r"(void\s+grouped_multihead_attention)\s*\(",
        lambda m: m.group(1) + dim_suffix + "(",
        generated_code,
        count=1
    )
    
    return new_generated_code, func_name


def generate_sliding_window_attention_code(
    template_path,
    DATA_TYPE="float",
    SEQ_LENGTH=128,
    DIM_IN=512,
    NUM_HEADS=8,
    HEAD_DIM=64,
    use_rope=True
):
    DIM_OUT = NUM_HEADS * HEAD_DIM

    if use_rope:
        rope_inline = f"""
    // Inline RoPE logic
    for (int seq = 0; seq < {SEQ_LENGTH}; seq++) {{
        for (int h = 0; h < {NUM_HEADS}; h++) {{
            for (int d = 0; d < {HEAD_DIM}; d += 2) {{
                int idx = h * {HEAD_DIM} + d;
                data_t theta = (data_t)hls::powf(10000.0f, -((float)d) / (float){HEAD_DIM});
                data_t angle = seq * theta;
                data_t cos_val = hls::cos(angle);
                data_t sin_val = hls::sin(angle);

                // Apply RoPE to Q
                data_t q0 = Q[seq][idx];
                data_t q1 = Q[seq][idx + 1];
                Q[seq][idx]     = q0 * cos_val - q1 * sin_val;
                Q[seq][idx + 1] = q0 * sin_val + q1 * cos_val;

                // Apply RoPE to K
                data_t k0 = K[seq][idx];
                data_t k1 = K[seq][idx + 1];
                K[seq][idx]     = k0 * cos_val - k1 * sin_val;
                K[seq][idx + 1] = k0 * sin_val + k1 * cos_val;
            }}
        }}
    }}"""
    else:
        rope_inline = ''

    # Read template
    with open(template_path, "r") as f:
        template_code = f.read()

    generated_code = template_code.format(
        DATA_TYPE=DATA_TYPE,
        SEQ_LENGTH=SEQ_LENGTH,
        DIM_IN=DIM_IN,
        DIM_OUT=DIM_OUT,
        NUM_HEADS=NUM_HEADS,
        HEAD_DIM=HEAD_DIM,
        ROPE_INLINE=rope_inline
    )

    DATA_TYPE = replace_data_type(DATA_TYPE)
    if use_rope == True:
        dim_suffix = f"_{SEQ_LENGTH}_{DIM_IN}_{NUM_HEADS}_{HEAD_DIM}_rope_{DATA_TYPE}"
    else:
        dim_suffix = f"_{SEQ_LENGTH}_{DIM_IN}_{NUM_HEADS}_{HEAD_DIM}_{DATA_TYPE}"
    func_name = "sliding_window_attention" + dim_suffix
    # Use regex to capture the function signature of maxpool.
    # We assume the template defines the function starting with "void maxpool(".
    new_generated_code = re.sub(
        r"(void\s+sliding_window_attention)\s*\(",
        lambda m: m.group(1) + dim_suffix + "(",
        generated_code,
        count=1
    )
    
    return new_generated_code, func_name


def generate_matrix_add_code(
    template_path,    # Path to matrix_add_template.cpp
    DATA_TYPE="float",
    SEQ_LENGTH=64,
    DIM=64
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
        SEQ_LENGTH = SEQ_LENGTH,
        DIM = DIM
    )
    
    DATA_TYPE = replace_data_type(DATA_TYPE)
    dim_suffix = f"_{SEQ_LENGTH}_{DIM}_{DATA_TYPE}"
    
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
    
def generate_elementwise_mult_code(
    template_path,    # Path to matrix_add_template.cpp
    DATA_TYPE="float",
    SEQ_LENGTH=64,
    DIM=64
):
    """
    Reads the matrix addition template and substitutes the placeholders with the provided parameters.
    The template uses the following placeholders:
       {DATA_TYPE}, {C}, {H}, {W}
    
    The final function has the following signature:
       void elementwise_mult(data_t in1[C][H][W], data_t in2[C][H][W], data_t out[C][H][W])
    
    Parameters:
       template_path: path to the elementwise_mult_template.cpp file.
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
        SEQ_LENGTH = SEQ_LENGTH,
        DIM = DIM
    )
    
    DATA_TYPE = replace_data_type(DATA_TYPE)
    dim_suffix = f"_{SEQ_LENGTH}_{DIM}_{DATA_TYPE}"
    
    # Use regex to capture the function signature of maxpool.
    # We assume the template defines the function starting with "void maxpool(".
    new_generated_code = re.sub(
        r"(void\s+elementwise_mult)\s*\(",
        lambda m: m.group(1) + dim_suffix + "(",
        generated_code,
        count=1
    )
    
    func_name = "elementwise_mult" + dim_suffix
    
    return new_generated_code, func_name
    
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

def generate_maxpool_pad_code(
    template_path,    # Path to maxpool_pad_template.cpp
    DATA_TYPE="float",
    C=16,
    H_IN=64,
    W_IN=64,
    H_OUT=32,
    W_OUT=32,
    K_H=3,
    K_W=3,
    STRIDE_H=2,
    STRIDE_W=2,
    PAD_H=1,
    PAD_W=1
):
    """
    Reads the padded maxpool template and substitutes placeholders for a maxpool
    implementation with explicit padding support.
    """
    with open(template_path, "r") as f:
        template_code = f.read()

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
        STRIDE_W=STRIDE_W,
        PAD_H=PAD_H,
        PAD_W=PAD_W
    )

    DATA_TYPE = replace_data_type(DATA_TYPE)
    dim_suffix = (
        f"_{C}_{H_IN}_{W_IN}_{H_OUT}_{W_OUT}_{K_H}_{K_W}_{STRIDE_H}_{STRIDE_W}"
        f"_{PAD_H}_{PAD_W}_{DATA_TYPE}"
    )

    new_generated_code = re.sub(
        r"(void\s+maxpool_pad)\s*\(",
        lambda m: m.group(1) + dim_suffix + "(",
        generated_code,
        count=1
    )

    func_name = "maxpool_pad" + dim_suffix

    return new_generated_code, func_name

def generate_matrix_add_conv_code(
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


def generate_func_def(op_info, data_type):
    if op_info['func_name'] in {'loop_begin', 'loop_end'}:
        return "", ""
    
    if op_info['func_name'] == 'load':
        code_line, full_func_name = generate_load_function(op_info["dims"], data_type, func_prefix="load")
    elif op_info['func_name'] == 'store':
        code_line, full_func_name = generate_store_function(op_info["dims"], data_type, func_prefix="store")
    elif op_info['func_name'] == 'clear_tile':
        code_line, full_func_name = generate_clear_tile_function(op_info["dims"], data_type)
    elif op_info['func_name'] == 'clear_matrix_tile':
        clear_dims = normalize_clear_matrix_tile_dims(op_info["dims"])
        code_line, full_func_name = generate_clear_matrix_tile_function(clear_dims[0], clear_dims[1], data_type, col_factor=clear_dims[2])
    elif op_info['func_name'] == 'clear_vector_tile':
        code_line, full_func_name = generate_clear_vector_tile_function(op_info["dims"][0], data_type)
    elif op_info['func_name'] == 'init_rowmax_tile':
        init_rowmax_dims = normalize_init_rowmax_tile_dims(op_info["dims"])
        code_line, full_func_name = generate_init_rowmax_tile_function(init_rowmax_dims[0], init_rowmax_dims[1], data_type, col_factor=init_rowmax_dims[2])
    elif op_info['func_name'] == 'load_matrix_tile':
        matrix_tile_dims = normalize_matrix_tile_dims(op_info["dims"])
        code_line, full_func_name = generate_load_matrix_tile_function(data_type, matrix_tile_dims[0], matrix_tile_dims[1], matrix_tile_dims[2], matrix_tile_dims[3], matrix_tile_dims[4])
    elif op_info['func_name'] == 'store_matrix_tile':
        matrix_tile_dims = normalize_matrix_tile_dims(op_info["dims"])
        code_line, full_func_name = generate_store_matrix_tile_function(data_type, matrix_tile_dims[0], matrix_tile_dims[1], matrix_tile_dims[2], matrix_tile_dims[3], matrix_tile_dims[4])
    elif op_info['func_name'] == 'load_vector_tile':
        vector_tile_dims = normalize_vector_tile_dims(op_info["dims"])
        code_line, full_func_name = generate_load_vector_tile_function(data_type, vector_tile_dims[0], vector_tile_dims[1], vector_tile_dims[2])
    elif op_info['func_name'] == 'load_layer_vector':
        code_line, full_func_name = generate_load_layer_vector_function(data_type, op_info["dims"][0], op_info["dims"][1])
    elif op_info['func_name'] == 'load_layer_vector_tile':
        layer_vector_tile_dims = normalize_layer_vector_tile_dims(op_info["dims"])
        code_line, full_func_name = generate_load_layer_vector_tile_function(data_type, layer_vector_tile_dims[0], layer_vector_tile_dims[1], layer_vector_tile_dims[2], layer_vector_tile_dims[3])
    elif op_info['func_name'] == 'load_weight_tile_layered':
        weight_tile_layered_dims = normalize_weight_tile_layered_dims(op_info["dims"])
        code_line, full_func_name = generate_load_weight_tile_layered_function(data_type, weight_tile_layered_dims[0], weight_tile_layered_dims[1], weight_tile_layered_dims[2], weight_tile_layered_dims[3], weight_tile_layered_dims[4], weight_tile_layered_dims[5], weight_tile_layered_dims[6])
    elif op_info['func_name'] == 'load_weight_tile_2d':
        weight_tile_2d_dims = normalize_weight_tile_2d_dims(op_info["dims"])
        code_line, full_func_name = generate_load_weight_tile_2d_function(data_type, weight_tile_2d_dims[0], weight_tile_2d_dims[1], weight_tile_2d_dims[2], weight_tile_2d_dims[3], weight_tile_2d_dims[4], weight_tile_2d_dims[5])
    elif op_info['func_name'] == 'linear_tile':
        linear_tile_dims = normalize_linear_tile_dims(op_info["dims"])
        code_line, full_func_name = generate_linear_tile_function(data_type, linear_tile_dims[0], linear_tile_dims[1], linear_tile_dims[2], linear_tile_dims[3], linear_tile_dims[4])
    elif op_info['func_name'] == 'rmsnorm_tile_full':
        code_line, full_func_name = generate_rmsnorm_tile_full_function(data_type, op_info["dims"][0], op_info["dims"][1])
    elif op_info['func_name'] == 'rmsnorm_accumulate_tile':
        rmsnorm_tile_dims = normalize_rmsnorm_tile_dims(op_info["dims"])
        code_line, full_func_name = generate_rmsnorm_accumulate_tile_function(data_type, rmsnorm_tile_dims[0], rmsnorm_tile_dims[1], rmsnorm_tile_dims[2])
    elif op_info['func_name'] == 'rmsnorm_finalize_rows':
        code_line, full_func_name = generate_rmsnorm_finalize_rows_function(data_type, op_info["dims"][0], op_info["dims"][1])
    elif op_info['func_name'] == 'rmsnorm_apply_tile':
        rmsnorm_tile_dims = normalize_rmsnorm_tile_dims(op_info["dims"])
        code_line, full_func_name = generate_rmsnorm_apply_tile_function(data_type, rmsnorm_tile_dims[0], rmsnorm_tile_dims[1], rmsnorm_tile_dims[2])
    elif op_info['func_name'] == 'matrix_add_tile_2d':
        matrix_2d_dims = normalize_matrix_2d_op_dims(op_info["dims"])
        code_line, full_func_name = generate_matrix_add_tile_2d_function(data_type, matrix_2d_dims[0], matrix_2d_dims[1], matrix_2d_dims[2])
    elif op_info['func_name'] == 'activation_tile_2d':
        matrix_2d_dims = normalize_matrix_2d_op_dims(op_info["dims"])
        code_line, full_func_name = generate_activation_tile_2d_function(data_type, matrix_2d_dims[0], matrix_2d_dims[1], op_info["func_info"][1], matrix_2d_dims[2])
    elif op_info['func_name'] == 'elementwise_mult_tile_2d':
        matrix_2d_dims = normalize_matrix_2d_op_dims(op_info["dims"])
        code_line, full_func_name = generate_elementwise_mult_tile_2d_function(data_type, matrix_2d_dims[0], matrix_2d_dims[1], matrix_2d_dims[2])
    elif op_info['func_name'] == 'embedding_lookup_chunk':
        embedding_lookup_dims = normalize_embedding_lookup_chunk_dims(op_info["dims"])
        code_line, full_func_name = generate_embedding_lookup_chunk_function(data_type, embedding_lookup_dims[0], embedding_lookup_dims[1], embedding_lookup_dims[2], embedding_lookup_dims[3], embedding_lookup_dims[4], embedding_lookup_dims[5])
    elif op_info['func_name'] == 'embedding_lookup_tile':
        code_line, full_func_name = generate_embedding_lookup_tile_function(data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3])
    elif op_info['func_name'] == 'kv_cache_store_tile':
        code_line, full_func_name = generate_kv_cache_store_tile_function(data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3], op_info["dims"][4], op_info["dims"][5])
    elif op_info['func_name'] == 'kv_cache_load_tile':
        code_line, full_func_name = generate_kv_cache_load_tile_function(data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3])
    elif op_info['func_name'] == 'kv_cache_load_group_tile':
        kv_group_dims = normalize_kv_cache_load_group_tile_dims(op_info["dims"])
        code_line, full_func_name = generate_kv_cache_load_group_tile_function(
            data_type,
            kv_group_dims[0],
            kv_group_dims[1],
            kv_group_dims[2],
            kv_group_dims[3],
            kv_group_dims[4],
            kv_group_dims[5],
        )
    elif op_info['func_name'] == 'apply_rope_tile':
        rope_dims = normalize_rope_tile_dims(op_info["dims"])
        code_line, full_func_name = generate_apply_rope_tile_function(data_type, rope_dims[0], rope_dims[1], rope_dims[2], rope_dims[3], rope_dims[4])
    elif op_info['func_name'] == 'attention_score_tile':
        attention_dims = normalize_attention_tile_dims(op_info["dims"])
        code_line, full_func_name = generate_attention_score_tile_function(
            data_type,
            attention_dims[0],
            attention_dims[1],
            attention_dims[2],
            attention_dims[3],
            attention_dims[4],
            attention_dims[5],
            attention_dims[6],
        )
    elif op_info['func_name'] == 'attention_rowmax_tile':
        rowmax_dims = normalize_attention_rowmax_dims(op_info["dims"])
        code_line, full_func_name = generate_attention_rowmax_tile_function(data_type, rowmax_dims[0], rowmax_dims[1], rowmax_dims[2], rowmax_dims[3])
    elif op_info['func_name'] == 'attention_softmax_context_tile':
        attention_dims = normalize_attention_tile_dims(op_info["dims"])
        code_line, full_func_name = generate_attention_softmax_context_tile_function(
            data_type,
            attention_dims[0],
            attention_dims[1],
            attention_dims[2],
            attention_dims[3],
            attention_dims[4],
            attention_dims[5],
            attention_dims[6],
        )
    elif op_info['func_name'] == 'attention_finalize_tile':
        finalize_dims = normalize_attention_finalize_dims(op_info["dims"])
        code_line, full_func_name = generate_attention_finalize_tile_function(data_type, finalize_dims[0], finalize_dims[1], finalize_dims[2], finalize_dims[3], finalize_dims[4])
    elif op_info['func_name'] == 'load_fmap_patch':
        code_line, full_func_name = generate_load_fmap_patch_function(data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3], op_info["dims"][4], op_info["dims"][5], op_info["dims"][6], op_info["dims"][7])
    elif op_info['func_name'] == 'load_fmap_tile':
        code_line, full_func_name = generate_load_fmap_tile_function(data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3], op_info["dims"][4], op_info["dims"][5])
    elif op_info['func_name'] == 'store_fmap_tile':
        code_line, full_func_name = generate_store_fmap_tile_function(data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3], op_info["dims"][4], op_info["dims"][5])
    elif op_info['func_name'] == 'load_weight_tile':
        code_line, full_func_name = generate_load_weight_tile_function(data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3], op_info["dims"][4])
    elif op_info['func_name'] == 'load_bn_tile':
        code_line, full_func_name = generate_load_bn_tile_function(data_type, op_info["dims"][0], op_info["dims"][1])
    elif op_info['func_name'] == 'conv_tile':
        conv_tile_dims = normalize_conv_tile_dims(op_info["dims"])
        code_line, full_func_name = generate_conv_tile_function(
            data_type,
            conv_tile_dims[0],
            conv_tile_dims[1],
            conv_tile_dims[2],
            conv_tile_dims[3],
            conv_tile_dims[4],
            conv_tile_dims[5],
            conv_tile_dims[6],
            conv_tile_dims[7],
            conv_tile_dims[8],
            conv_tile_dims[9],
            conv_tile_dims[10],
            conv_tile_dims[11],
        )
    elif op_info['func_name'] == 'batchnorm_tile':
        code_line, full_func_name = generate_batchnorm_tile_function(data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2])
    elif op_info['func_name'] == 'activation_tile':
        code_line, full_func_name = generate_activation_tile_function(data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["func_info"][1])
    elif op_info['func_name'] == 'matrix_add_tile':
        code_line, full_func_name = generate_matrix_add_tile_function(data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2])
    elif op_info['func_name'] == 'maxpool_tile':
        code_line, full_func_name = generate_maxpool_tile_function(data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3], op_info["dims"][4], op_info["dims"][5], op_info["dims"][6], op_info["dims"][7], op_info["dims"][8], op_info["dims"][9])
    elif op_info['func_name'] == 'adaptive_avgpool_tile':
        code_line, full_func_name = generate_adaptive_avgpool_tile_function(data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2])
    elif op_info['func_name'] == 'avgpool_accumulate_tile':
        code_line, full_func_name = generate_avgpool_accumulate_tile_function(data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2])
    elif op_info['func_name'] == 'avgpool_finalize_tile':
        code_line, full_func_name = generate_avgpool_finalize_tile_function(data_type, op_info["dims"][0], op_info["dims"][1])
    elif op_info['func_name'] == 'matmul':
        code_line, full_func_name = generate_matmul_code(op_info["func_info"][0], data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["func_info"][1])
    elif op_info['func_name'] == 'mha':
        code_line, full_func_name = generate_grouped_mha_code(op_info["func_info"][0], data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3], op_info["func_info"][1])
    elif op_info['func_name'] == 'swa':
        code_line, full_func_name = generate_sliding_window_attention_code(op_info["func_info"][0], data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3], op_info["func_info"][1])
    elif op_info['func_name'] == 'layernorm':
        code_line, full_func_name = generate_layer_norm_code(op_info["func_info"][0], data_type,  op_info["dims"][0], op_info["dims"][1], op_info["dims"][2])
    elif op_info['func_name'] == 'rmsnorm':
        code_line, full_func_name = generate_rms_norm_code(op_info["func_info"][0], data_type,  op_info["dims"][0], op_info["dims"][1], op_info["dims"][2])
    elif op_info['func_name'] == 'activation':
        code_line, full_func_name = generate_activation_function(op_info["func_info"][0], op_info["func_info"][1], data_type,  op_info["dims"][0], op_info["dims"][1])
    elif op_info['func_name'] == 'activation_conv':
        code_line, full_func_name = generate_activation_function_conv(op_info["func_info"][0], op_info["func_info"][1], data_type,  op_info["dims"][0], op_info["dims"][1], op_info["dims"][2])
    elif op_info['func_name'] == 'dropout':
        code_line, full_func_name = generate_dropout_code(op_info["func_info"][0], data_type, op_info["dims"][0], op_info["dims"][1])
    elif op_info['func_name'] == 'matrix_add':
        code_line, full_func_name = generate_matrix_add_code(op_info["func_info"][0], data_type, op_info["dims"][0], op_info["dims"][1])
    elif op_info['func_name'] == 'matrix_add_conv':
        code_line, full_func_name = generate_matrix_add_conv_code(op_info["func_info"][0], data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2])
    elif op_info['func_name'] == 'elementwise_mult':
        code_line, full_func_name = generate_elementwise_mult_code(op_info["func_info"][0], data_type, op_info["dims"][0], op_info["dims"][1]) 
    elif op_info['func_name'] == 'conv':
        conv_dims = normalize_conv_dims(op_info["dims"])
        code_line, full_func_name = generate_conv_function(
            op_info["func_info"][0],
            op_info["func_info"][1],
            data_type,
            conv_dims[0],
            conv_dims[1],
            conv_dims[2],
            conv_dims[3],
            conv_dims[4],
            conv_dims[5],
            conv_dims[6],
            conv_dims[7],
            conv_dims[8],
            op_info["func_info"][2],
            conv_dims[9],
            conv_dims[10],
            conv_dims[11],
            conv_dims[12],
            conv_dims[13],
            conv_dims[14],
            conv_dims[15],
        )
    elif op_info['func_name'] == 'conv_tiled':
        code_line, full_func_name = generate_conv_tiled_function(op_info["func_info"][0], op_info["func_info"][1], data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3], op_info["dims"][4], op_info["dims"][5], op_info["dims"][6], op_info["dims"][7], op_info["dims"][8], op_info["dims"][9], op_info["dims"][10], op_info["dims"][11], op_info["dims"][12], op_info["dims"][13], op_info["dims"][14])
    elif op_info['func_name'] == 'batchnorm':
        code_line, full_func_name = generate_batch_norm_code(op_info["func_info"][0], data_type,  op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["func_info"][1])
    elif op_info['func_name'] == 'maxpool':
        code_line, full_func_name = generate_maxpool_code(op_info["func_info"][0], data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3], op_info["dims"][4], op_info["dims"][5], op_info["dims"][6], op_info["dims"][7], op_info["dims"][8])
    elif op_info['func_name'] == 'maxpool_pad':
        code_line, full_func_name = generate_maxpool_pad_code(op_info["func_info"][0], data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3], op_info["dims"][4], op_info["dims"][5], op_info["dims"][6], op_info["dims"][7], op_info["dims"][8], op_info["dims"][9], op_info["dims"][10])
    elif op_info['func_name'] == 'adaptive_avgpool':
        code_line, full_func_name = generate_adaptive_avgpool_code(op_info["func_info"][0], data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3], op_info["dims"][4])

    else:
        print(f"the {op_info['func_name']} operator we do not support!")
        
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

    if op_info['func_name'] in {'loop_begin', 'loop_end'}:
        return ""

    if op_info['func_name'] == 'load':
        code_line, full_func_name = generate_load_function(op_info["dims"], data_type, func_prefix="load")
    elif op_info['func_name'] == 'store':
        code_line, full_func_name = generate_store_function(op_info["dims"], data_type, func_prefix="store")
    elif op_info['func_name'] == 'clear_tile':
        code_line, full_func_name = generate_clear_tile_function(op_info["dims"], data_type)
    elif op_info['func_name'] == 'clear_matrix_tile':
        clear_dims = normalize_clear_matrix_tile_dims(op_info["dims"])
        code_line, full_func_name = generate_clear_matrix_tile_function(clear_dims[0], clear_dims[1], data_type, col_factor=clear_dims[2])
    elif op_info['func_name'] == 'clear_vector_tile':
        code_line, full_func_name = generate_clear_vector_tile_function(op_info["dims"][0], data_type)
    elif op_info['func_name'] == 'init_rowmax_tile':
        init_rowmax_dims = normalize_init_rowmax_tile_dims(op_info["dims"])
        code_line, full_func_name = generate_init_rowmax_tile_function(init_rowmax_dims[0], init_rowmax_dims[1], data_type, col_factor=init_rowmax_dims[2])
    elif op_info['func_name'] == 'load_matrix_tile':
        matrix_tile_dims = normalize_matrix_tile_dims(op_info["dims"])
        code_line, full_func_name = generate_load_matrix_tile_function(data_type, matrix_tile_dims[0], matrix_tile_dims[1], matrix_tile_dims[2], matrix_tile_dims[3], matrix_tile_dims[4])
    elif op_info['func_name'] == 'store_matrix_tile':
        matrix_tile_dims = normalize_matrix_tile_dims(op_info["dims"])
        code_line, full_func_name = generate_store_matrix_tile_function(data_type, matrix_tile_dims[0], matrix_tile_dims[1], matrix_tile_dims[2], matrix_tile_dims[3], matrix_tile_dims[4])
    elif op_info['func_name'] == 'load_vector_tile':
        vector_tile_dims = normalize_vector_tile_dims(op_info["dims"])
        code_line, full_func_name = generate_load_vector_tile_function(data_type, vector_tile_dims[0], vector_tile_dims[1], vector_tile_dims[2])
    elif op_info['func_name'] == 'load_layer_vector':
        code_line, full_func_name = generate_load_layer_vector_function(data_type, op_info["dims"][0], op_info["dims"][1])
    elif op_info['func_name'] == 'load_layer_vector_tile':
        layer_vector_tile_dims = normalize_layer_vector_tile_dims(op_info["dims"])
        code_line, full_func_name = generate_load_layer_vector_tile_function(data_type, layer_vector_tile_dims[0], layer_vector_tile_dims[1], layer_vector_tile_dims[2], layer_vector_tile_dims[3])
    elif op_info['func_name'] == 'load_weight_tile_layered':
        weight_tile_layered_dims = normalize_weight_tile_layered_dims(op_info["dims"])
        code_line, full_func_name = generate_load_weight_tile_layered_function(data_type, weight_tile_layered_dims[0], weight_tile_layered_dims[1], weight_tile_layered_dims[2], weight_tile_layered_dims[3], weight_tile_layered_dims[4], weight_tile_layered_dims[5], weight_tile_layered_dims[6])
    elif op_info['func_name'] == 'load_weight_tile_2d':
        weight_tile_2d_dims = normalize_weight_tile_2d_dims(op_info["dims"])
        code_line, full_func_name = generate_load_weight_tile_2d_function(data_type, weight_tile_2d_dims[0], weight_tile_2d_dims[1], weight_tile_2d_dims[2], weight_tile_2d_dims[3], weight_tile_2d_dims[4], weight_tile_2d_dims[5])
    elif op_info['func_name'] == 'linear_tile':
        linear_tile_dims = normalize_linear_tile_dims(op_info["dims"])
        code_line, full_func_name = generate_linear_tile_function(data_type, linear_tile_dims[0], linear_tile_dims[1], linear_tile_dims[2], linear_tile_dims[3], linear_tile_dims[4])
    elif op_info['func_name'] == 'rmsnorm_tile_full':
        code_line, full_func_name = generate_rmsnorm_tile_full_function(data_type, op_info["dims"][0], op_info["dims"][1])
    elif op_info['func_name'] == 'rmsnorm_accumulate_tile':
        rmsnorm_tile_dims = normalize_rmsnorm_tile_dims(op_info["dims"])
        code_line, full_func_name = generate_rmsnorm_accumulate_tile_function(data_type, rmsnorm_tile_dims[0], rmsnorm_tile_dims[1], rmsnorm_tile_dims[2])
    elif op_info['func_name'] == 'rmsnorm_finalize_rows':
        code_line, full_func_name = generate_rmsnorm_finalize_rows_function(data_type, op_info["dims"][0], op_info["dims"][1])
    elif op_info['func_name'] == 'rmsnorm_apply_tile':
        rmsnorm_tile_dims = normalize_rmsnorm_tile_dims(op_info["dims"])
        code_line, full_func_name = generate_rmsnorm_apply_tile_function(data_type, rmsnorm_tile_dims[0], rmsnorm_tile_dims[1], rmsnorm_tile_dims[2])
    elif op_info['func_name'] == 'matrix_add_tile_2d':
        matrix_2d_dims = normalize_matrix_2d_op_dims(op_info["dims"])
        code_line, full_func_name = generate_matrix_add_tile_2d_function(data_type, matrix_2d_dims[0], matrix_2d_dims[1], matrix_2d_dims[2])
    elif op_info['func_name'] == 'activation_tile_2d':
        matrix_2d_dims = normalize_matrix_2d_op_dims(op_info["dims"])
        code_line, full_func_name = generate_activation_tile_2d_function(data_type, matrix_2d_dims[0], matrix_2d_dims[1], op_info["func_info"][1], matrix_2d_dims[2])
    elif op_info['func_name'] == 'elementwise_mult_tile_2d':
        matrix_2d_dims = normalize_matrix_2d_op_dims(op_info["dims"])
        code_line, full_func_name = generate_elementwise_mult_tile_2d_function(data_type, matrix_2d_dims[0], matrix_2d_dims[1], matrix_2d_dims[2])
    elif op_info['func_name'] == 'embedding_lookup_chunk':
        embedding_lookup_dims = normalize_embedding_lookup_chunk_dims(op_info["dims"])
        code_line, full_func_name = generate_embedding_lookup_chunk_function(data_type, embedding_lookup_dims[0], embedding_lookup_dims[1], embedding_lookup_dims[2], embedding_lookup_dims[3], embedding_lookup_dims[4], embedding_lookup_dims[5])
    elif op_info['func_name'] == 'embedding_lookup_tile':
        code_line, full_func_name = generate_embedding_lookup_tile_function(data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3])
    elif op_info['func_name'] == 'kv_cache_store_tile':
        code_line, full_func_name = generate_kv_cache_store_tile_function(data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3], op_info["dims"][4], op_info["dims"][5])
    elif op_info['func_name'] == 'kv_cache_load_tile':
        code_line, full_func_name = generate_kv_cache_load_tile_function(data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3])
    elif op_info['func_name'] == 'kv_cache_load_group_tile':
        kv_group_dims = normalize_kv_cache_load_group_tile_dims(op_info["dims"])
        code_line, full_func_name = generate_kv_cache_load_group_tile_function(
            data_type,
            kv_group_dims[0],
            kv_group_dims[1],
            kv_group_dims[2],
            kv_group_dims[3],
            kv_group_dims[4],
            kv_group_dims[5],
        )
    elif op_info['func_name'] == 'apply_rope_tile':
        rope_dims = normalize_rope_tile_dims(op_info["dims"])
        code_line, full_func_name = generate_apply_rope_tile_function(data_type, rope_dims[0], rope_dims[1], rope_dims[2], rope_dims[3], rope_dims[4])
    elif op_info['func_name'] == 'attention_score_tile':
        attention_dims = normalize_attention_tile_dims(op_info["dims"])
        code_line, full_func_name = generate_attention_score_tile_function(
            data_type,
            attention_dims[0],
            attention_dims[1],
            attention_dims[2],
            attention_dims[3],
            attention_dims[4],
            attention_dims[5],
            attention_dims[6],
        )
    elif op_info['func_name'] == 'attention_rowmax_tile':
        rowmax_dims = normalize_attention_rowmax_dims(op_info["dims"])
        code_line, full_func_name = generate_attention_rowmax_tile_function(data_type, rowmax_dims[0], rowmax_dims[1], rowmax_dims[2], rowmax_dims[3])
    elif op_info['func_name'] == 'attention_softmax_context_tile':
        attention_dims = normalize_attention_tile_dims(op_info["dims"])
        code_line, full_func_name = generate_attention_softmax_context_tile_function(
            data_type,
            attention_dims[0],
            attention_dims[1],
            attention_dims[2],
            attention_dims[3],
            attention_dims[4],
            attention_dims[5],
            attention_dims[6],
        )
    elif op_info['func_name'] == 'attention_finalize_tile':
        finalize_dims = normalize_attention_finalize_dims(op_info["dims"])
        code_line, full_func_name = generate_attention_finalize_tile_function(data_type, finalize_dims[0], finalize_dims[1], finalize_dims[2], finalize_dims[3], finalize_dims[4])
    elif op_info['func_name'] == 'load_fmap_patch':
        code_line, full_func_name = generate_load_fmap_patch_function(data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3], op_info["dims"][4], op_info["dims"][5], op_info["dims"][6], op_info["dims"][7])
    elif op_info['func_name'] == 'load_fmap_tile':
        code_line, full_func_name = generate_load_fmap_tile_function(data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3], op_info["dims"][4], op_info["dims"][5])
    elif op_info['func_name'] == 'store_fmap_tile':
        code_line, full_func_name = generate_store_fmap_tile_function(data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3], op_info["dims"][4], op_info["dims"][5])
    elif op_info['func_name'] == 'load_weight_tile':
        code_line, full_func_name = generate_load_weight_tile_function(data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3], op_info["dims"][4])
    elif op_info['func_name'] == 'load_bn_tile':
        code_line, full_func_name = generate_load_bn_tile_function(data_type, op_info["dims"][0], op_info["dims"][1])
    elif op_info['func_name'] == 'conv_tile':
        conv_tile_dims = normalize_conv_tile_dims(op_info["dims"])
        code_line, full_func_name = generate_conv_tile_function(
            data_type,
            conv_tile_dims[0],
            conv_tile_dims[1],
            conv_tile_dims[2],
            conv_tile_dims[3],
            conv_tile_dims[4],
            conv_tile_dims[5],
            conv_tile_dims[6],
            conv_tile_dims[7],
            conv_tile_dims[8],
            conv_tile_dims[9],
            conv_tile_dims[10],
            conv_tile_dims[11],
        )
    elif op_info['func_name'] == 'batchnorm_tile':
        code_line, full_func_name = generate_batchnorm_tile_function(data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2])
    elif op_info['func_name'] == 'activation_tile':
        code_line, full_func_name = generate_activation_tile_function(data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["func_info"][1])
    elif op_info['func_name'] == 'matrix_add_tile':
        code_line, full_func_name = generate_matrix_add_tile_function(data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2])
    elif op_info['func_name'] == 'maxpool_tile':
        code_line, full_func_name = generate_maxpool_tile_function(data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3], op_info["dims"][4], op_info["dims"][5], op_info["dims"][6], op_info["dims"][7], op_info["dims"][8], op_info["dims"][9])
    elif op_info['func_name'] == 'adaptive_avgpool_tile':
        code_line, full_func_name = generate_adaptive_avgpool_tile_function(data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2])
    elif op_info['func_name'] == 'avgpool_accumulate_tile':
        code_line, full_func_name = generate_avgpool_accumulate_tile_function(data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2])
    elif op_info['func_name'] == 'avgpool_finalize_tile':
        code_line, full_func_name = generate_avgpool_finalize_tile_function(data_type, op_info["dims"][0], op_info["dims"][1])
    elif op_info['func_name'] == 'matmul':
        code_line, full_func_name = generate_matmul_code(op_info["func_info"][0], data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["func_info"][1])
    elif op_info['func_name'] == 'mha':
        code_line, full_func_name = generate_grouped_mha_code(op_info["func_info"][0], data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3], op_info["func_info"][1])
    elif op_info['func_name'] == 'swa':
        code_line, full_func_name = generate_sliding_window_attention_code(op_info["func_info"][0], data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3], op_info["func_info"][1])
    elif op_info['func_name'] == 'layernorm':
        code_line, full_func_name = generate_layer_norm_code(op_info["func_info"][0], data_type,  op_info["dims"][0], op_info["dims"][1], op_info["dims"][2])
    elif op_info['func_name'] == 'rmsnorm':
        code_line, full_func_name = generate_rms_norm_code(op_info["func_info"][0], data_type,  op_info["dims"][0], op_info["dims"][1], op_info["dims"][2])
    elif op_info['func_name'] == 'activation':
        code_line, full_func_name = generate_activation_function(op_info["func_info"][0], op_info["func_info"][1], data_type,  op_info["dims"][0], op_info["dims"][1])
    elif op_info['func_name'] == 'activation_conv':
        code_line, full_func_name = generate_activation_function_conv(op_info["func_info"][0], op_info["func_info"][1], data_type,  op_info["dims"][0], op_info["dims"][1], op_info["dims"][2])
    elif op_info['func_name'] == 'dropout':
        code_line, full_func_name = generate_dropout_code(op_info["func_info"][0], data_type, op_info["dims"][0], op_info["dims"][1])
    elif op_info['func_name'] == 'matrix_add':
        code_line, full_func_name = generate_matrix_add_code(op_info["func_info"][0], data_type, op_info["dims"][0], op_info["dims"][1])
    elif op_info['func_name'] == 'matrix_add_conv':
        code_line, full_func_name = generate_matrix_add_conv_code(op_info["func_info"][0], data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2])
    elif op_info['func_name'] == 'elementwise_mult':
        code_line, full_func_name = generate_elementwise_mult_code(op_info["func_info"][0], data_type, op_info["dims"][0], op_info["dims"][1])
    elif op_info['func_name'] == 'conv':
        conv_dims = normalize_conv_dims(op_info["dims"])
        code_line, full_func_name = generate_conv_function(
            op_info["func_info"][0],
            op_info["func_info"][1],
            data_type,
            conv_dims[0],
            conv_dims[1],
            conv_dims[2],
            conv_dims[3],
            conv_dims[4],
            conv_dims[5],
            conv_dims[6],
            conv_dims[7],
            conv_dims[8],
            op_info["func_info"][2],
            conv_dims[9],
            conv_dims[10],
            conv_dims[11],
            conv_dims[12],
            conv_dims[13],
            conv_dims[14],
            conv_dims[15],
        )
    elif op_info['func_name'] == 'conv_tiled':
        code_line, full_func_name = generate_conv_tiled_function(op_info["func_info"][0], op_info["func_info"][1], data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3], op_info["dims"][4], op_info["dims"][5], op_info["dims"][6], op_info["dims"][7], op_info["dims"][8], op_info["dims"][9], op_info["dims"][10], op_info["dims"][11], op_info["dims"][12], op_info["dims"][13], op_info["dims"][14])
    elif op_info['func_name'] == 'batchnorm':
        code_line, full_func_name = generate_batch_norm_code(op_info["func_info"][0], data_type,  op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["func_info"][1])
    elif op_info['func_name'] == 'maxpool':
        code_line, full_func_name = generate_maxpool_code(op_info["func_info"][0], data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3], op_info["dims"][4], op_info["dims"][5], op_info["dims"][6], op_info["dims"][7], op_info["dims"][8])
    elif op_info['func_name'] == 'maxpool_pad':
        code_line, full_func_name = generate_maxpool_pad_code(op_info["func_info"][0], data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3], op_info["dims"][4], op_info["dims"][5], op_info["dims"][6], op_info["dims"][7], op_info["dims"][8], op_info["dims"][9], op_info["dims"][10])
    elif op_info['func_name'] == 'adaptive_avgpool':
        code_line, full_func_name = generate_adaptive_avgpool_code(op_info["func_info"][0], data_type, op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["dims"][3], op_info["dims"][4])
    
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
    code_lines.append(f"typedef {data_type} data_t;")
    code_lines.append("typedef ap_fixed<32,10> acc_t;\n")
    
    # 2. Declare BRAM arrays.
    for bram in brams:
        dims_str = "".join(f"[{d}]" for d in bram["dims"])
        code_lines.append(f"data_t {bram['name']}{dims_str};")
    
    code_lines.append("")  # blank line
    
    func_name_set = set()
    func_def_name_list = []
   
    control_ops = {"loop_begin", "loop_end"}

    for key, op_info in ops.items():
       if op_info["func_name"] in control_ops:
           continue
       func_def_code, func_name  = generate_func_def(op_info, data_type)
       if func_name and func_name not in func_name_set:
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
    indent_level = 1
    for key, op_info in ops.items():
        if op_info["func_name"] == "loop_begin":
            var_name, start_expr, end_expr, step_expr, valid_name, tile_expr = op_info["args"]
            indent = "    " * indent_level
            code_lines.append(f"{indent}for (int {var_name} = {start_expr}; {var_name} < {end_expr}; {var_name} += {step_expr}) {{")
            indent_level += 1
            if valid_name != "_":
                inner_indent = "    " * indent_level
                code_lines.append(
                    f"{inner_indent}int {valid_name} = ((({var_name}) + ({tile_expr})) <= ({end_expr}) ? ({tile_expr}) : (({end_expr}) - ({var_name})));"
                )
            continue
        if op_info["func_name"] == "loop_end":
            indent_level -= 1
            code_lines.append(f'{"    " * indent_level}}}')
            continue

        call_str = generate_operator_call(op_info, data_type)
        code_lines.append(f'{"    " * indent_level}{call_str}')
    
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
    # for dram in drams:
    #     lines.append(f"add_files -tb {dram['name']}.txt")
    
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
        lines.append("export_design -format ip_catalog -flow impl")
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

## Want to create a block opeartor
## the block creates a function that wraps the internal json
## this block can then be called in other jsons passing the corresponding

if __name__ == "__main__":
    # Define transformer block parameters.
    SEQ_LENGTH = 8
    DIM_IN     = 32
    NUM_HEAD   = 4
    DIM_Q      = DIM_IN / NUM_HEAD
    DIM_K      = DIM_IN / NUM_HEAD
    DIM_V      = DIM_IN / NUM_HEAD
    DIM_OUT    = 32

    brams = [
            {"name": "BRAM_attn_input",      "dims": [SEQ_LENGTH, DIM_IN]},
            {"name": "BRAM_weights_q",       "dims": [DIM_IN, DIM_IN]},
            {"name": "BRAM_weights_k",       "dims": [DIM_IN, DIM_IN]},
            {"name": "BRAM_weights_v",       "dims": [DIM_IN, DIM_IN]},
            {"name": "BRAM_1",     "dims": [SEQ_LENGTH, DIM_OUT]},
            {"name": "BRAM_2",     "dims": [SEQ_LENGTH, DIM_OUT]},
            {"name": "BRAM_MLP_1",     "dims": [SEQ_LENGTH, 4 * DIM_OUT]},
            {"name": "BRAM_MLP_2",     "dims": [SEQ_LENGTH, 4 * DIM_OUT]},
            {"name": "BRAM_layer_norm_weights_1",     "dims": [2, DIM_OUT]},
            {"name": "FF_weights_1",     "dims": [4 * DIM_OUT, DIM_OUT]},
            {"name": "FF_weights_2",     "dims": [DIM_OUT, 4 * DIM_OUT]},
            {"name": "BRAM_layer_norm_weights_2",     "dims": [2, DIM_OUT]},
            
        ]

        # Define DRAM configuration (for input/output).
    drams = [
        {"name": "DRAM_attn_input",  "dims": [SEQ_LENGTH, DIM_IN],  "bundle": "mem1"},
        {"name": "DRAM_weights_q",  "dims": [DIM_IN, DIM_IN],  "bundle": "mem1"},
        {"name": "DRAM_weights_k",  "dims": [DIM_IN, DIM_IN],  "bundle": "mem1"},
        {"name": "DRAM_weights_v",  "dims": [DIM_IN, DIM_IN],  "bundle": "mem1"},
        {"name": "DRAM_layer_norm_weights_1",  "dims": [2, DIM_OUT],  "bundle": "mem1"},
        {"name": "DRAM_FF_weights_1",  "dims": [4 * DIM_OUT, DIM_OUT],  "bundle": "mem1"},
        {"name": "DRAM_FF_weights_2",  "dims": [DIM_OUT, 4 * DIM_OUT],  "bundle": "mem1"},
        {"name": "DRAM_layer_norm_weights_2",  "dims": [2, DIM_OUT],  "bundle": "mem1"},
        {"name": "DRAM_output", "dims": [SEQ_LENGTH, DIM_OUT], "bundle": "mem2"}
    ]



    # Build the ops dictionary.
    ops = {
        "load_1": {
            "func_name": "load",
            "dims": [SEQ_LENGTH, DIM_IN],
            "args": ["DRAM_attn_input", "BRAM_attn_input"]
        },
        "load_2": {
            "func_name": "load",
            "dims": [DIM_IN, DIM_IN],
            "args": ["DRAM_weights_q", "BRAM_weights_q"]
        },
        "load_3": {
            "func_name": "load",
            "dims": [DIM_IN, DIM_IN],
            "args": ["DRAM_weights_k", "BRAM_weights_k"]
        },
        "load_4": {
            "func_name": "load",
            "dims": [DIM_IN, DIM_IN],
            "args": ["DRAM_weights_v", "BRAM_weights_v"]
        },
        "load_5": {
            "func_name": "load",
            "dims": [2, DIM_OUT],
            "args": ["DRAM_layer_norm_weights_1", "BRAM_layer_norm_weights_1"]
        },
        "load_6": {
            "func_name": "load",
            "dims": [4 * DIM_OUT, DIM_OUT],
            "args": ["DRAM_FF_weights_1", "FF_weights_1"]
        },
        "load_7": {
            "func_name": "load",
            "dims": [DIM_OUT, 4 * DIM_OUT],
            "args": ["DRAM_FF_weights_2", "FF_weights_2"]
        },
        "load_8": {
            "func_name": "load",
            "dims": [2, DIM_OUT],
            "args": ["DRAM_layer_norm_weights_2", "BRAM_layer_norm_weights_2"]
        },
        "mha": {
            "func_name": "mha",
            "dims": [SEQ_LENGTH, DIM_IN, NUM_HEAD, int(DIM_IN/NUM_HEAD)],
            "func_info": ["grouped_mha_rope_template.cpp", False],
            "args": ["BRAM_attn_input", "BRAM_weights_q", "BRAM_weights_k", "BRAM_weights_v", "BRAM_1", "8"]
        },
        "swa": {
            "func_name": "swa",
            "dims": [SEQ_LENGTH, DIM_IN, NUM_HEAD, int(DIM_IN/NUM_HEAD)],
            "func_info": ["sliding_window_attention_template.cpp", False],
            "args": ["BRAM_attn_input", "BRAM_weights_q", "BRAM_weights_k", "BRAM_weights_v", "BRAM_1", "8"]
        },
        "dropout": {
            "func_name": "dropout",
            "dims": [SEQ_LENGTH, DIM_OUT],
            "func_info": ["dropout_template.cpp"],
            "args": ["BRAM_1", "BRAM_2", "0.5", "47"]
        },
        "layernorm_1": {
            "func_name": "layernorm",
            "dims": [SEQ_LENGTH, DIM_OUT, 1e-2],
            "func_info": ["layer_norm_template.cpp"],
            "args": ["BRAM_2", "BRAM_layer_norm_weights_1[0]", "BRAM_layer_norm_weights_1[1]", "BRAM_1"]
        },
        "matmul_1": {
            "func_name": "matmul",
            "dims": [SEQ_LENGTH, DIM_OUT, 4 * DIM_OUT],
            "func_info": ["matmul_template.cpp", False],
            "args": ["BRAM_1", "FF_weights_1", "BRAM_MLP_1"]
        },
        "activation": {
                "func_name": "activation",
                "dims": [SEQ_LENGTH, 4 * DIM_OUT],
                "func_info":["activation_template.cpp", "gelu"],
                "args": ["BRAM_MLP_1", "BRAM_MLP_2"]
            },
        "matmul_2": {
            "func_name": "matmul",
            "dims": [SEQ_LENGTH, 4 * DIM_OUT, DIM_OUT],
            "func_info": ["matmul_template.cpp", False],
            "args": ["BRAM_MLP_2", "FF_weights_2", "BRAM_2"]
        },
        "dropout_2": {
            "func_name": "dropout",
            "dims": [SEQ_LENGTH, DIM_OUT],
            "func_info": ["dropout_template.cpp"],
            "args": ["BRAM_2", "BRAM_1", "0.5", "47"]
        },
        "layernorm_2": {
            "func_name": "layernorm",
            "dims": [SEQ_LENGTH, DIM_OUT, 1e-2],
            "func_info": ["layer_norm_template.cpp"],
            "args": ["BRAM_1", "BRAM_layer_norm_weights_2[0]", "BRAM_layer_norm_weights_2[1]", "BRAM_2"]
        },
        "matrix_add": {
            "func_name": "matrix_add",
            "dims": [SEQ_LENGTH, DIM_OUT, 1e-2],
            "func_info": ["matrix_add_template.cpp"],
            "args": ["BRAM_1", "BRAM_2", "BRAM_2"]
        },
        "store": {
            "func_name": "store",
            "dims": [SEQ_LENGTH, DIM_OUT],
            "args": ["BRAM_2", "DRAM_output"]
        }
    }

    output_dram_names = ["DRAM_output"]
        
    FPGA_name = "xczu9eg-ffvb1156-2-e"
    clock_period = 10
    task = ["csim", "csynth", "cosim", "export_ip"]
    #task = ["csynth"]
    # Generate the complete HLS C code for the top function.
    top_code = generate_top_function(brams, drams, ops, data_type="ap_fixed<16, 5>", top_func_name="top")

    # Write the generated code to a file, for example "top.cpp"
    output_filename = "top.cpp"
    with open(output_filename, "w") as f:
        f.write(top_code)
    print("Generated top.cpp:")    

    top_h_code = generate_top_h(drams, data_type="ap_fixed<16, 5>", top_func_name="top")
    with open("top.h", "w") as f:
        f.write(top_h_code)
    print("Generated top.h:")

    tb_code = generate_testbench_code(drams, output_dram_names, data_type="ap_fixed<16, 5>", top_func_name="top")
    with open("tb_top.cpp", "w") as f:
        f.write(tb_code)
    print("Generated tb_top.cpp:")

    generate_dram_txt_files(drams, seed=42)
    print("Generated dram initialization.")

    generate_full_tcl_file(drams, FPGA_name, clock_period, task, output_filename="run_hls.tcl")
    print("Generated tcl file to launch tasks.")
