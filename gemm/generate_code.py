import re
import random

def replace_data_type(data_type: str) -> str:
    # Replace all occurrences of <, >, and , with an underscore
    result = re.sub(r'[<>,]', '_', data_type)
    # Remove all spaces
    result = result.replace(" ", "")
    return result


def generate_load_function(
    dims,              # List of dimensions, e.g., [8, 16, 16]
    data_type="float", # Underlying data type
    func_prefix="load" # Function prefix (will be appended with dims and data_type)
):
    """
    Generates an HLS C function that copies an input array to an output array.
    The function is for arrays of dimensions given in the list 'dims'.
    
    The generated function will have a name of the form:
      <func_prefix>_<dim1>_<dim2>_..._<dimN>_data_t
    
    Parameters:
      dims: list of integers, e.g., [8, 16, 16]
      data_type: string, e.g., "float" (used in typedef and function signature)
      func_prefix: prefix for the function name (default "load")
    
    Returns:
      A string with the generated HLS C code.
    """
    # Construct the typedef and include lines
    code_lines = []
    
    
    #code_lines.append(f"typedef {data_type} data_t;\n")
    
    # Build array dimension string, e.g., "[8][16][16]"
    array_dims = "".join(f"[{d}]" for d in dims)
    
    # Construct the function name. E.g., load_8_16_16_data_t
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
    dims,              # List of dimensions, e.g., [8, 16, 16]
    data_type="float", # Underlying data type
    func_prefix="store" # Function prefix (will be appended with dims and data_type)
):
    """
    Generates an HLS C function that copies an input array to an output array.
    The function is for arrays of dimensions given in the list 'dims'.
    
    The generated function will have a name of the form:
      <func_prefix>_<dim1>_<dim2>_..._<dimN>_data_t
    
    Parameters:
      dims: list of integers, e.g., [8, 16, 16]
      data_type: string, e.g., "float" (used in typedef and function signature)
      func_prefix: prefix for the function name (default "load")
    
    Returns:
      A string with the generated HLS C code.
    """
    # Construct the typedef and include lines
    code_lines = []
    #code_lines.append(f"typedef {data_type} data_t;\n")
    
    # Build array dimension string, e.g., "[8][16][16]"
    array_dims = "".join(f"[{d}]" for d in dims)
    
    # Construct the function name. E.g., load_8_16_16_data_t
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

###### GEMM FUNCTIONS ######

"""
GEMM JSON FORM:
{
    "func_name": "gemm",
    "dims": [M, N, K],
    "func_info":[order, with_bias, inline],  
    "args": [input_A, input_B, bias, output] 
},

order is ['i', 'j', 'k'] or a permutation of it
with_bias is bool
inline is bool

"""

def generate_gemm_function(
        data_type="float",
        func_type="gemm",
        M=16, N=16, K=16,
        order=['i', 'j', 'k'],
        with_bias=False,
):
    """
    Generate a C++ HLS GEMM function based on the specified parameters.
    
    Args:
        data_type (str): Data type to use (e.g., "float", "int", "data_t")
        func_type (str): Function name
        M (int): First matrix dimension
        N (int): Second matrix dimension
        K (int): Third matrix dimension
        order (list): Order of loops (permutation of 'i', 'j', 'k')
        with_bias (bool): Whether to include bias in the computation
        
    Returns:
        str: The generated C++ function as a string
    """
    # Validate order parameter
    if sorted(order) != sorted(['i', 'j', 'k']):
        raise ValueError("The 'order' parameter must be a permutation of ['i', 'j', 'k']")
    
    # Determine function name and signature based on whether bias is included
    if with_bias:
        function_name = f"{func_type}_{''.join(order)}_bias"
        function_signature =    f"""void {function_name}(
    {data_type} input_A[{M}][{N}],
    {data_type} input_B[{N}][{K}],
    {data_type} bias[{M}][{K}],
    {data_type} output[{M}][{K}]
)"""
        init_out = "output[i][k] = bias[i][k];"

    else:
        function_name = f"{func_type}_{''.join(order)}"
        function_signature =    f"""void {function_name}(
    {data_type} input_A[{M}][{N}],
    {data_type} input_B[{N}][{K}],
    {data_type} output[{M}][{K}]
)"""
        init_out = "output[i][k] = 0;"

    computation = "output[i][k] += input_A[i][j] * input_B[j][k];"
    
    # Create the loop structure based on specified order
    loop_starts = ""
    loop_ends = ""
    
    loop_vars = {'i': M, 'j': N, 'k': K}
    
    for var in order:
        limit = loop_vars[var]
        loop_starts += f"for (int {var} = 0; {var} < {limit}; {var}++) {{\n"
        loop_ends = "}\n" + loop_ends
    
    # Create the complete function
    function =  f"""//////////////////////////////////////////
// Begin: {function_name.upper()} FUNCTION{' with BIAS' if with_bias else ''}
//////////////////////////////////////////
/*==== {function_name.upper()} FUNCTION START ====*/
{function_signature}

{{
{loop_starts}    {init_out}
{loop_ends}

{loop_starts}    {computation}
{loop_ends}}}
/*==== {function_name.upper()} FUNCTION END ====*/
//////////////////////////////////////////
// END: {function_name.upper()} FUNCTION{' with BIAS' if with_bias else ''}
//////////////////////////////////////////
"""
    ext_func_name = f"{function_name}_{M}_{N}_{K}"
    return function, ext_func_name

def call_gemm(
        func_type="gemm",
        M=16, N=16, K=16,
        order=['i', 'j', 'k'],
        with_bias=False,
        input_A_var="input_A",
        input_B_var="input_B",
        bias_var="bias",
        output_var="output"
):
    """
    Generate C code that calls the appropriate GEMM function.
    
    Args:
        func_type (str): Function name prefix
        M, N, K (int): Matrix dimensions
        order (list): Order of loops (permutation of 'i', 'j', 'k')
        with_bias (bool): Whether to include bias in the computation
        input_A_var (str): Variable name for input A
        input_B_var (str): Variable name for input B
        bias_var (str): Variable name for bias
        output_var (str): Variable name for output
        
    Returns:
        str: C code that calls the GEMM function
    """
    # Validate order parameter
    if sorted(order) != sorted(['i', 'j', 'k']):
        raise ValueError("The 'order' parameter must be a permutation of ['i', 'j', 'k']")
    
    # Determine function name based on parameters
    if with_bias:
        function_name = f"{func_type}_{''.join(order)}_bias"
        call_code = f"{function_name}({input_A_var}, {input_B_var}, {bias_var}, {output_var});"
    else:
        function_name = f"{func_type}_{''.join(order)}"
        call_code = f"{function_name}({input_A_var}, {input_B_var}, {output_var});"
    
    # Create the complete function call with appropriate comment
    result = f"""//////////////////////////////////////////
// Begin: Call to {function_name.upper()}
//////////////////////////////////////////
{call_code}
//////////////////////////////////////////
// End: Call to {function_name.upper()}
//////////////////////////////////////////
"""
    
    return result

def call_gemm_inline(
        func_type="gemm",
        M=16, N=16, K=16,
        order=['i', 'j', 'k'],
        with_bias=False,
        input_A_var="input_A",
        input_B_var="input_B",
        bias_var="bias",
        output_var="output"
):
    """
    Generate inline C code that implements GEMM functionality without a function call.
    
    Args:
        func_type (str): Used for naming in comments only
        M, N, K (int): Matrix dimensions
        order (list): Order of loops (permutation of 'i', 'j', 'k')
        with_bias (bool): Whether to include bias in the computation
        input_A_var (str): Variable name for input A
        input_B_var (str): Variable name for input B
        bias_var (str): Variable name for bias
        output_var (str): Variable name for output
        
    Returns:
        str: Inline C code that implements the GEMM functionality
    """
    # Validate order parameter
    if sorted(order) != sorted(['i', 'j', 'k']):
        raise ValueError("The 'order' parameter must be a permutation of ['i', 'j', 'k']")
    
    # Determine function name (for comments only) and computation
    if with_bias:
        function_name = f"{func_type}_{''.join(order)}_bias"
        init_out = f"{output_var}[i][k] = {bias_var}[i][k];"
    else:
        function_name = f"{func_type}_{''.join(order)}"
        init_out = f"{output_var}[i][k] = 0;"
    
    computation = f"{output_var}[i][k] += {input_A_var}[i][j] * {input_B_var}[j][k];"
    
    # Create the loop structure based on specified order
    loop_starts = ""
    loop_ends = ""
    
    loop_vars = {'i': M, 'j': N, 'k': K}
    
    for var in order:
        limit = loop_vars[var]
        loop_starts += f"for (int {var} = 0; {var} < {limit}; {var}++) {{\n"
        loop_ends = "}\n" + loop_ends
    
    # Create the complete inline implementation
    inline_code =   f"""//////////////////////////////////////////
// Begin: Inline implementation of {function_name.upper()}
//////////////////////////////////////////
{loop_starts}    {init_out}
{loop_ends}

{loop_starts}    {computation}
{loop_ends}//////////////////////////////////////////
// End: Inline implementation of {function_name.upper()}
//////////////////////////////////////////
"""
    
    return inline_code

####### MM-V FUNCTIONS #######

"""
MM-V JSON FORM:
{
    "func_name": "mmv",
    "dims": [M, N],
    "func_info":[order, with_bias, inline],  
    "args": [input_A, input_B, bias, output] 
},

order is ['i', 'j'] or a permutation of it
with_bias is bool
inline is bool

"""

def generate_mmv_function(
        data_type="int",
        func_type="mmv",
        M=16, N=16,
        order=['i', 'j',],
        with_bias=False,
):
    """
    Generate a C++ HLS Matrix * Vector (MMV) function based on the specified parameters.
    
    Args:
        data_type (str): Data type to use (e.g., "float", "int", "data_t")
        func_type (str): Function name
        M (int): First matrix dimension
        N (int): Second matrix dimension
        order (list): Order of loops (permutation of 'i', 'j',) leading dim determines if output is row or column vector, i -> column, j -> row
        with_bias (bool): Whether to include bias in the computation
        
    Returns:
        str: The generated C++ function as a string
    """
    # Validate order parameter
    if sorted(order) != sorted(['i', 'j']):
        raise ValueError("The 'order' parameter must be a permutation of ['i', 'j']")
    

    # Determine function name and signature based on whether bias is included
    if with_bias:
        function_name = f"{func_type}_{''.join(order)}_bias"
        function_signature =    f"""void {function_name}(
    {data_type} input_A[{M}][{N}],
    {data_type} input_B[{N}],
    {data_type} bias[{M}],
    {data_type} output[{M}]
)"""
        init_out = "output[i] = bias[i];"

    else:
        function_name = f"{func_type}_{''.join(order)}"
        function_signature =    f"""void {function_name}(
    {data_type} input_A[{M}][{N}],
    {data_type} input_B[{N}],
    {data_type} output[{M}]
)"""
        init_out = "output[i] = 0;"
    
    computation = "output[i] += input_A[i][j] * input_B[j];"
    
    # Create the loop structure based on specified order
    loop_starts = ""
    loop_ends = ""
    
    loop_vars = {'i': M, 'j': N,}
    
    for var in order:
        limit = loop_vars[var]
        loop_starts += f"for (int {var} = 0; {var} < {limit}; {var}++) {{\n"
        loop_ends = "}\n" + loop_ends
    
    # Create the complete function
    function =  f"""//////////////////////////////////////////
// Begin: {function_name.upper()} FUNCTION{' with BIAS' if with_bias else ''}
//////////////////////////////////////////
/*==== {function_name.upper()} FUNCTION START ====*/
{function_signature}
{{
{loop_starts}    {init_out}
{loop_ends}

{loop_starts}    {computation}
{loop_ends}}}
/*==== {function_name.upper()} FUNCTION END ====*/
//////////////////////////////////////////
// END: {function_name.upper()} FUNCTION{' with BIAS' if with_bias else ''}
//////////////////////////////////////////
"""
    ext_func_name = f"{function_name}_{M}_{N}"
    return function, ext_func_name

def call_mmv(
        func_type="mmv",
        M=16, N=16,
        order=['i', 'j',],
        with_bias=False,  
        input_A_var="input_A",
        input_B_var="input_B",
        bias_var="bias",
        output_var="output"
):
    """
    Generate C code that calls the appropriate Matrix * Vector (MMV) function.
    
    Args:
        data_type (str): Data type to use
        func_type (str): Function name prefix
        M, N (int): Matrix dimensions
        order (list): Order of loops (permutation of 'i', 'j',)
        with_bias (bool): Whether to include bias in the computation
        input_A_var (str): Variable name for input A
        input_B_var (str): Variable name for input B
        bias_var (str): Variable name for bias
        output_var (str): Variable name for output
        
    Returns:
        str: C code that calls the MMV function
    """
    # Validate order parameter
    if sorted(order) != sorted(['i', 'j']):
        raise ValueError("The 'order' parameter must be a permutation of ['i', 'j']")
    
    # Determine function name based on parameters
    if with_bias:
        function_name = f"{func_type}_{''.join(order)}_bias"
        call_code = f"{function_name}({input_A_var}, {input_B_var}, {bias_var}, {output_var});"
    else:
        function_name = f"{func_type}_{''.join(order)}"
        call_code = f"{function_name}({input_A_var}, {input_B_var}, {output_var});"
    
    # Create the complete function call with appropriate comment
    result = f"""//////////////////////////////////////////
    // Begin: Call to {function_name.upper()}
//////////////////////////////////////////
{call_code}
//////////////////////////////////////////
// End: Call to {function_name.upper()}
//////////////////////////////////////////
"""
    
    return result

def call_mmv_inline(
        func_type="mmv",
        M=16, N=16,
        order=['i', 'j',],
        with_bias=False,  
        input_A_var="input_A",
        input_B_var="input_B",
        bias_var="bias",
        output_var="output"
):
    """
    Generate inline C code that implements Matrix * Vector (MMV) functionality without a function call.

    Args:
        func_type (str): Function name prefix
        M, N (int): Matrix dimensions
        order (list): Order of loops (permutation of 'i', 'j',)
        with_bias (bool): Whether to include bias in the computation
        input_A_var (str): Variable name for input A
        input_B_var (str): Variable name for input B
        bias_var (str): Variable name for bias
        output_var (str): Variable name for output
        
    Returns:
        str: Inline C code that implements the MMV functionality
    """

    if sorted(order) != sorted(['i', 'j']):
        raise ValueError("The 'order' parameter must be a permutation of ['i', 'j']")
    
        
    # Determine function name (for comments only) and computation
    if with_bias:
        function_name = f"{func_type}_{''.join(order)}_bias"
        init_out = f"{output_var}[i] = {bias_var}[i];"
    else:
        function_name = f"{func_type}_{''.join(order)}"
        init_out = f"{output_var}[i] = 0;"
    
    computation = f"{output_var}[i] += {input_A_var}[i][j] * {input_B_var}[j];"
    
    loop_starts = ""
    loop_ends = ""
    
    loop_vars = {'i': M, 'j': N,}
    
    for var in order:
        limit = loop_vars[var]
        loop_starts += f"for (int {var} = 0; {var} < {limit}; {var}++) {{\n"
        loop_ends = "}\n" + loop_ends
    
    # Create the complete inline implementation
    inline_code =   f"""//////////////////////////////////////////
// Begin: Inline implementation of {function_name.upper()}
//////////////////////////////////////////
{loop_starts}    {init_out}
{loop_ends}

{loop_starts}    {computation}
{loop_ends}//////////////////////////////////////////
// End: Inline implementation of {function_name.upper()}
//////////////////////////////////////////
"""
    
    return inline_code

####### V-MM FUNCTIONS #######

"""
V-MM JSON FORM:
{
    "func_name": "vmm",
    "dims": [M, N],
    "func_info":[order, with_bias, inline],  
    "args": [input_A, input_B, bias, output] 
},

order is ['i', 'j'] or a permutation of it, 
with_bias is bool
inline is bool

"""

def generate_vmm_function(
        data_type="int",
        func_type="vmm",
        M=16, N=16,
        order=['i', 'j',],
        with_bias=False,
):
    """
    Generate a C++ HLS Vector * Matrix (VMM) function based on the specified parameters.
    
    Args:
        data_type (str): Data type to use (e.g., "float", "int", "data_t")
        func_type (str): Function name
        M (int): First matrix dimension
        N (int): Second matrix dimension
        order (list): Order of loops (permutation of 'i', 'j',) 
        with_bias (bool): Whether to include bias in the computation
        
    Returns:
        str: The generated C++ function as a string
    """
    # Validate order parameter
    if sorted(order) != sorted(['i', 'j']):
        raise ValueError("The 'order' parameter must be a permutation of ['i', 'j']")
    

    # Determine function name and signature based on whether bias is included
    if with_bias:
        function_name = f"{func_type}_{''.join(order)}_bias"
        function_signature =    f"""void {function_name}(
    {data_type} input_A[{M}][{N}],
    {data_type} input_B[{M}],
    {data_type} bias[{N}],
    {data_type} output[{N}]
)"""
        init_out = "output[j] = bias[j];"
    else:
        function_name = f"{func_type}_{''.join(order)}"
        function_signature =    f"""void {function_name}(
    {data_type} input_A[{M}][{N}],
    {data_type} input_B[{M}],
    {data_type} output[{N}]
)"""
        init_out = "output[j] = 0;"

    computation = "output[j] += input_A[i][j] * input_B[i];"

    
    # Create the loop structure based on specified order
    loop_starts = ""
    loop_ends = ""
    
    loop_vars = {'i': M, 'j': N,}
    
    for var in order:
        limit = loop_vars[var]
        loop_starts += f"for (int {var} = 0; {var} < {limit}; {var}++) {{\n"
        loop_ends = "}\n" + loop_ends
    
    # Create the complete function
    function =  f"""//////////////////////////////////////////
// Begin: {function_name.upper()} FUNCTION{' with BIAS' if with_bias else ''}
//////////////////////////////////////////
/*==== {function_name.upper()} FUNCTION START ====*/
{function_signature}
{{
{loop_starts}    {init_out}
{loop_ends}

{loop_starts}    {computation}
{loop_ends}}}
/*==== {function_name.upper()} FUNCTION END ====*/
//////////////////////////////////////////
// END: {function_name.upper()} FUNCTION{' with BIAS' if with_bias else ''}
//////////////////////////////////////////
"""
    
    ext_func_name = f"{function_name}_{M}_{N}"
    return function, ext_func_name

def call_vmm(
        data_type="int",
        func_type="vmm",
        M=16, N=16,
        order=['i', 'j',],
        with_bias=False,  
        input_A_var="input_A",
        input_B_var="input_B",
        bias_var="bias",
        output_var="output"
):
    """
    Generate C code that calls the appropriate Vector * Matrix (VMM) function.
    
    Args:
        data_type (str): Data type to use
        func_type (str): Function name prefix
        M, N (int): Matrix dimensions
        order (list): Order of loops (permutation of 'i', 'j',)
        with_bias (bool): Whether to include bias in the computation
        input_A_var (str): Variable name for input A
        input_B_var (str): Variable name for input B
        bias_var (str): Variable name for bias
        output_var (str): Variable name for output
        
    Returns:
        str: C code that calls the VMM function
    """
    # Validate order parameter
    if sorted(order) != sorted(['i', 'j']):
        raise ValueError("The 'order' parameter must be a permutation of ['i', 'j']")
    
    # Determine function name based on parameters
    if with_bias:
        function_name = f"{func_type}_{''.join(order)}_bias"
        call_code = f"{function_name}({input_A_var}, {input_B_var}, {bias_var}, {output_var});"
    else:
        function_name = f"{func_type}_{''.join(order)}"
        call_code = f"{function_name}({input_A_var}, {input_B_var}, {output_var});"
    
    # Create the complete function call with appropriate comment
    result = f"""//////////////////////////////////////////
    // Begin: Call to {function_name.upper()}
//////////////////////////////////////////
{call_code}
//////////////////////////////////////////
// End: Call to {function_name.upper()}
//////////////////////////////////////////
"""
    
    return result

def call_vmm_inline(
        func_type="vmm",
        M=16, N=16,
        order=['i', 'j',],
        with_bias=False,  
        input_A_var="input_A",
        input_B_var="input_B",
        bias_var="bias",
        output_var="output"
):
    
    """
    Generate inline C code that implements Vector * Matrix (VMM) functionality without a function call.
    
    Args:
        func_type (str): Function name prefix
        M, N (int): Matrix dimensions
        order (list): Order of loops (permutation of 'i', 'j',)
        with_bias (bool): Whether to include bias in the computation
        input_A_var (str): Variable name for input A
        input_B_var (str): Variable name for input B
        bias_var (str): Variable name for bias
        output_var (str): Variable name for output
        
    Returns:
        str: Inline C code that implements the VMM functionality
    """
        
    if sorted(order) != sorted(['i', 'j']):
        raise ValueError("The 'order' parameter must be a permutation of ['i', 'j']")
    
        
    # Determine function name (for comments only) and computation
    if with_bias:
        function_name = f"{func_type}_{''.join(order)}_bias"
        init_out = f"{output_var}[j] = {bias_var}[j];"
    else:
        function_name = f"{func_type}_{''.join(order)}"
        init_out = f"{output_var}[j] = 0;"

    computation = f"{output_var}[j] += {input_A_var}[i][j] * {input_B_var}[i];"

    loop_starts = ""
    loop_ends = ""
    
    loop_vars = {'i': M, 'j': N,}
    
    for var in order:
        limit = loop_vars[var]
        loop_starts += f"for (int {var} = 0; {var} < {limit}; {var}++) {{\n"
        loop_ends = "}\n" + loop_ends
    
    # Create the complete inline implementation
    inline_code =   f"""//////////////////////////////////////////
// Begin: Inline implementation of {function_name.upper()}
//////////////////////////////////////////
{loop_starts}    {init_out}
{loop_ends}

{loop_starts}    {computation}
{loop_ends}//////////////////////////////////////////
// End: Inline implementation of {function_name.upper()}
//////////////////////////////////////////
"""
    
    return inline_code

####### DOT PRODUCT FUNCTIONS #######

"""
DOT PRODUCT JSON FORM:
{
    "func_name": "dot_product",
    "dims": [M],
    "func_info":[with_bias, inline],  
    "args": [input_A, input_B, bias, output] 
},

with_bias is bool
inline is bool

"""

def generate_dot_function(
        data_type="int",
        func_type="dot_product",
        M=16,
        with_bias=False,
):
    """
    Generate a C++ HLS Dot Product function based on the specified parameters.
    
    Args:
        data_type (str): Data type to use (e.g., "float", "int", "data_t")
        func_type (str): Function name
        M (int): Vector dimension
        with_bias (bool): Whether to include bias in the computation
        
    Returns:
        str: The generated C++ function as a string
    """
    # Determine function name and signature based on whether bias is included
    if with_bias:
        function_name = f"{func_type}_bias"
        function_signature =    f"""void {function_name}(
    {data_type} input_A[{M}],
    {data_type} input_B[{M}],
    {data_type} bias[1],
    {data_type} output[1]
)"""
        init_out = "output[0] = bias[0];"

    else:
        function_name = f"{func_type}"
        function_signature =    f"""void {function_name}(
    {data_type} input_A[{M}],
    {data_type} input_B[{M}],
    {data_type} output[1]
)"""
        init_out = "output[0] = 0;"
    
    computation = "output[0] += input_A[i] * input_B[i];"
    
    # Create the loop structure based on specified order
    loop_starts = ""
    loop_ends = ""
    
    for var in ['i']:
        loop_starts += f"for (int {var} = 0; {var} < {M}; {var}++) {{\n"
        loop_ends = "}\n" + loop_ends
    
    # Create the complete function
    function =  f"""//////////////////////////////////////////
// Begin: {function_name.upper()} FUNCTION{' with BIAS' if with_bias else ''}
//////////////////////////////////////////
/*==== {function_name.upper()} FUNCTION START ====*/
{function_signature}
{{
{loop_starts}    {init_out}
{loop_ends}

{loop_starts}    {computation}
{loop_ends}}}
/*==== {function_name.upper()} FUNCTION END ====*/
//////////////////////////////////////////
// END: {function_name.upper()} FUNCTION{' with BIAS' if with_bias else ''}
//////////////////////////////////////////
"""
    
    return function, function_name

def call_dot(
        func_type="dot_product",
        M=16,
        with_bias=False,  
        input_A_var="input_A",
        input_B_var="input_B",
        bias_var="bias",
        output_var="output"
):
    """
    Generate C code that calls the appropriate Dot Product function.

    Args:
        func_type (str): Function name
        M (int): Vector dimension
        with_bias (bool): Whether to include bias in the computation
        input_A_var (str): Variable name for input A
        input_B_var (str): Variable name for input B
        bias_var (str): Variable name for bias
        output_var (str): Variable name for output

    Returns:
        str: C code that calls the Dot Product function
    """
    
    # Determine function name based on parameters
    if with_bias:
        function_name = f"{func_type}_bias"
        call_code = f"{function_name}({input_A_var}, {input_B_var}, {bias_var}, {output_var});"
    else:
        function_name = f"{func_type}"
        call_code = f"{function_name}({input_A_var}, {input_B_var}, {output_var});"
    
    # Create the complete function call with appropriate comment
    result = f"""//////////////////////////////////////////
    // Begin: Call to {function_name.upper()}
//////////////////////////////////////////
{call_code}
//////////////////////////////////////////
// End: Call to {function_name.upper()}
//////////////////////////////////////////
"""
    
    return result

def call_dot_inline(
        func_type="dot_product",
        M=16,
        with_bias=False,  
        input_A_var="input_A",
        input_B_var="input_B",
        bias_var="bias",
        output_var="output"
):
    """
    Generate inline C code that implements Dot Product functionality without a function call.

    Args:
        func_type (str): Function name
        M (int): Vector dimension
        with_bias (bool): Whether to include bias in the computation
        input_A_var (str): Variable name for input A
        input_B_var (str): Variable name for input B
        bias_var (str): Variable name for bias
        output_var (str): Variable name for output

    Returns:
        str: Inline C code that implements the Dot Product functionality
    """

    # Determine function name (for comments only) and computation
    if with_bias:
        function_name = f"{func_type}_bias"
        init_out = f"{output_var}[0] = {bias_var}[0];"
    else:
        function_name = f"{func_type}"
        init_out = f"{output_var}[0] = 0;"

    computation = f"{output_var}[0] += {input_A_var}[i] * {input_B_var}[i];"
    
    loop_starts = ""
    loop_ends = ""
    
    for var in ['i']:
        loop_starts += f"for (int {var} = 0; {var} < {M}; {var}++) {{\n"
        loop_ends = "}\n" + loop_ends
    
    # Create the complete inline implementation
    inline_code =   f"""//////////////////////////////////////////
// Begin: Inline implementation of {function_name.upper()}
//////////////////////////////////////////
{loop_starts}    {init_out}
{loop_ends}

{loop_starts}    {computation}
{loop_ends}//////////////////////////////////////////
// End: Inline implementation of {function_name.upper()}
//////////////////////////////////////////
"""
    
    return inline_code

####### 2D ACTIVATION FUNCTIONS #######

def generate_activation_function(
    template_path,      # Path to activations_template.cpp
    func_name,          # One of: "relu", "leaky_relu", "prelu", "rrelu", "thresholded_relu", "relu6", "sigmoid", "tanh_act", "elu", "selu", "gelu", "swish", "softmax"
    DATA_TYPE="float",
    H=64,
    W=64
):
    """
    Reads an external activation functions template, substitutes common placeholders,
    extracts the specified function block (for a 3D tensor with dimensions [C][H][W]),
    appends the dimension information to the function name, and writes the final HLS C code to output_path.
    
    Expected placeholders in the template:
      {DATA_TYPE}, {H}, {W}
    """
    # 1) Read the full template file.
    with open(template_path, "r") as f:
        template_code = f.read()
    
    # 2) Substitute common placeholders.
    formatted_code = template_code.format(
        DATA_TYPE=DATA_TYPE,
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
    dim_suffix = f"_{H}_{W}_{DATA_TYPE}"
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

def generate_func_def(op_info, data_type):
    
    if op_info['func_name'] == 'load':
        code_line, full_func_name = generate_load_function(op_info["dims"], data_type, func_prefix="load")
    elif op_info['func_name'] == 'store':
        code_line, full_func_name = generate_store_function(op_info["dims"], data_type, func_prefix="store")
    elif op_info['func_name'] == 'gemm':
        code_line, full_func_name = generate_gemm_function(data_type, "gemm", op_info["dims"][0], op_info["dims"][1], op_info["dims"][2], op_info["func_info"][0], op_info["func_info"][1])
    elif op_info['func_name'] == 'vmm':
        code_line, full_func_name = generate_vmm_function(data_type, "vmm", op_info["dims"][0], op_info["dims"][1], op_info["func_info"][0], op_info["func_info"][1])
    elif op_info['func_name'] == 'mmv':
        code_line, full_func_name = generate_mmv_function(data_type, "mmv", op_info["dims"][0], op_info["dims"][1], op_info["func_info"][0], op_info["func_info"][1])
    elif op_info['func_name'] == 'dot_product':
        code_line, full_func_name = generate_dot_function(data_type, "dot_product", op_info["dims"][0], op_info["func_info"][0])
    elif op_info['func_name'] == 'activation':
        code_line, full_func_name = generate_activation_function(op_info["func_info"][0], op_info["func_info"][1], data_type,  op_info["dims"][0], op_info["dims"][1], op_info["dims"][2])

    else:
        print("the operator we do not support!")
        
    return code_line, full_func_name

def generate_operator_call(op_info, data_type):
    """
    Generates a single function call string given an operator dictionary.
    
    op_info: dict with keys:
       - "func_name": base function name (e.g., "load", "conv2d", etc.)
       - "dims": list of integers (e.g., [8,16,16] or [8,16,16,8,16,16])
       - "args": list of argument strings for the function call
       
    Returns a string like:
       load_8_16_16_data_t(DRAM_1, BRAM_1);
    """

    # Append _data_t to follow the convention.
    if op_info['func_name'] == 'load':
        code_line, full_func_name = generate_load_function(op_info["dims"], data_type, func_prefix="load")
        args_str = ", ".join(op_info["args"])
        call_str =  f"{full_func_name}({args_str});"
    elif op_info['func_name'] == 'store':
        code_line, full_func_name = generate_store_function(op_info["dims"], data_type, func_prefix="store")
        args_str = ", ".join(op_info["args"])
        call_str =  f"{full_func_name}({args_str});"
    elif op_info['func_name'] == 'activation':
        code_line, full_func_name = generate_activation_function(op_info["func_info"][0], op_info["func_info"][1], data_type,  op_info["dims"][0], op_info["dims"][1], op_info["dims"][2])
        args_str = ", ".join(op_info["args"])
        call_str =  f"{full_func_name}({args_str});"
    elif op_info['func_name'] == 'gemm':
        if op_info["func_info"][-1]: # inline
            call_str = call_gemm_inline(func_type="gemm", M=op_info["dims"][0], N=op_info["dims"][1], K=op_info["dims"][2], 
                        order=op_info["func_info"][0], with_bias=op_info["func_info"][1], 
                        input_A_var=op_info["args"][0], input_B_var=op_info["args"][1], bias_var=op_info["args"][2], output_var=op_info["args"][3])
        else:
            call_str = call_gemm(func_type="gemm", M=op_info["dims"][0], N=op_info["dims"][1], K=op_info["dims"][2], 
                        order=op_info["func_info"][0], with_bias=op_info["func_info"][1], 
                        input_A_var=op_info["args"][0], input_B_var=op_info["args"][1], bias_var=op_info["args"][2], output_var=op_info["args"][3])
    elif op_info['func_name'] == 'vmm':
        if op_info["func_info"][-1]:
            call_str = call_vmm_inline(func_type="vmm", M=op_info["dims"][0], N=op_info["dims"][1], 
                        order=op_info["func_info"][0], with_bias=op_info["func_info"][1], 
                        input_A_var=op_info["args"][0], input_B_var=op_info["args"][1], bias_var=op_info["args"][2], output_var=op_info["args"][3])
        else:
            call_str = call_vmm(func_type="vmm", M=op_info["dims"][0], N=op_info["dims"][1], 
                        order=op_info["func_info"][0], with_bias=op_info["func_info"][1], 
                        input_A_var=op_info["args"][0], input_B_var=op_info["args"][1], bias_var=op_info["args"][2], output_var=op_info["args"][3])
    elif op_info['func_name'] == 'mmv':
        if op_info["func_info"][-1]:
            call_str = call_mmv_inline(func_type="mmv", M=op_info["dims"][0], N=op_info["dims"][1], 
                        order=op_info["func_info"][0], with_bias=op_info["func_info"][1], 
                        input_A_var=op_info["args"][0], input_B_var=op_info["args"][1], bias_var=op_info["args"][2], output_var=op_info["args"][3])
        else:
            call_str = call_mmv(func_type="mmv", M=op_info["dims"][0], N=op_info["dims"][1], 
                        order=op_info["func_info"][0], with_bias=op_info["func_info"][1], 
                        input_A_var=op_info["args"][0], input_B_var=op_info["args"][1], bias_var=op_info["args"][2], output_var=op_info["args"][3])
    elif op_info['func_name'] == 'dot_product':
        if op_info["func_info"][-1]:
            call_str = call_dot_inline(func_type="dot_product", M=op_info["dims"][0], 
                        with_bias=op_info["func_info"][0], 
                        input_A_var=op_info["args"][0], input_B_var=op_info["args"][1], bias_var=op_info["args"][2], output_var=op_info["args"][3])
        else:
            call_str = call_dot(func_type="dot_product", M=op_info["dims"][0], 
                        with_bias=op_info["func_info"][0], 
                        input_A_var=op_info["args"][0], input_B_var=op_info["args"][1], bias_var=op_info["args"][2], output_var=op_info["args"][3])
    else:
        print("the operator we do not support!")

    return call_str

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
            numbers = [str(random.random()) for _ in range(total_elements)]
            # Write each number on a new line.
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
    lines.append("open_project project_1")
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

if __name__ == "__main__":
    # Example BRAM configuration:
    brams = [
        {"name": "BRAM_1", "dims": [16, 8]}, # Used to store Matrix 1 [M, N]
        {"name": "BRAM_2", "dims": [8, 32]}, # Used to store Matrix 2 [N, K]
        {"name": "BRAM_3", "dims": [16, 32]}, # Used to store bias [M, K]
        {"name": "BRAM_4", "dims": [16, 32]}, # Used to store output [M, K]

        {"name": "BRAM_5", "dims": [32]}, # Used to store Vector 2 [K]
        {"name": "BRAM_6", "dims": [16]}, # Unused Bias
        {"name": "BRAM_7", "dims": [16]}, # Used to store output of BRAM_4 * BRAM_5 -> [K]

        {"name": "BRAM_8", "dims": [32]}, # Unused Bias
        {"name": "BRAM_9", "dims": [32]}, # Used to store output of BRAM_7 * BRAM_4 -> [M]
        
        {"name": "BRAM_10", "dims": [32]},
        {"name": "BRAM_11", "dims": [1]}, # Bias of BRAM_9 * BRAM_10 -> [1]
        {"name": "BRAM_12", "dims": [1]},# Output of BRAM_7 * BRAM_8 + bias -> [1]
    ]
    
    # Example DRAM configuration:
    drams = [
        # {"name": "DRAM_1", "dims": [2, 4, 4], "bundle": "mem1"}, # Used to load the input [c, h, w]
        # {"name": "DRAM_2", "dims": [2, 2, 3, 3], "bundle": "mem1"}, # Used to load the conv weights [cout, cin, k, k]
        # {"name": "DRAM_3", "dims": [2], "bundle": "mem1"},  #Used to load the bias [cout]
        # {"name": "DRAM_4", "dims": [4, 2], "bundle": "mem1"}, #Used to load the batch norm weights [4][cout]
        # {"name": "DRAM_5", "dims": [2, 2, 2], "bundle": "mem2"} #Used to write back the output [c, h, w]

        {"name": "DRAM_1", "dims": [16, 8], "bundle": "mem1"}, # Used to load the Matrix 1 [M, N]
        {"name": "DRAM_2", "dims": [8, 32], "bundle": "mem1"}, # Used to load the Matrix 2 [N, K]
        {"name": "DRAM_3", "dims": [16, 32], "bundle": "mem1"}, # Used to load the bias [K]
        {"name": "DRAM_4", "dims": [16, 32], "bundle": "mem2"}, # Used to write back the output [M, K]

        {"name": "DRAM_5", "dims": [32], "bundle": "mem1"}, # Used to load the Vector 2 [K]
        # {"name": "DRAM_6", "dims": [16], "bundle": "mem2"}, # Unused Bias
        # {"name": "DRAM_7", "dims": [16], "bundle": "mem2"}, # Used to write back the output of BRAM_4 * BRAM_5 -> [K]

        # {"name": "DRAM_8", "dims": [32], "bundle": "mem3"}, # Unused Bias
        # {"name": "DRAM_9", "dims": [32], "bundle": "mem3"}, # Used to write back the output of BRAM_6 * BRAM_4 -> [M]
        
        {"name": "DRAM_10", "dims": [32], "bundle": "mem1"},
        {"name": "DRAM_11", "dims": [1], "bundle": "mem1"}, # Bias of BRAM_9 * BRAM_10 -> [1]
        {"name": "DRAM_12", "dims": [1], "bundle": "mem2"},# Output of BRAM_7 * BRAM_8 + bias

    ]
    
    # Example operators dictionary.
    # The keys determine the order of function calls.
    ops = {
        "load_1": {
            "func_name": "load",
            "dims": [16, 8],
            "args": ["DRAM_1", "BRAM_1"]
        },
        "load_2": {
            "func_name": "load",
            "dims": [8, 32],
            "args": ["DRAM_2", "BRAM_2"]
        },
        "load_3": {
            "func_name": "load",
            "dims": [16, 32],
            "args": ["DRAM_3", "BRAM_3"]
        },
        "load_4": {
            "func_name": "load",
            "dims": [32],
            "args": ["DRAM_5", "BRAM_5"]
        },

        "load_5": {
            "func_name": "load",
            "dims": [32],
            "args": ["DRAM_10", "BRAM_10"]
        },

        "load_6": {
            "func_name": "load",
            "dims": [1],
            "args": ["DRAM_11", "BRAM_11"]
        },

        "gemm_1": {
            "func_name": "gemm",
            "dims": [16, 8, 32],
            "func_info": [['i', 'j', 'k'], True, True],
            "args": ["BRAM_1", "BRAM_2", "BRAM_3", "BRAM_4"]
        },
        
        "store_1": {
            "func_name": "store",
            "dims": [16, 32],
            "args": ["BRAM_4", "DRAM_4"]
        },

        "vmm": {
            "func_name": "vmm",
            "dims": [16, 32],
            "func_info": [['i', 'j'], False, False],
            "args": ["BRAM_4", "BRAM_5", "BRAM_6", "BRAM_7"]
        },

        "mmv": {
            "func_name": "mmv",
            "dims": [16, 32],
            "func_info": [['j', 'i'], False, False],
            "args": ["BRAM_4", "BRAM_7", "BRAM_8", "BRAM_9"]
        },

        "dot_product": {
            "func_name": "dot_product",
            "dims": [16],
            "func_info": [True, False, True],
            "args": ["BRAM_9", "BRAM_10", "BRAM_11", "BRAM_12"]
        },

        "store_2": {
            "func_name": "store",
            "dims": [1],
            "args": ["BRAM_12", "DRAM_12"]
        }
    }
    
    output_dram_names = ["DRAM_4", "DRAM_12"]
    
    FPGA_name = "xczu9eg-ffvb1156-2-e"
    clock_period = 10
    task = ["csim", "csynth", "cosim", "export_ip"]
    
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
    
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

