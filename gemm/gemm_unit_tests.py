import unittest
import os
import sys
from generate_code import generate_gemm_function, call_gemm, call_gemm_inline

def write_to_file(filename: str, content: str, test_type: str) -> None:
    """Write content to a file."""
    write_path = os.path.join("generator_tests", "gemm", test_type)
    os.makedirs(write_path, exist_ok=True)
    with open(os.path.join(write_path, filename), "w") as f:
        f.write(content)

class TestGemmFunctionGenerator(unittest.TestCase):
    """Test suite for the GEMM function generator."""

    def test_generate_gemm_function_default(self):
        """Test generate_gemm_function with default parameters."""
        result = generate_gemm_function()
        write_to_file("gemm_default.c", result, "function_definition")
        self.assertIn("void gemm_ijk(", result)
        self.assertIn("float input_A[64][64]", result)
        self.assertIn("float input_B[64][64]", result)
        self.assertIn("float output[64][64]", result)
        self.assertIn("for (int i = 0; i < 64; i++)", result)
        self.assertIn("for (int j = 0; j < 64; j++)", result)
        self.assertIn("for (int k = 0; k < 64; k++)", result)
        self.assertIn("output[i][k] += input_A[i][j] * input_B[j][k];", result)
        self.assertNotIn("bias", result)

    def test_generate_gemm_function_with_bias(self):
        """Test generate_gemm_function with bias enabled."""
        result = generate_gemm_function(with_bias=True)
        write_to_file("gemm_bias.c", result, "function_definition")
        self.assertIn("void gemm_ijk_bias(", result)
        self.assertIn("float bias[64][64]", result)
        self.assertIn("output[i][k] += input_A[i][j] * input_B[j][k] + bias[i][k];", result)
        
    def test_generate_gemm_function_custom_loop_order(self):
        """Test generate_gemm_function with custom loop ordering."""
        result = generate_gemm_function(order=['k', 'j', 'i'])
        write_to_file("gemm_custom_order.c", result, "function_definition")
        self.assertIn("void gemm_kji(", result)
        self.assertIn("for (int k = 0; k < 64; k++)", result)
        self.assertIn("for (int j = 0; j < 64; j++)", result)
        self.assertIn("for (int i = 0; i < 64; i++)", result)
        
    def test_generate_gemm_function_custom_dimensions(self):
        """Test generate_gemm_function with custom dimensions."""
        result = generate_gemm_function(M=32, N=128, K=64)
        write_to_file("gemm_custom_dimensions.c", result, "function_definition")
        self.assertIn("float input_A[32][128]", result)
        self.assertIn("float input_B[128][64]", result)
        self.assertIn("float output[32][64]", result)
        self.assertIn("for (int i = 0; i < 32; i++)", result)
        self.assertIn("for (int j = 0; j < 128; j++)", result)
        self.assertIn("for (int k = 0; k < 64; k++)", result)
        
    def test_generate_gemm_function_custom_data_type(self):
        """Test generate_gemm_function with custom data type."""
        result = generate_gemm_function(data_type="data_t")
        write_to_file("gemm_custom_data_type.c", result, "function_definition")
        self.assertIn("data_t input_A[64][64]", result)
        self.assertIn("data_t input_B[64][64]", result)
        self.assertIn("data_t output[64][64]", result)
        
    def test_generate_gemm_function_invalid_order(self):
        """Test generate_gemm_function with invalid loop order."""
        with self.assertRaises(ValueError):
            generate_gemm_function(order=['i', 'j', 'x'])
        
        with self.assertRaises(ValueError):
            generate_gemm_function(order=['i', 'i', 'k'])
            
        with self.assertRaises(ValueError):
            generate_gemm_function(order=['i', 'j'])
            
    def test_call_gemm_default(self):
        """Test call_gemm with default parameters."""
        result = call_gemm()
        write_to_file("call_gemm_default.c", result, "function_call")
        self.assertIn("gemm_ijk(input_A, input_B, output);", result)
        self.assertIn("// Begin: Call to GEMM_IJK", result)
        self.assertIn("// End: Call to GEMM_IJK", result)
        self.assertNotIn("bias", result)
        
    def test_call_gemm_with_bias(self):
        """Test call_gemm with bias enabled."""
        result = call_gemm(with_bias=True)
        write_to_file("call_gemm_bias.c", result, "function_call")
        self.assertIn("gemm_ijk_bias(input_A, input_B, bias, output);", result)
        self.assertIn("// Begin: Call to GEMM_IJK_BIAS", result)
        self.assertIn("// End: Call to GEMM_IJK_BIAS", result)
        
    def test_call_gemm_custom_loop_order(self):
        """Test call_gemm with custom loop ordering."""
        result = call_gemm(order=['k', 'j', 'i'])
        write_to_file("call_gemm_custom_order.c", result, "function_call")
        self.assertIn("gemm_kji(input_A, input_B, output);", result)
        self.assertIn("// Begin: Call to GEMM_KJI", result)
        self.assertIn("// End: Call to GEMM_KJI", result)
        
    def test_call_gemm_custom_variable_names(self):
        """Test call_gemm with custom variable names."""
        result = call_gemm(
            input_A_var="weights",
            input_B_var="activations",
            bias_var="bias_terms",
            output_var="results"
        )
        write_to_file("call_gemm_custom_vars.c", result, "function_call")
        self.assertIn("gemm_ijk(weights, activations, results);", result)
        
    def test_call_gemm_custom_function_type(self):
        """Test call_gemm with custom function type."""
        result = call_gemm(func_type="matrix_multiply")
        write_to_file("call_gemm_custom_type.c", result, "function_call")
        self.assertIn("matrix_multiply_ijk(input_A, input_B, output);", result)
        self.assertIn("// Begin: Call to MATRIX_MULTIPLY_IJK", result)
        self.assertIn("// End: Call to MATRIX_MULTIPLY_IJK", result)
        
    def test_call_gemm_invalid_order(self):
        """Test call_gemm with invalid loop order."""
        with self.assertRaises(ValueError):
            call_gemm(order=['i', 'j', 'x'])
            
    def test_call_gemm_inline_default(self):
        """Test call_gemm_inline with default parameters."""
        result = call_gemm_inline()
        write_to_file("call_gemm_inline_default.c", result, "inlines")
        self.assertIn("// Begin: Inline implementation of GEMM_IJK", result)
        self.assertIn("// End: Inline implementation of GEMM_IJK", result)
        self.assertIn("for (int i = 0; i < 64; i++) {", result)
        self.assertIn("for (int j = 0; j < 64; j++) {", result)
        self.assertIn("for (int k = 0; k < 64; k++) {", result)
        self.assertIn("output[i][k] += input_A[i][j] * input_B[j][k];", result)
        self.assertNotIn("bias", result)
        
    def test_call_gemm_inline_with_bias(self):
        """Test call_gemm_inline with bias enabled."""
        result = call_gemm_inline(with_bias=True)
        write_to_file("call_gemm_inline_bias.c", result, "inlines")
        self.assertIn("// Begin: Inline implementation of GEMM_IJK_BIAS", result)
        self.assertIn("// End: Inline implementation of GEMM_IJK_BIAS", result)
        self.assertIn("output[i][k] += input_A[i][j] * input_B[j][k] + bias[i][k];", result)
        
    def test_call_gemm_inline_custom_loop_order(self):
        """Test call_gemm_inline with custom loop ordering."""
        result = call_gemm_inline(order=['k', 'j', 'i'])
        write_to_file("call_gemm_inline_custom_order.c", result, "inlines")
        self.assertIn("// Begin: Inline implementation of GEMM_KJI", result)
        self.assertIn("for (int k = 0; k < 64; k++) {", result)
        self.assertIn("for (int j = 0; j < 64; j++) {", result)
        self.assertIn("for (int i = 0; i < 64; i++) {", result)
        
    def test_call_gemm_inline_custom_dimensions(self):
        """Test call_gemm_inline with custom dimensions."""
        result = call_gemm_inline(M=32, N=128, K=64)
        write_to_file("call_gemm_inline_custom_dimensions.c", result, "inlines")
        self.assertIn("for (int i = 0; i < 32; i++) {", result)
        self.assertIn("for (int j = 0; j < 128; j++) {", result)
        self.assertIn("for (int k = 0; k < 64; k++) {", result)
        
    def test_call_gemm_inline_custom_variable_names(self):
        """Test call_gemm_inline with custom variable names."""
        result = call_gemm_inline(
            input_A_var="weights",
            input_B_var="activations",
            bias_var="bias_terms",
            output_var="results"
        )
        write_to_file("call_gemm_inline_custom_vars.c", result, "inlines")
        self.assertIn("results[i][k] += weights[i][j] * activations[j][k];", result)
        
    def test_call_gemm_inline_invalid_order(self):
        """Test call_gemm_inline with invalid loop order."""
        with self.assertRaises(ValueError):
            call_gemm_inline(order=['i', 'j', 'x'])
            
    def test_complex_scenario(self):
        """Test a complex scenario with all three functions."""
        params = {
            "data_type": "int",
            "func_type": "matmul",
            "M": 32, 
            "N": 64, 
            "K": 128,
            "order": ['j', 'i', 'k'],
            "with_bias": True,
            "input_A_var": "layer1_weights",
            "input_B_var": "layer1_input",
            "bias_var": "layer1_bias",
            "output_var": "layer1_output"
        }
        
        # Function definition
        func_def = generate_gemm_function(
            data_type=params["data_type"],
            func_type=params["func_type"],
            M=params["M"],
            N=params["N"],
            K=params["K"],
            order=params["order"],
            with_bias=params["with_bias"]
        )
        
        # Function call
        func_call = call_gemm(**params)
        
        # Inline implementation
        inline_impl = call_gemm_inline(**params)
        j = "\n".join([func_def, func_call, inline_impl])
        write_to_file("joint.c", j, '.')
        # Verify function definition
        self.assertIn("void matmul_jik_bias(", func_def)
        self.assertIn("int input_A[32][64]", func_def)
        self.assertIn("int input_B[64][128]", func_def)
        self.assertIn("int bias[32][128]", func_def)
        self.assertIn("int output[32][128]", func_def)
        
        # Verify function call
        self.assertIn("matmul_jik_bias(layer1_weights, layer1_input, layer1_bias, layer1_output);", func_call)
        
        # Verify inline implementation
        self.assertIn("// Begin: Inline implementation of MATMUL_JIK_BIAS", inline_impl)
        self.assertIn("for (int j = 0; j < 64; j++) {", inline_impl)
        self.assertIn("for (int i = 0; i < 32; i++) {", inline_impl)
        self.assertIn("for (int k = 0; k < 128; k++) {", inline_impl)
        self.assertIn("layer1_output[i][k] += layer1_weights[i][j] * layer1_input[j][k] + layer1_bias[i][k];", inline_impl)


if __name__ == "__main__":
    unittest.main()