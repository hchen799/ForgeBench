#!/usr/bin/env python3
"""
Complete test runner for the GEMM generator suite.
This script demonstrates the usage of all GEMM generator functions
and runs the test suite to verify their correctness.
"""

import unittest
import sys
from generate_code import generate_gemm_function, call_gemm, call_gemm_inline
from gemm_unit_tests import TestGemmFunctionGenerator

def show_example_usage():
    """Demonstrate example usage of all GEMM generator functions."""
    print("=" * 80)
    print("EXAMPLE USAGE OF GEMM GENERATOR FUNCTIONS")
    print("=" * 80)
    
    # Basic example - Standard GEMM function
    print("\n1. STANDARD GEMM FUNCTION\n")
    
    std_gemm = generate_gemm_function()
    print(std_gemm)
    
    std_call = call_gemm()
    print(std_call)
    
    std_inline = call_gemm_inline()
    print(std_inline)
    
    # Example with bias
    print("\n2. GEMM FUNCTION WITH BIAS\n")
    
    bias_gemm = generate_gemm_function(with_bias=True)
    print(bias_gemm)
    
    bias_call = call_gemm(with_bias=True)
    print(bias_call)
    
    bias_inline = call_gemm_inline(with_bias=True)
    print(bias_inline)
    
    # Example with custom loop ordering
    print("\n3. GEMM FUNCTION WITH CUSTOM LOOP ORDERING\n")
    
    order_gemm = generate_gemm_function(order=['j', 'k', 'i'])
    print(order_gemm)
    
    order_call = call_gemm(order=['j', 'k', 'i'])
    print(order_call)
    
    order_inline = call_gemm_inline(order=['j', 'k', 'i'])
    print(order_inline)
    
    # Complex example with all customizations
    print("\n4. COMPLEX EXAMPLE WITH ALL CUSTOMIZATIONS\n")
    
    params = {
        "data_type": "data_t",
        "func_type": "matrix_multiply",
        "M": 32, 
        "N": 128, 
        "K": 64,
        "order": ['k', 'i', 'j'],
        "with_bias": True,
        "input_A_var": "weights",
        "input_B_var": "input_data",
        "bias_var": "bias_params",
        "output_var": "output_result"
    }
    
    complex_gemm = generate_gemm_function(
        data_type=params["data_type"],
        func_type=params["func_type"],
        M=params["M"],
        N=params["N"],
        K=params["K"],
        order=params["order"],
        with_bias=params["with_bias"]
    )
    print(complex_gemm)
    
    complex_call = call_gemm(**params)
    print(complex_call)
    
    complex_inline = call_gemm_inline(**params)
    print(complex_inline)

def run_tests():
    """Run the complete test suite."""
    print("=" * 80)
    print("RUNNING TEST SUITE")
    print("=" * 80)
    
    # Create a test suite and add the test cases
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGemmFunctionGenerator)
    
    # Run the tests
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    # Return success or failure
    return result.wasSuccessful()

if __name__ == "__main__":
    # Show example usage
    show_example_usage()
    
    # Run tests if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        success = run_tests()
        sys.exit(0 if success else 1)
    else:
        print("\nTo run tests, use the --test option:")
        print("python test_gemm.py --test")