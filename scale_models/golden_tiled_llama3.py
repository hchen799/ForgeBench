"""Compatibility entry point; new configurable workflows use verification/verify.py."""
from verification.legacy.cli_llama import main

if __name__ == "__main__":
    main()
