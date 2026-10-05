"""Compatibility entry point for the frozen, validated ResNet-18 workflow."""
from verification.legacy.cli_resnet import main

if __name__ == "__main__":
    raise SystemExit(main())
