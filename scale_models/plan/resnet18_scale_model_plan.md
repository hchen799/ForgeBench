# Add Standard ResNet-18 Scale-Model Config Generation

## Summary
Add ResNet-18 as a first-class generator in `ForgeBench/scale_models/auto_generate_json.py` and emit `auto_generated_configs/RESNET18_config_ap_fixed_16_5_.json`.

Target the standard ImageNet ResNet-18:
- Input `3x224x224`
- Stem `7x7/2`, BN, ReLU, maxpool, then stages `[64, 128, 256, 512]` with `2` basic blocks each
- Downsample on the first block of stages 2-4 with `1x1/2`
- Final adaptive average pool to `1x1`
- Classifier as `1x1 conv` from `512 -> 1000`

## Implementation Changes
- Add ResNet-18 basic-block and full-architecture generators in `scale_models/auto_generate_json.py`.
- Add a generic scale-model JSON serializer and emit `RESNET18_config_ap_fixed_16_5_.json` from `main()`.
- Add padded maxpool operator support in `scale_models/generate_code.py` and a new `maxpool_pad_template.cpp`.

## Test Plan
- Run `python auto_generate_json.py` and confirm the ResNet-18 JSON is generated.
- Parse the generated JSON and verify the stem, 8 residual blocks, downsample ops, final pool, and classifier dimensions.
- Run `scale_models/gen_configs.py` against the generated ResNet-18 config and confirm codegen completes without unsupported-operator errors.

## Assumptions
- “ResNet-18” means the standard ImageNet classifier with `1000` classes.
- `scale_models` is the only target; no new `conv/test_case_configs` files are needed.
- The stem pool uses a padded maxpool operator because the existing `scale_models` maxpool template does not support padding.
