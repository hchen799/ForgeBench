# Q16.5 production results

FP64 acceptance is relative L2 <= 5% at every observed tensor. Fixed comparison
requires exact integer-code agreement; all twelve ResNet and four Llama-capacity
runs meet that fixed requirement.

| Model | Fixed | FP64 | Final-logit relative L2 |
|---|---|---|---:|
| ResNet-18 full/tiled | PASS | PASS | 0.511272% |
| ResNet-34 full/tiled | PASS | PASS | 0.764899% |
| ResNet-50 full/tiled | PASS | PASS | 1.241235% |
| ResNet-101 full/tiled | PASS | PASS | 1.942698% |
| ResNet-152 full/tiled | PASS | PASS | 2.046199% |
| Llama-3-8B ctx2048, prefill + 2 decode | PASS | FAIL | 123.293%–125.028% |
| Llama-3-8B ctx8192, prefill + 2 decode | PASS | FAIL | 123.293%–125.028% |

The Llama runs use four prefill tokens and two decode calls; context is cache
capacity. Individual reports in this directory retain every tensor metric.
