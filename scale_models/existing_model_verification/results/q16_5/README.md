# Q16.5 production results

These are the preserved `ap_fixed<16,5>` results generated and executed through
the production JSON-to-HLS workflow. Fixed-point comparison requires exact
integer-code agreement. FP64 comparison requires max absolute error at most
0.1 and relative L2 error at most 0.01 for every observed tensor.

| Model | Fixed | FP64 | Logits relative L2 |
|---|---|---|---:|
| [ResNet-18 full](resnet18_full.json) | PASS | PASS | 0.511272% |
| [ResNet-18 tiled](resnet18_tiled.json) | PASS | PASS | 0.511272% |
| [ResNet-50 full](resnet50_full.json) | PASS | FAIL | 1.241235% |
| [ResNet-50 tiled](resnet50_tiled.json) | PASS | FAIL | 1.241235% |
| [Llama-3-8B](llama3_8b.json), prefill + two decode | PASS | FAIL | 123.293%–125.028% |

The failures are intentional preserved measurements, not missing runs. See the
[detailed result explanation](../DETAILED_RESULTS.md) and the
[Q32.10 additional campaign](../q32_10/README.md).
