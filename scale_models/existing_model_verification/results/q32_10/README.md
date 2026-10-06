# Q32.10 production results

All twelve ResNet and four Llama-capacity runs pass exact fixed-point comparison
and the 5% FP64 relative-L2 criterion.

| Model | Fixed | FP64 | Final-logit relative L2 |
|---|---|---|---:|
| ResNet-18 full/tiled | PASS | PASS | 0.000191553% |
| ResNet-34 full/tiled | PASS | PASS | 0.000218755% |
| ResNet-50 full/tiled | PASS | PASS | 0.000507193% |
| ResNet-101 full/tiled | PASS | PASS | 0.000629521% |
| ResNet-152 full/tiled | PASS | PASS | 0.000612012% |
| Llama-3-8B ctx2048, prefill + 2 decode | PASS | PASS | 0.000265945%–0.000306626% |
| Llama-3-8B ctx8192, prefill + 2 decode | PASS | PASS | 0.000265945%–0.000306626% |

Llama context values denote cache capacity. Both campaigns execute four prefill
tokens and two decode calls. Individual JSON reports retain every tensor metric,
source hash, input manifest, tool hash, and simulation provenance.
