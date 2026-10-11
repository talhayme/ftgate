| dataset | sampled | adapter fail | rows w/ calls | tokens p50 / p95 / max | >2048 | cut before answer @2048 | >4096 | args not JSON | undeclared tool | missing required | unknown key | tool msg w/o call | empty assistant | no final assistant turn | exact dup rows | same conversation, other tools |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NousResearch/hermes-function-calling-v1 | 1893 | 0 | 57.6% | 2170 / 3438 / 4750 | 52.4% | 45.1% | 0.1% | 0.0% | 0.0% | 0.0% | 0.0% | 0.5% | 0.0% | 8.6% | 0.0% | 0.0% |
| glaiveai/glaive-function-calling-v2 | 1492 | 0 | 54.4% | 404 / 1210 / 3537 | 0.2% | 0.1% | 0.0% | 0.1% | 0.0% | 0.2% | 0.0% | 0.1% | 0.0% | 1.4% | 0.0% | 0.0% |
| lockon/xlam-function-calling-60k | 1500 | 0 | 100.0% | 496 / 966 / 1984 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 1.5% | 0.1% | 0.0% | 0.0% | 0.0% | 0.1% | 0.3% |
| argilla/apigen-function-calling | 1500 | 0 | 100.0% | 482 / 910 / 2228 | 0.1% | 0.0% | 0.0% | 0.0% | 0.0% | 1.3% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| hiyouga/glaive-function-calling-v2-sharegpt | 1483 | 0 | 52.3% | 472 / 1225 / 2412 | 0.1% | 0.1% | 0.0% | 0.0% | 0.0% | 0.1% | 0.0% | 0.2% | 0.0% | 0.0% | 0.0% | 3.2% |
| allenai/Dolci-Instruct-SFT-Tool-Use | 1500 | 0 | 91.3% | 1194 / 8897 / 334792 | 16.9% | 14.9% | 6.7% | 3.8% | 0.1% | 0.0% | 0.1% | 0.1% | 0.5% | 0.1% | 0.0% | 0.0% |
