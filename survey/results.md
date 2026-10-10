| dataset | sampled | adapter fail | rows w/ calls | tokens p50 / p95 / max | >2048 | cut before answer @2048 | >4096 | args not JSON | undeclared tool | missing required | unknown key | tool msg w/o call | empty assistant | ends on user | dup rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NousResearch/hermes-function-calling-v1 | 1115 | 0 | 64.3% | 1838 / 3396 / 3964 | 44.8% | 38.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.4% | 0.0% | 9.7% | 0.0% |
| glaiveai/glaive-function-calling-v2 | 1492 | 0 | 54.4% | 404 / 1210 / 3537 | 0.2% | 0.1% | 0.0% | 0.1% | 0.0% | 0.2% | 0.0% | 0.1% | 0.0% | 1.4% | 0.0% |
| lockon/xlam-function-calling-60k | 1500 | 0 | 100.0% | 496 / 966 / 1984 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 1.5% | 0.1% | 0.0% | 0.0% | 0.0% | 0.3% |
| argilla/apigen-function-calling | 1500 | 0 | 100.0% | 482 / 910 / 2228 | 0.1% | 0.0% | 0.0% | 0.0% | 0.0% | 1.3% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| hiyouga/glaive-function-calling-v2-sharegpt | 1483 | 0 | 52.3% | 472 / 1225 / 2412 | 0.1% | 0.1% | 0.0% | 0.0% | 0.0% | 0.1% | 0.0% | 0.2% | 0.0% | 0.0% | 3.2% |
