**Table @rq5.** Text × structure ablation at *k* = 3 (Spearman correlation; primary protocol, mean ± std over 15 runs). Δ is the paired difference between the graph model and the MLP with the same inputs. Primary analysis.

| Author and venue features | Text | MLP | GraphSAGE | GAT | Δ GraphSAGE − MLP | Δ GAT − MLP |
|---|---|---|---|---|---|---|
| no | No text | 0.519 ± 0.002 | 0.531 ± 0.009 | 0.535 ± 0.003 | +0.012 [−0.004, +0.027] | +0.016 [+0.002, +0.030]ᵇ |
| no | TF-IDF | 0.489 ± 0.011 | 0.492 ± 0.011 | 0.492 ± 0.014 | +0.003 [−0.0102, +0.017] | +0.003 [−0.015, +0.020] |
| no | SciBERT | 0.574 ± 0.008 | 0.559 ± 0.007 | 0.542 ± 0.012 | −0.015 [−0.027, −0.004]ᵇ | −0.032 [−0.046, −0.018]ᵇ |
| no | SPECTER | 0.536 ± 0.005 | 0.527 ± 0.006 | 0.518 ± 0.010 | −0.009 [−0.022, +0.003] | −0.018 [−0.032, −0.004]ᵇ |
| yes | No text | 0.614 ± 0.001 | 0.629 ± 0.003 | 0.631 ± 0.004 | +0.015 [+0.001, +0.028]ᵇ | +0.017 [+0.004, +0.031]ᵇ |
| yes | TF-IDF | 0.595 ± 0.006 | 0.603 ± 0.005 | 0.595 ± 0.006 | +0.008 [−0.002, +0.018] | 0.000 [−0.013, +0.012] |
| yes | SciBERT | 0.639 ± 0.005 | 0.624 ± 0.009 | 0.614 ± 0.008 | −0.015 [−0.025, −0.006]ᵇ | −0.025 [−0.035, −0.014]ᵇ |
| yes | SPECTER | 0.620 ± 0.005 | 0.615 ± 0.006 | 0.606 ± 0.006 | −0.005 [−0.014, +0.005] | −0.014 [−0.025, −0.003]ᵇ |

ᵇ Different: the 95 % interval excludes zero. The other differences are inconclusive. None of these differences is a registered contrast. Abbreviations: MLP: multilayer perceptron; GAT: graph attention network; TF-IDF: term frequency–inverse document frequency; Δ: paired difference (first model minus second model, or model minus reference).
