**Table @rq5_kfold.** Text × structure ablation at *k* = 3 (Spearman correlation; retrospective K-fold, mean ± std over 12 test folds). Δ is the paired difference between the graph model and the MLP with the same inputs. Exploratory analysis.

| Author and venue features | Text | MLP | GraphSAGE | GAT | Δ GraphSAGE − MLP | Δ GAT − MLP |
|---|---|---|---|---|---|---|
| no | No text | 0.509 ± 0.015 | 0.600 ± 0.016 | 0.570 ± 0.017 | +0.091 [+0.083, +0.099]ᵇ | +0.061 [+0.054, +0.068]ᵇ |
| no | TF-IDF | 0.570 ± 0.021 | 0.588 ± 0.016 | 0.583 ± 0.017 | +0.018 [+0.010, +0.025]ᵇ | +0.012 [−0.001, +0.026] |
| no | SciBERT | 0.604 ± 0.018 | 0.625 ± 0.013 | 0.615 ± 0.017 | +0.021 [+0.014, +0.028]ᵇ | +0.011 [+0.006, +0.017]ᵇ |
| no | SPECTER | 0.596 ± 0.019 | 0.609 ± 0.015 | 0.599 ± 0.018 | +0.012 [+0.005, +0.019]ᵇ | +0.002 [−0.005, +0.0101] |
| yes | No text | 0.623 ± 0.012 | 0.672 ± 0.012 | 0.672 ± 0.014 | +0.049 [+0.040, +0.057]ᵇ | +0.049 [+0.038, +0.059]ᵇ |
| yes | TF-IDF | 0.655 ± 0.013 | 0.663 ± 0.012 | 0.666 ± 0.012 | +0.008 [+0.0004, +0.016]ᵇ | +0.012 [+0.003, +0.020]ᵇ |
| yes | SciBERT | 0.671 ± 0.014 | 0.682 ± 0.013 | 0.685 ± 0.012 | +0.011 [+0.005, +0.017]ᵇ | +0.015 [+0.006, +0.023]ᵇ |
| yes | SPECTER | 0.663 ± 0.011 | 0.676 ± 0.011 | 0.677 ± 0.012 | +0.013 [+0.005, +0.020]ᵇ | +0.014 [+0.007, +0.021]ᵇ |

ᵇ Different: the 95 % interval excludes zero. The other differences are inconclusive. Abbreviations: MLP: multilayer perceptron; GAT: graph attention network; TF-IDF: term frequency–inverse document frequency; Δ: paired difference (first model minus second model, or model minus reference).
