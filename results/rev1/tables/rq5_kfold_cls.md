**Table @rq5_kfold_cls.** Text × structure ablation at *k* = 3 (ROC-AUC for the top-10 % class; retrospective K-fold, mean ± std over 12 test folds). Δ is the paired difference between the graph model and the MLP with the same inputs. Exploratory analysis.

| Author and venue features | Text | MLP | GraphSAGE | GAT | Δ GraphSAGE − MLP | Δ GAT − MLP |
|---|---|---|---|---|---|---|
| no | No text | 0.726 ± 0.011 | 0.807 ± 0.016 | 0.805 ± 0.014 | +0.081 [+0.065, +0.097]ᵇ | +0.079 [+0.066, +0.092]ᵇ |
| no | TF-IDF | 0.783 ± 0.018 | 0.797 ± 0.016 | 0.783 ± 0.015 | +0.014 [+0.002, +0.025]ᵇ | −0.001 [−0.020, +0.019] |
| no | SciBERT | 0.807 ± 0.013 | 0.820 ± 0.008 | 0.812 ± 0.007 | +0.013 [+0.003, +0.022]ᵇ | +0.005 [−0.010, +0.019] |
| no | SPECTER | 0.809 ± 0.013 | 0.816 ± 0.011 | 0.807 ± 0.013 | +0.007 [−0.0005, +0.015] | −0.002 [−0.012, +0.008] |
| yes | No text | 0.794 ± 0.012 | 0.840 ± 0.012 | 0.833 ± 0.014 | +0.046 [+0.034, +0.057]ᵇ | +0.039 [+0.030, +0.048]ᵇ |
| yes | TF-IDF | 0.839 ± 0.015 | 0.842 ± 0.011 | 0.830 ± 0.012 | +0.003 [−0.004, +0.011] | −0.008 [−0.023, +0.006] |
| yes | SciBERT | 0.843 ± 0.009 | 0.852 ± 0.009 | 0.846 ± 0.006 | +0.010 [+0.003, +0.017]ᵇ | +0.003 [−0.004, +0.011] |
| yes | SPECTER | 0.846 ± 0.011 | 0.848 ± 0.012 | 0.842 ± 0.010 | +0.001 [−0.006, +0.008]ᵃ | −0.004 [−0.011, +0.003] |

ᵃ Equivalent: the 95 % interval lies within ±0.01. ᵇ Different: the 95 % interval excludes zero. Other differences are inconclusive. Abbreviations: MLP: multilayer perceptron; GAT: graph attention network; TF-IDF: term frequency–inverse document frequency; Δ: paired difference (first model minus second model, or model minus reference).
