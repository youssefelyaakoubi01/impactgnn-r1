**Table @rq5_cls.** Text × structure ablation at *k* = 3 (ROC-AUC for the top-10 % class; primary protocol, mean ± std over 15 runs). Δ is the paired difference between the graph model and the MLP with the same inputs. Exploratory analysis.

| Author and venue features | Text | MLP | GraphSAGE | GAT | Δ GraphSAGE − MLP | Δ GAT − MLP |
|---|---|---|---|---|---|---|
| no | No text | 0.744 ± 0.001 | 0.757 ± 0.002 | 0.749 ± 0.004 | +0.013 [−0.002, +0.028] | +0.005 [−0.015, +0.025] |
| no | TF-IDF | 0.694 ± 0.005 | 0.688 ± 0.011 | 0.707 ± 0.010 | −0.006 [−0.014, +0.002] | +0.012 [−0.0001, +0.024] |
| no | SciBERT | 0.772 ± 0.011 | 0.773 ± 0.011 | 0.773 ± 0.013 | +0.001 [−0.007, +0.009]ᵃ | +0.001 [−0.011, +0.013] |
| no | SPECTER | 0.747 ± 0.011 | 0.743 ± 0.015 | 0.748 ± 0.013 | −0.004 [−0.012, +0.003] | 0.000 [−0.010, +0.011] |
| yes | No text | 0.764 ± 0.002 | 0.775 ± 0.014 | 0.778 ± 0.006 | +0.011 [−0.001, +0.023] | +0.014 [+0.0003, +0.026]ᵇ |
| yes | TF-IDF | 0.755 ± 0.006 | 0.754 ± 0.009 | 0.750 ± 0.010 | −0.001 [−0.007, +0.006]ᵃ | −0.006 [−0.015, +0.004] |
| yes | SciBERT | 0.796 ± 0.013 | 0.801 ± 0.010 | 0.807 ± 0.015 | +0.005 [−0.002, +0.013] | +0.010 [+0.001, +0.020]ᵇ |
| yes | SPECTER | 0.787 ± 0.007 | 0.786 ± 0.011 | 0.786 ± 0.013 | −0.001 [−0.008, +0.007]ᵃ | −0.001 [−0.011, +0.010] |

ᵃ Equivalent: the 95 % interval lies within ±0.01. ᵇ Different: the 95 % interval excludes zero. Other differences are inconclusive. Abbreviations: MLP: multilayer perceptron; GAT: graph attention network; TF-IDF: term frequency–inverse document frequency; Δ: paired difference (first model minus second model, or model minus reference).
