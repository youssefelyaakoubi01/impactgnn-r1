**Table @biblio.** Author and venue statistics at *k* = 3 (mean ± std over runs or over test folds). Exact: citations received before publication; strict: no citation counts; initial: statistics of the initial version, which contain hindsight. Δ is the paired difference of the MLP with the same model on the exact statistics. Exploratory analysis.

| Protocol | Text | Variant | MLP: Spearman | MLP: Δ Spearman vs exact | GraphSAGE: Spearman | MLP: ROC-AUC | GraphSAGE: ROC-AUC |
|---|---|---|---|---|---|---|---|
| Primary (15 runs) | No text | Exact | 0.614 ± 0.001 | reference | 0.629 ± 0.003 | 0.764 ± 0.002 | 0.775 ± 0.013 |
|  | No text | Strict | 0.586 ± 0.002 | −0.028 [−0.038, −0.018]ᵇ | 0.593 ± 0.003 | 0.753 ± 0.003 | 0.755 ± 0.006 |
|  | No text | Initial | 0.643 ± 0.005 | +0.029 [+0.020, +0.039]ᵇ | 0.661 ± 0.005 | 0.781 ± 0.005 | 0.788 ± 0.012 |
|  | SciBERT | Exact | 0.639 ± 0.005 | reference | 0.624 ± 0.009 | 0.796 ± 0.013 | 0.801 ± 0.010 |
|  | SciBERT | Strict | 0.614 ± 0.007 | −0.025 [−0.033, −0.017]ᵇ | 0.600 ± 0.007 | 0.790 ± 0.012 | 0.788 ± 0.014 |
|  | SciBERT | Initial | 0.665 ± 0.005 | +0.026 [+0.019, +0.033]ᵇ | 0.655 ± 0.006 | 0.810 ± 0.013 | 0.808 ± 0.015 |
| K-fold (12 folds) | No text | Exact | 0.623 ± 0.012 | reference | 0.673 ± 0.013 | 0.794 ± 0.012 | 0.840 ± 0.012 |
|  | No text | Strict | 0.596 ± 0.013 | −0.027 [−0.033, −0.021]ᵇ | 0.650 ± 0.015 | 0.776 ± 0.014 | 0.827 ± 0.018 |
|  | No text | Initial | 0.660 ± 0.010 | +0.037 [+0.030, +0.044]ᵇ | 0.703 ± 0.009 | 0.818 ± 0.012 | 0.861 ± 0.011 |
|  | SciBERT | Exact | 0.671 ± 0.014 | reference | 0.682 ± 0.013 | 0.843 ± 0.009 | 0.852 ± 0.009 |
|  | SciBERT | Strict | 0.645 ± 0.017 | −0.026 [−0.033, −0.019]ᵇ | 0.655 ± 0.016 | 0.826 ± 0.012 | 0.836 ± 0.009 |
|  | SciBERT | Initial | 0.703 ± 0.011 | +0.032 [+0.024, +0.039]ᵇ | 0.711 ± 0.010 | 0.859 ± 0.010 | 0.864 ± 0.008 |

ᵇ Different: the 95 % interval excludes zero. The other differences are inconclusive. Abbreviations: MLP: multilayer perceptron; ROC-AUC: area under the ROC curve; K-fold: 6-fold cross-validation repeated twice (retrospective); Δ: paired difference (first model minus second model, or model minus reference).
