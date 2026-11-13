**Table @topics.** Performance within each of the 48 topics (Spearman correlation at *k* = 3, out-of-fold predictions of the retrospective K-fold, averaged over runs). Exploratory analysis.

| Model | Median over topics [quartiles] | Range | Topics where the model exceeds the MLP | Median gain over the MLP | Wilcoxon p | Correlation of the gain with the reference coverage of the topic |
|---|---|---|---|---|---|---|
| MLP | 0.597 [0.546, 0.635] | 0.211 to 0.715 | – | – | – | – |
| MLP + all-reference aggregates | 0.614 [0.564, 0.652] | 0.308 to 0.718 | 39 of 48 | +0.018 | < 0.001 | −0.14 |
| GraphSAGE | 0.598 [0.570, 0.641] | 0.206 to 0.722 | 36 of 48 | +0.014 | < 0.001 | +0.06 |
| GAT | 0.601 [0.556, 0.643] | 0.263 to 0.731 | 34 of 48 | +0.010 | < 0.001 | +0.13 |

Abbreviations: MLP: multilayer perceptron; GAT: graph attention network; Wilcoxon p: p-value of the Wilcoxon signed-rank test over the topics.
