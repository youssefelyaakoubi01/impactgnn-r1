**Table @attention.** Spearman correlation between the last-layer attention of GAT and the attributes of the references, computed within each citing test paper with at least three in-graph references (mean over papers, then mean ± std over 12 test folds of the retrospective K-fold). Exploratory analysis.

| Model | *k* | Network | Correlation with prior citations | Share of papers with a positive correlation | Correlation with reference age | Share positive (age) | Test papers per fold |
|---|---|---|---|---|---|---|---|
| GAT | 3 | trained | 0.198 ± 0.020 | 0.72 | 0.378 ± 0.036 | 0.86 | 1,682 |
| GAT | 3 | untrained | 0.120 ± 0.008 | 0.64 | 0.077 ± 0.012 | 0.59 | 1,682 |
| GAT | 5 | trained | 0.262 ± 0.027 | 0.78 | 0.428 ± 0.031 | 0.89 | 941 |
| GAT | 5 | untrained | 0.120 ± 0.013 | 0.64 | 0.087 ± 0.012 | 0.59 | 941 |
| GAT + edge features | 3 | trained | −0.034 ± 0.122 | 0.46 | 0.168 ± 0.128 | 0.67 | 1,682 |
| GAT + edge features | 3 | untrained | −0.061 ± 0.010 | 0.42 | −0.102 ± 0.009 | 0.37 | 1,682 |
| GAT + edge features | 5 | trained | 0.136 ± 0.045 | 0.66 | 0.287 ± 0.045 | 0.80 | 941 |
| GAT + edge features | 5 | untrained | −0.078 ± 0.015 | 0.40 | −0.111 ± 0.012 | 0.36 | 941 |

Abbreviations: GAT: graph attention network; *k*: horizon in years.
