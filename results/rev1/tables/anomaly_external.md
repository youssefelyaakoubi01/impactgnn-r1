**Table @anomaly_external.** Descriptive check on the real venue graph: rank of the 16 venues that were suppressed from the Journal Citation Reports for anomalous citation patterns (2018–2025 releases) among the 3,136 venues. The p-value comes from 5,000 permutations of the labels. Exploratory analysis.

| Detector | ROC-AUC | p | ROC-AUC against size-matched controls | Median percentile of the suppressed venues | Suppressed venues in the top decile |
|---|---|---|---|---|---|
| Reciprocity | 0.524 | 0.363 | 0.467 | 0.63 | 1 |
| Self-citation ratio | 0.709 | 0.001 | 0.654 | 0.74 | 6 |
| Combined | 0.674 | 0.008 | 0.618 | 0.66 | 4 |
| Null-model excess | 0.588 | 0.115 | 0.532 | 0.56 | 3 |
| DOMINANT | 0.433 | 0.829 | 0.463 | 0.30 | 1 |
| Graph autoencoder | 0.456 | 0.730 | 0.515 | 0.54 | 0 |
| Isolation Forest | 0.452 | 0.739 | 0.462 | 0.43 | 0 |

Abbreviations: ROC-AUC: area under the ROC curve; Combined: the larger of the two heuristics, each expressed as a percentile rank.
