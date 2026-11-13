**Table @anomaly_matched.** ROC-AUC of each detector for the injected venues against controls matched on size (mean over 30 draws; best detector of each row in bold). Exploratory analysis.

| Injected pattern | *f* | Reciprocity | Self-citation ratio | Combined | Null-model excess | DOMINANT | Graph autoencoder | Isolation Forest |
|---|---|---|---|---|---|---|---|---|
| Reciprocal cartel | 0.25 | **0.870** | 0.455 | 0.811 | 0.765 | 0.621 | 0.531 | 0.599 |
|  | 0.50 | **0.904** | 0.419 | 0.863 | 0.870 | 0.717 | 0.537 | 0.685 |
| Self-citation | 0.25 | 0.480 | **0.985** | 0.967 | 0.828 | 0.653 | 0.505 | 0.611 |
|  | 0.50 | 0.464 | **0.994** | 0.986 | 0.913 | 0.845 | 0.531 | 0.721 |
| Budget-preserving cartel | 0.25 | **0.923** | 0.504 | 0.877 | 0.778 | 0.669 | 0.517 | 0.644 |
|  | 0.50 | **0.958** | 0.504 | 0.934 | 0.904 | 0.794 | 0.430 | 0.753 |
| Large cartel | 0.25 | **0.881** | 0.456 | 0.828 | 0.718 | 0.611 | 0.524 | 0.578 |
|  | 0.50 | **0.905** | 0.424 | 0.864 | 0.807 | 0.682 | 0.516 | 0.639 |
| Camouflaged cartel | 0.25 | **0.792** | 0.427 | 0.722 | 0.738 | 0.632 | 0.628 | 0.589 |
|  | 0.50 | 0.804 | 0.366 | 0.744 | **0.840** | 0.706 | 0.638 | 0.636 |
| One-directional ring | 0.25 | 0.415 | 0.463 | 0.429 | **0.878** | 0.625 | 0.555 | 0.568 |
|  | 0.50 | 0.355 | 0.416 | 0.363 | **0.942** | 0.730 | 0.591 | 0.658 |
| Citation stacking | 0.25 | 0.485 | 0.481 | 0.471 | **0.799** | 0.564 | 0.523 | 0.539 |
|  | 0.50 | 0.457 | 0.463 | 0.444 | **0.899** | 0.660 | 0.529 | 0.611 |

Abbreviations: *f*: injected share of the outgoing citations of a venue; Combined: the larger of the two heuristics, each expressed as a percentile rank.
