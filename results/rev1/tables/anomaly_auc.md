**Table @anomaly_auc.** ROC-AUC of each detector for the injected venues against all other venues (mean over 30 draws; the largest standard deviation of a cell is 0.203; best detector of each row in bold). Learned detectors use content and structural features. Primary analysis.

| Injected pattern | *f* | Reciprocity | Self-citation ratio | Combined | Null-model excess | DOMINANT | Graph autoencoder | Isolation Forest |
|---|---|---|---|---|---|---|---|---|
| Reciprocal cartel | 0.10 | **0.812** | 0.497 | 0.720 | 0.672 | 0.512 | 0.480 | 0.513 |
|  | 0.25 | **0.890** | 0.484 | 0.820 | 0.816 | 0.577 | 0.476 | 0.575 |
|  | 0.50 | **0.930** | 0.464 | 0.880 | 0.919 | 0.658 | 0.450 | 0.649 |
| Self-citation | 0.10 | 0.487 | **0.908** | 0.844 | 0.611 | 0.526 | 0.513 | 0.527 |
|  | 0.25 | 0.487 | **0.980** | 0.960 | 0.809 | 0.645 | 0.500 | 0.602 |
|  | 0.50 | 0.487 | **0.994** | 0.986 | 0.918 | 0.816 | 0.493 | 0.686 |
| Budget-preserving cartel | 0.10 | **0.832** | 0.503 | 0.745 | 0.596 | 0.561 | 0.543 | 0.546 |
|  | 0.25 | **0.927** | 0.503 | 0.878 | 0.761 | 0.668 | 0.512 | 0.642 |
|  | 0.50 | **0.961** | 0.503 | 0.936 | 0.882 | 0.794 | 0.427 | 0.752 |
| Large cartel | 0.10 | **0.884** | 0.486 | 0.814 | 0.620 | 0.560 | 0.482 | 0.543 |
|  | 0.25 | **0.897** | 0.478 | 0.830 | 0.775 | 0.580 | 0.477 | 0.563 |
|  | 0.50 | **0.926** | 0.461 | 0.874 | 0.879 | 0.625 | 0.434 | 0.608 |
| Camouflaged cartel | 0.10 | **0.774** | 0.482 | 0.672 | 0.658 | 0.536 | 0.574 | 0.524 |
|  | 0.25 | **0.836** | 0.459 | 0.744 | 0.807 | 0.580 | 0.565 | 0.563 |
|  | 0.50 | 0.865 | 0.429 | 0.784 | **0.911** | 0.629 | 0.521 | 0.608 |
| One-directional ring | 0.10 | 0.470 | 0.489 | 0.473 | **0.726** | 0.544 | 0.524 | 0.513 |
|  | 0.25 | 0.450 | 0.475 | 0.445 | **0.897** | 0.598 | 0.522 | 0.560 |
|  | 0.50 | 0.426 | 0.456 | 0.409 | **0.964** | 0.687 | 0.532 | 0.635 |
| Citation stacking | 0.10 | 0.519 | 0.502 | 0.500 | **0.647** | 0.495 | 0.493 | 0.490 |
|  | 0.25 | 0.507 | 0.495 | 0.485 | **0.823** | 0.538 | 0.499 | 0.528 |
|  | 0.50 | 0.493 | 0.485 | 0.465 | **0.923** | 0.632 | 0.488 | 0.601 |

Abbreviations: *f*: injected share of the outgoing citations of a venue; Combined: the larger of the two heuristics, each expressed as a percentile rank.
