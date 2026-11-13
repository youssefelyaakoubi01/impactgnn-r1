**Table @anomaly_ap.** Average precision of each detector at each injection intensity; the prevalence is the average precision of a random ranking (mean over 30 draws; the largest standard deviation of a cell is 0.126; best detector of each row in bold). Learned detectors use content and structural features. Primary analysis.

| Injected pattern | Prevalence | *f* | Reciprocity | Self-citation ratio | Combined | Null-model excess | DOMINANT | Graph autoencoder | Isolation Forest |
|---|---|---|---|---|---|---|---|---|---|
| Reciprocal cartel | 0.0064 | 0.10 | **0.023** | 0.008 | 0.016 | 0.015 | 0.010 | 0.009 | 0.008 |
|  |  | 0.25 | 0.035 | 0.007 | 0.021 | **0.043** | 0.016 | 0.010 | 0.012 |
|  |  | 0.50 | 0.061 | 0.007 | 0.034 | **0.142** | 0.027 | 0.010 | 0.017 |
| Self-citation | 0.0032 | 0.10 | 0.005 | **0.020** | 0.012 | 0.009 | 0.005 | 0.006 | 0.011 |
|  |  | 0.25 | 0.005 | **0.076** | 0.041 | 0.034 | 0.007 | 0.005 | 0.012 |
|  |  | 0.50 | 0.005 | **0.204** | 0.105 | 0.097 | 0.012 | 0.005 | 0.016 |
| Budget-preserving cartel | 0.0064 | 0.10 | **0.044** | 0.008 | 0.035 | 0.013 | 0.013 | 0.012 | 0.011 |
|  |  | 0.25 | **0.131** | 0.008 | 0.093 | 0.041 | 0.026 | 0.010 | 0.021 |
|  |  | 0.50 | **0.409** | 0.008 | 0.224 | 0.132 | 0.040 | 0.010 | 0.043 |
| Large cartel | 0.0077 | 0.10 | **0.050** | 0.008 | 0.031 | 0.014 | 0.015 | 0.016 | 0.011 |
|  |  | 0.25 | **0.051** | 0.008 | 0.032 | 0.024 | 0.018 | 0.016 | 0.014 |
|  |  | 0.50 | **0.071** | 0.008 | 0.042 | 0.051 | 0.021 | 0.012 | 0.018 |
| Camouflaged cartel | 0.0064 | 0.10 | 0.015 | 0.007 | 0.010 | **0.019** | 0.014 | 0.017 | 0.010 |
|  |  | 0.25 | 0.019 | 0.006 | 0.012 | **0.050** | 0.019 | 0.015 | 0.012 |
|  |  | 0.50 | 0.024 | 0.006 | 0.015 | **0.125** | 0.022 | 0.014 | 0.016 |
| One-directional ring | 0.0064 | 0.10 | 0.007 | 0.008 | 0.007 | **0.024** | 0.016 | 0.010 | 0.009 |
|  |  | 0.25 | 0.006 | 0.007 | 0.006 | **0.125** | 0.027 | 0.013 | 0.012 |
|  |  | 0.50 | 0.006 | 0.006 | 0.005 | **0.313** | 0.045 | 0.015 | 0.016 |
| Citation stacking | 0.0064 | 0.10 | 0.009 | 0.008 | 0.008 | **0.017** | 0.010 | 0.009 | 0.008 |
|  |  | 0.25 | 0.008 | 0.008 | 0.007 | **0.069** | 0.021 | 0.011 | 0.010 |
|  |  | 0.50 | 0.008 | 0.007 | 0.007 | **0.166** | 0.046 | 0.011 | 0.015 |

Abbreviations: *f*: injected share of the outgoing citations of a venue; Combined: the larger of the two heuristics, each expressed as a percentile rank.
