# External data

`jcr_suppressed.csv`: journals suppressed from the Journal Citation Reports for anomalous citation patterns
(excessive self-citation or citation stacking), JCR releases 2018-2026. Titles, years and reasons come from
Clarivate's public "Title Suppressions" lists and from the reports listed in `source_url`; ISSNs were looked up in
Crossref or OpenAlex (`issn_source`). `jcr_year` is the release year. Rows whose `reason` contains "reversed" were
listed at release and later withdrawn. A suppression is an editorial decision of the index, not a finding of
misconduct. Used only by `anomaly_rev1.py --external` as a descriptive check.
