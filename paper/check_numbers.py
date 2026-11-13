"""Check that every number in the paper comes from a file written by the pipeline.

``.venv/bin/python paper/check_numbers.py [paper/axis1.md | paper/response_to_reviewers.tex]``

The response letter (a ``.tex`` file) is checked in the same way after its LaTeX markup has been reduced to text;
the manuscript itself is then one of the sources, since the letter quotes it.

Sources: results/summary.md, results/*.json, results/exp2_lab_predictions.csv, data/processed/full/stats.json,
the configuration and code (hyperparameters), the open-corpus report and the verified references; for the revision,
results/rev1 (statistics, generated tables, selected hyperparameters) and the revision code. Ignored:
citation brackets, table/figure/section/algorithm/equation numbers, years, integers up to 10, and the formulas
(display and inline math, the algorithm block and the nomenclature table), which hold symbols rather than results.
Every other number that is not found is printed for manual review.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCES = [ROOT / "results" / "summary.md", ROOT / "results" / "exp2_lab_predictions.csv",
           ROOT / "data" / "processed" / "full" / "stats.json", ROOT / "config.py", ROOT / "models.py",
           ROOT / "experiments.py", ROOT / "build_graph.py", ROOT / "collect.py",
           ROOT.parent / "DataOpenCorpus" / "report.md", HERE / "references_check.json"]
SOURCES += sorted((ROOT / "results").glob("*.json"))
# revision: statistics, generated tables, selected hyperparameters, registration of the holdout, revision code
REV = ROOT / "results" / "rev1"
SOURCES += sorted(REV.glob("*.json")) + sorted((REV / "stats").glob("*.json")) + sorted((REV / "tables").glob("*"))
SOURCES += [REV / "hpo" / "best.json", ROOT / "data" / "processed" / "full" / "rev1" / "stats.json",
            ROOT / "data" / "processed" / "full" / "rev1_firsthop" / "stats.json",
            ROOT / "data" / "raw" / "full" / "collect_report.json", ROOT / "prereg" / "frozen.json",
            ROOT / "grids_rev1.py", ROOT / "experiments_rev1.py", ROOT / "protocols.py", ROOT / "collect_rev1.py",
            ROOT / "build_rev1.py", ROOT / "anomaly_rev1.py", ROOT / "stats.py", ROOT / "external" / "README.md"]

NUM = re.compile(r"(?<![\w.])[−-]?\d[\d,]*(?:\.\d+)?(?![\w])")


def corpus_numbers() -> set[str]:
    out: set[str] = set()
    for p in SOURCES:
        if not p.exists():
            continue
        text = p.read_text()
        for m in NUM.finditer(text):
            v = m.group(0).replace(",", "").replace("−", "-").lstrip("+")
            out.add(v.lstrip("-"))
            try:
                f = float(v)
                for nd in (1, 2, 3):
                    out.add(f"{abs(f):.{nd}f}")
            except ValueError:
                pass
    return out


def tex_to_text(tex: str) -> str:
    """The letter: preamble dropped (line numbers kept), macros and math reduced to the numbers they print."""
    head, _, body = tex.partition(r"\begin{document}")
    body = "\n" * head.count("\n") + body
    body = re.sub(r"\\ci\{([^{}]*)\}\{([^{}]*)\}\{([^{}]*)\}", r"\1 [\2, \3]", body)
    body = re.sub(r"\\(ref|label|needspace|externaldocument)\{[^{}]*\}", " ", body)
    body = re.sub(r"IJASCA-\d+", " ", body)                                       # manuscript identifier
    body = body.replace(r"\,\%", " %").replace(r"\%", "%").replace("{,}", ",").replace("~", " ")
    body = body.replace(r"\pm", "±").replace(r"\leq", "≤").replace("$", " ")
    return re.sub(r"(point|points|Report|Reports|Eq\.|Eqs\.|Editorial point) (\d+)", r"\1", body)


def main() -> None:
    paper = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "axis1.md"
    text = paper.read_text()
    if paper.suffix == ".tex":
        SOURCES.append(HERE / "axis1.md")
        text = tex_to_text(text)
    known = corpus_numbers()
    missing = []
    in_code = in_nomenclature = False  # the algorithm and the nomenclature hold symbols, not results
    for ln, line in enumerate(text.splitlines(), 1):
        if line.startswith("```"):
            in_code = not in_code
            continue
        if line == "**Nomenclature**":
            in_nomenclature = True
            continue
        if in_nomenclature and (line.startswith("|") or not line.strip()):
            continue
        in_nomenclature = False
        if in_code or line.startswith(("#", "![", "<!--", "$$", "%")):
            continue
        clean = re.sub(r"\$[^$]+\$", " ", line)                                   # inline math
        clean = re.sub(r"\[[\d,\s–-]+\]", " ", clean)                             # citations [3, 7]
        clean = re.sub(r"Eqs?\.\s*\(\d+\)", " ", clean)                           # Eq. (3)
        clean = re.sub(r"(Table|Figure|Section|Sections|Algorithm|Fig\.)\s+\d+(\.\d+)?", " ", clean)
        clean = re.sub(r"10\.\d{4,}/\S+", " ", clean)                             # DOIs
        clean = re.sub(r"\b(19|20)\d{2}\b(–(19|20)?\d{2})?", " ", clean)          # years and year ranges
        for m in NUM.finditer(clean):
            raw = m.group(0)
            v = raw.replace(",", "").replace("−", "-").lstrip("-")
            try:
                f = float(v)
            except ValueError:
                continue
            if "." not in v and f <= 10:
                continue
            after = clean[m.end():m.end() + 2]
            cands = {v}
            if "%" in after:  # 42.1 % <- 0.421
                cands |= {f"{f / 100:.3f}", f"{f / 100:.2f}", f"{f / 100:.4f}".rstrip("0")}
            if not cands & known:
                missing.append((ln, raw, line.strip()[:110]))
    for ln, raw, ctx in missing:
        print(f"line {ln}: {raw}  | {ctx}")
    print(f"{len(missing)} number(s) not found in the pipeline outputs")


if __name__ == "__main__":
    main()
