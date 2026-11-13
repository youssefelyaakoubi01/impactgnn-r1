"""Archive of the revision: everything needed to check or repeat the results without the OpenAlex API.

``python make_release.py [--with-data] [--out release] [--public DIR]``

The archive is a directory (``release/impactgnn-r1/``) with a checksum file:

``code.tar.gz``         the tracked files of the current commit (code, tests, figure and table scripts, result files)
``environment/``        pinned packages with the wheel indexes, Python / PyTorch / CUDA / GPU versions
``identifiers/``        OpenAlex identifiers of every paper node, with its year, topic, role, labels and the graph
                        variants it belongs to; identifiers of the fresh holdout sample
``splits/``             training, validation and test identifiers of every partition used by a run
``runs/``               one line per run: configuration, seeds, hyperparameters and metrics
``predictions.tar``     per-paper predictions of every run (test and validation)
``checkpoints.tar``     the registered models of the confirmatory holdout, with their scalers and projections
``results/``            search of hyperparameters, anomaly results, holdout result, statistics and generated tables
``prereg/``             registration file of the confirmatory holdout
``openalex_data.tar``   (``--with-data``) the downloaded OpenAlex records, so that the graphs can be rebuilt from the
                        same snapshot; OpenAlex data are CC0

Withheld, in line with the ethical considerations of the article: the names of the venues that an anomaly detector
ranked highest on the real graph, and the predictions for the laboratory papers of the case study (the table and the
listings of the initial pipeline, and the prediction files of the grid ``deploy``). The four laboratory papers with
a complete citation window are in ``results/rev1/tables/lab.md``, as in the article.

``--public DIR`` writes the same content unpacked, as the tree of a public repository: the code at the root, the
predictions and the saved models at their place under ``results/rev1/``, and ``archive/`` with the identifiers, the
partitions and the environment.

Nothing is uploaded: the directory is meant to be deposited by the authors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import re
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import ROOT, get_config

RES = ROOT / "results" / "rev1"
SPLIT_KEYS = ["protocol", "T", "k", "rep", "fold", "split_seed", "seed", "test_group", "train_subsample"]
VARIANTS = ["submitted", "uncapped", "cap30k", "cap45k", "cap60k", "cut2022", "cut2022lso", "cut2022sym", "cut2021sym"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tar_dir(src: Path, dst: Path, arcname: str) -> None:
    with tarfile.open(dst, "w") as tar:          # the files are already compressed (npz, pt, parquet)
        tar.add(src, arcname=arcname)


def environment(out: Path) -> None:
    import torch
    import torch_geometric

    out.mkdir(parents=True, exist_ok=True)
    req = (ROOT / "requirements.txt").read_text()
    (out / "requirements.txt").write_text(req)
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "torch_geometric": torch_geometric.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
            "tags": subprocess.run(["git", "tag", "--points-at", "HEAD"], cwd=ROOT, capture_output=True,
                                   text=True).stdout.split()}
    (out / "environment.json").write_text(json.dumps(info, indent=1))


def identifiers(out: Path, cfg) -> dict:
    import torch

    from graph_variants import node_mask

    out.mkdir(parents=True, exist_ok=True)
    g = torch.load(cfg.processed_dir / "rev1" / "graph.pt", weights_only=False)
    df = pd.DataFrame({"openalex_id": list(g["ids"]), "year": g["year"].numpy(), "role": list(g["origin"]),
                       "labelled_k3": g["m3"].bool().numpy(), "labelled_k5": g["m5"].bool().numpy(),
                       "citations_k3": g["c3"].numpy(), "citations_k5": g["c5"].numpy()})
    for v in VARIANTS:
        df[f"in_{v}"] = node_mask(g, v).cpu().numpy()
    import stats as S

    nt = S.node_table()
    for c in ("topic", "field", "subfield", "topic_group"):
        df[c] = nt[c].reindex(df.openalex_id).to_numpy()
    df.to_parquet(out / "nodes.parquet", index=False)
    hold = cfg.processed_dir / "rev1_holdout" / "graph.pt"
    if hold.exists():
        h = torch.load(hold, weights_only=False)
        fresh = np.asarray(h["origin"]) == "fresh"
        pd.DataFrame({"openalex_id": np.asarray(h["ids"])[fresh], "year": h["year"].numpy()[fresh]}).to_parquet(
            out / "fresh_holdout.parquet", index=False)
    return g


def splits(out: Path, g: dict, cfg) -> None:
    from experiments_rev1 import read_runs
    from protocols import make_split

    out.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(g["ids"])
    seen, rows = {}, []
    for f in sorted((RES / "runs").glob("*.jsonl")):
        if f.stem in ("hpo", "holdout", "kfold_firsthop", "mature_firsthop"):   # same partitions as the other grids
            continue
        for r in read_runs(f.stem):
            job = r["job"]
            key = json.dumps({k: job.get(k) for k in SPLIT_KEYS}, sort_keys=True)
            if key in seen:
                continue
            s = make_split(g, job, cfg)
            seen[key] = len(seen)
            for part in ("tr", "va", "te"):
                rows.append({"split": seen[key], **{k: job.get(k) for k in SPLIT_KEYS},
                             "part": {"tr": "train", "va": "validation", "te": "test"}[part],
                             "n": int(len(s[part])), "openalex_ids": " ".join(ids[np.asarray(s[part])])})
    pd.DataFrame(rows).to_parquet(out / "splits.parquet", index=False)
    print(f"splits: {len(seen)} partitions")


# the manuscript, the correspondence with the journal, the run logs, the working notes and the licensed Scopus
# export stay out; "glob" keeps * inside one directory, so the vector figures in paper/figures are archived
SKIP = ["logs/**", "docs/**", "paper/scopus/**", "paper/response_to_reviewers.*", "paper/*.pdf", "paper/paper*.tex",
        "paper/supplement*.tex", "paper/axis1.*"]
WITHHELD_SECTIONS = ("### Lab papers", "### Real graph: venues ranked highest")


def redact(tree: Path) -> None:
    """Remove from the result files of the initial pipeline what the article undertakes not to publish: the names
    of the venues ranked highest by a detector on the real graph and the predictions for the laboratory papers."""
    for d in (tree / "results", tree / "results" / "small"):
        (d / "exp2_lab_predictions.csv").unlink(missing_ok=True)
        for name, key in (("exp2.json", "lab_predictions"), ("exp4.json", "cases_for_inspection")):
            f = d / name
            if f.exists():
                r = json.loads(f.read_text())
                if r.pop(key, None) is not None:
                    r["withheld"] = f"{key}: not published (see the ethical considerations of the article)"
                    f.write_text(json.dumps(r, indent=1))
        f = d / "summary.md"
        if f.exists():
            out, drop = [], False
            for line in f.read_text().splitlines():
                if re.match(r"#{2,3} ", line):
                    drop = line.startswith(WITHHELD_SECTIONS)
                    if drop:
                        out += [line, "", "Not published (see the ethical considerations of the article).", ""]
                if not drop:
                    out.append(line)
            f.write_text("\n".join(out) + "\n")


def export_code(dst: Path) -> None:
    """Tracked files of the current commit, without the files of SKIP, extracted to ``dst`` and redacted."""
    with tempfile.TemporaryDirectory() as tmp:
        tar = Path(tmp) / "code.tar"
        subprocess.run(["git", "archive", "--format=tar", "-o", str(tar), "HEAD", "--", "."]
                       + [f":(exclude,glob){x}" for x in SKIP], cwd=ROOT, check=True)
        dst.mkdir(parents=True)
        with tarfile.open(tar) as t:
            t.extractall(dst, filter="data")
    redact(dst)


def kept_predictions() -> list[Path]:
    """Prediction files of the runs that are in the run records. Files left by earlier trial runs and the
    predictions for the laboratory papers (grid ``deploy``) stay out."""
    from experiments_rev1 import read_runs

    ids = {r["id"] for f in sorted((RES / "runs").glob("*.jsonl")) if f.stem != "deploy" for r in read_runs(f.stem)}
    lab = {r["id"] for r in read_runs("deploy")}
    files = sorted((RES / "preds").glob("*.npz"))
    keep = [p for p in files if p.stem in ids]
    n_lab = sum(p.stem in lab for p in files)
    print(f"predictions: {len(keep)} files kept, {n_lab} of the laboratory case study withheld, "
          f"{len(files) - len(keep) - n_lab} without a run record left out")
    return keep


def public_tree(out: Path, code: Path, release: Path, keep: list[Path]) -> None:
    """The archive unpacked, as the tree of a public repository."""
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(code, out)
    (out / "results" / "rev1" / "preds").mkdir(parents=True)
    for p in keep:
        shutil.copy(p, out / "results" / "rev1" / "preds" / p.name)
    shutil.copytree(RES / "checkpoints", out / "results" / "rev1" / "checkpoints")
    for name in ("identifiers", "splits", "environment"):
        shutil.copytree(release / name, out / "archive" / name)
    ignore = out / ".gitignore"          # the predictions and the saved models are part of the public tree
    ignore.write_text("\n".join(ln for ln in ignore.read_text().splitlines()
                                if ln.strip() not in ("results/rev1/preds/", "results/rev1/checkpoints/")) + "\n")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (out / "ARCHIVE.md").write_text(
        "# Archive of the revision\n\n"
        f"Snapshot of the development repository at commit `{commit}`, without its history; the tags and branches "
        "named in `README.md` belong to that repository.\n\n"
        "| Path | Content |\n|---|---|\n"
        "| repository root | code, tests, table and figure scripts, result files (`results/`), registration of the "
        "confirmatory holdout (`prereg/`) |\n"
        "| `results/rev1/runs/` | one line per run: configuration, seeds, hyperparameters and metrics |\n"
        "| `results/rev1/preds/` | per-paper predictions of every run (test and validation), one file per run "
        "identifier |\n"
        "| `results/rev1/checkpoints/` | the registered models of the confirmatory holdout, with their scalers and "
        "projections |\n"
        "| `archive/identifiers/` | OpenAlex identifiers of every paper node, with its year, topic, role, labels and "
        "the graph variants it belongs to; identifiers of the fresh holdout sample |\n"
        "| `archive/splits/` | training, validation and test identifiers of every partition used by a run |\n"
        "| `archive/environment/` | pinned packages with the wheel indexes, Python / PyTorch / CUDA / GPU versions |\n"
        "| `SHA256SUMS` | checksum of every file |\n\n"
        "Not included: the downloaded OpenAlex records (CC0; the collection scripts download them again), the "
        "manuscript and the Scopus export used for the related work.\n\n"
        "Withheld, in line with the ethical considerations of the article: the names of the venues that an anomaly "
        "detector ranked highest on the real graph, and the predictions for the laboratory papers of the case study "
        "(the listings of the initial pipeline and the prediction files of the grid `deploy`, so `make_tables.py lab` "
        "cannot be run from this tree). The four laboratory papers with a complete citation window are in "
        "`results/rev1/tables/lab.md`, as in the article.\n")
    sums = [f"{sha256(p)}  {p.relative_to(out)}" for p in sorted(out.rglob("*")) if p.is_file()]
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 2**20
    print(f"public tree written to {out} ({len(sums)} files, {size:.0f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="release")
    ap.add_argument("--with-data", action="store_true", help="include the downloaded OpenAlex records")
    ap.add_argument("--public", default=None, help="also write the archive unpacked, as a public repository tree")
    args = ap.parse_args()
    cfg = get_config("full")
    out = ROOT / args.out / "impactgnn-r1"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    with tempfile.TemporaryDirectory() as tmp:
        code = Path(tmp) / "impactgnn"
        export_code(code)
        with tarfile.open(out / "code.tar.gz", "w:gz") as tar:
            tar.add(code, arcname="impactgnn")
        environment(out / "environment")
        g = identifiers(out / "identifiers", cfg)
        splits(out / "splits", g, cfg)
        shutil.copytree(RES / "runs", out / "runs")
        (out / "results").mkdir()
        for name in ("hpo", "stats", "tables"):
            shutil.copytree(RES / name, out / "results" / name)
        for name in ("anomaly.json", "anomaly_runs.jsonl", "anomaly_external.json", "holdout.json"):
            if (RES / name).exists():
                shutil.copy(RES / name, out / "results" / name)
        shutil.copytree(ROOT / "prereg", out / "prereg")
        keep = kept_predictions()
        with tarfile.open(out / "predictions.tar", "w") as tar:
            for p in keep:
                tar.add(p, arcname=f"preds/{p.name}")
        tar_dir(RES / "checkpoints", out / "checkpoints.tar", "checkpoints")
        if args.with_data:
            tar_dir(cfg.raw_dir, out / "openalex_data.tar", "data/raw/full")
        (out / "README.md").write_text(__doc__.split("\n\n", 1)[1].replace("``", "`"))
        sums = [f"{sha256(p)}  {p.relative_to(out)}" for p in sorted(out.rglob("*")) if p.is_file()]
        (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
        size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 2**20
        print(f"release written to {out} ({len(sums)} files, {size:.0f} MB)")
        if args.public:
            public_tree(Path(args.public).resolve(), code, out, keep)


if __name__ == "__main__":
    main()
