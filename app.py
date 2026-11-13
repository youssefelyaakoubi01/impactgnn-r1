"""Streamlit interface: run each step of the pipeline and display the results.

``.venv/bin/streamlit run app.py``
"""

from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd
import streamlit as st

from config import ROOT, get_config

st.set_page_config(page_title="Axis 1 – citation GNNs", layout="wide")
st.title("Axis 1 – Citation graphs and GNNs")

profile = st.sidebar.selectbox("Profile", ["small", "full"], index=0)
cfg = get_config(profile)
page = st.sidebar.radio("Step", ["Overview", "1. Collect", "2. Build graph", "3. Experiments", "4. Summary",
                                 "Lab predictions"])


def run(cmd: list[str]) -> None:
    """Run a pipeline step as a subprocess and stream its log."""
    box = st.empty()
    lines: list[str] = []
    proc = subprocess.Popen([sys.executable, *cmd, "--profile", profile], cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        if "Warning" in line or "warnings.warn" in line:
            continue
        lines.append(line.rstrip())
        box.code("\n".join(lines[-40:]))
    proc.wait()
    (st.success if proc.returncode == 0 else st.error)(f"exit code {proc.returncode}")


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


if page == "Overview":
    st.markdown(
        "Five sub-topics on our data: (1) GAT impact prediction and attention, (2) inductive cold start with "
        "GraphSAGE, (3) heterogeneous graph with HAN/HGT, (4) cartel and self-citation detection, (5) text vs "
        "structure. Every number displayed comes from files written by the pipeline.")
    env = read_json(cfg.results_dir / "env.json")
    if env:
        st.subheader("Environment")
        st.json(env)
    stats = read_json(cfg.processed_dir / "stats.json")
    if stats:
        st.subheader("Graph statistics")
        st.json(stats, expanded=False)

elif page == "1. Collect":
    steps = st.multiselect("Steps", ["seeds", "topics", "sample", "refs", "nodes", "authors", "venues"],
                           default=["seeds", "topics", "sample", "refs", "nodes", "authors", "venues"])
    if st.button("Run collection"):
        run(["collect.py", "--step", ",".join(steps)])
    rep = read_json(cfg.raw_dir / "collect_report.json")
    if rep:
        st.subheader("Requests and credits per step")
        st.dataframe(pd.DataFrame(rep["steps"]).T)
        st.caption(f"Credits remaining at last run: {rep.get('credits_remaining_at_end')}")
    topics = read_json(cfg.raw_dir / "topics.json")
    if topics:
        st.subheader("Topics of the labeled sample")
        st.dataframe(pd.DataFrame(topics))

elif page == "2. Build graph":
    if st.button("Build embeddings, features and graphs (GPU)"):
        run(["build_graph.py"])
    stats = read_json(cfg.processed_dir / "stats.json")
    if stats:
        c = st.columns(4)
        c[0].metric("paper nodes", stats["nodes"])
        c[1].metric("citation edges", stats["edges"])
        c[2].metric("labeled k=3", stats["labels_k3"]["n"])
        c[3].metric("labeled k=5", stats["labels_k5"]["n"])
        st.json(stats, expanded=False)

elif page == "3. Experiments":
    exps = st.multiselect("Experiments", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
    jobs = st.number_input("Parallel GPU workers", 1, 8, cfg.jobs)
    if st.button("Run experiments"):
        run(["experiments.py", "--exp", ",".join(map(str, exps)), "--jobs", str(int(jobs))])
    for e in exps:
        r = read_json(cfg.results_dir / f"exp{e}.json")
        if not r:
            continue
        st.subheader(f"exp{e} ({r.get('wall_time_s')} s on {r.get('gpu') or r.get('device')})")
        for key, s in r["summary"].items():
            rows = {m: {k: f"{v['mean']:.3f} ± {v['std']:.3f}" for k, v in mv.items()} for m, mv in s.items()}
            st.markdown(f"**{key}**")
            st.dataframe(pd.DataFrame(rows).T)

elif page == "4. Summary":
    if st.button("Regenerate summary.md"):
        run(["summarize.py"])
    p = cfg.results_dir / "summary.md"
    if p.exists():
        st.markdown(p.read_text())

elif page == "Lab predictions":
    p = cfg.results_dir / "exp2_lab_predictions.csv"
    if p.exists():
        st.markdown("Predicted citations of the lab papers by the inductive cold-start models (mean over seeds). "
                    "Papers published outside the labeled years are extrapolations.")
        st.dataframe(pd.read_csv(p))
    else:
        st.info("Run experiment 2 first.")
