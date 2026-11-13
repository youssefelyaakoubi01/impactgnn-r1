"""Verify every reference of the paper against Crossref (DOI) or arXiv (identifier) and write
``references.bib`` from the fetched metadata (never from memory) plus ``references_check.json``.

A few entries carry an override (fourth tuple element) for what the fetched record lacks: the peer-reviewed venue
of an arXiv paper (as stated in its arXiv comment, DBLP or the proceedings), the book and volume of a chapter whose
Crossref record only names the series, an article number, or the page range of the ACL Anthology. Title and
authors always come from the fetched record; the overridden fields are listed under ``overrides``.

``.venv/bin/python paper/verify_refs.py``
"""

from __future__ import annotations

import html
import json
import re
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from config import MAILTO  # noqa: E402

# key -> DOI or arXiv id; the expected title fragment guards against a wrong identifier
REFS = {
    # impact, citation prediction, research evaluation
    "garfield1972": ("doi", "10.1126/science.178.4060.471", "citation analysis as a tool in journal evaluation"),
    "hirsch2005": ("doi", "10.1073/pnas.0507655102", "an index to quantify an individual"),
    "hicks2015": ("doi", "10.1038/520429a", "leiden manifesto"),
    "wang2013": ("doi", "10.1126/science.1237825", "quantifying long-term scientific impact"),
    "sinatra2016": ("doi", "10.1126/science.aaf5239", "quantifying the evolution of individual scientific impact"),
    "fortunato2018": ("doi", "10.1126/science.aao0185", "science of science"),
    "yan2011": ("doi", "10.1145/2063576.2063757", "citation count prediction", {"subtitle": True}),
    "stegehuis2015": ("doi", "10.1016/j.joi.2015.06.005", "predicting the long-term citation impact of recent publications"),
    "xia2023": ("doi", "10.1007/s11192-022-04547-8", "a review of scientific impact prediction"),
    "merton1968": ("doi", "10.1126/science.159.3810.56", "the matthew effect in science"),
    "abrishami2019": ("doi", "10.1016/j.joi.2019.02.011", "predicting citation counts"),
    "holm2020": ("arxiv", "2012.05742", "longitudinal citation prediction"),
    "kapoor2023": ("doi", "10.1016/j.patter.2023.100804", "leakage and the reproducibility crisis"),
    # graph-based impact prediction (screened from the Scopus export in scopus/)
    "li2019biblio": ("doi", "10.1109/SMC.2019.8913961", "citation count prediction with large-scale biblio-features"),
    "cummings2020": ("doi", "10.1109/ICASSP40776.2020.9054769", "structured citation trend prediction"),
    "jiang2021hints": ("doi", "10.1145/3442381.3450107", "hints"),
    "kleminski2021": ("doi", "10.1016/j.joi.2021.101200", "where should i publish"),
    "xu2022sihdgnn": ("doi", "10.1016/j.knosys.2021.107839", "heterogeneous dynamical academic network"),
    "xiao2022esmr": ("doi", "10.1007/s10489-021-02468-2", "multivariate random-walk model"),
    "geng2022dgni": ("doi", "10.1145/3511808.3557398", "dynamic heterogeneous graph and node importance"),
    "ochi2022": ("doi", "10.1371/journal.pone.0274253", "multiple future scientific impacts"),
    "wang2023agsta": ("doi", "10.1007/s11192-022-04541-0", "agsta-net"),
    "he2023h2cgl": ("doi", "10.1016/j.ipm.2023.103512", "modeling dynamics of citation network"),
    "yang2023cahgn": ("doi", "10.1109/ICDE55515.2023.00058", "revisiting citation prediction"),
    "ochi2023": ("doi", "10.1007/978-3-031-43088-6_7", "predicting top-cited papers",
                 {"venue": "Web Information Systems and Technologies. Lecture Notes in Business Information Processing, vol 494"}),
    "xue2024dppdcc": ("doi", "10.1145/3627673.3679546", "diffusion, conformity, and contribution"),
    "qin2024mffggcn": ("doi", "10.1109/MLNLP63328.2024.10800652", "gated graph convolutional neural network"),
    "luo2025": ("doi", "10.1177/01655515231191213", "disclose its citation impact"),
    "zhang2025cdchgn": ("doi", "10.1016/j.knosys.2025.114282", "enhanced temporal graph neural network"),
    "lu2026newborn": ("doi", "10.1145/3774904.3792702", "bias-aware citation prediction"),
    "li2026collab": ("doi", "10.1016/j.ipm.2026.104634", "collaboration networks predict citation impact"),
    "zhao2025words": ("doi", "10.1609/aaai.v39i1.32106", "newborn article impact prediction"),
    # graph neural networks
    "kipf2017": ("arxiv", "1609.02907", "semi-supervised classification with graph convolutional networks",
                 {"venue": "5th International Conference on Learning Representations (ICLR 2017)", "year": 2017}),
    "velickovic2018": ("arxiv", "1710.10903", "graph attention networks",
                       {"venue": "6th International Conference on Learning Representations (ICLR 2018)", "year": 2018}),
    "brody2022": ("arxiv", "2105.14491", "how attentive are graph attention networks",
                  {"venue": "10th International Conference on Learning Representations (ICLR 2022)", "year": 2022}),
    "hamilton2017": ("arxiv", "1706.02216", "inductive representation learning on large graphs",
                     {"venue": "Advances in Neural Information Processing Systems 30 (NIPS 2017)", "year": 2017, "pages": "1024-1034"}),
    "schlichtkrull2018": ("doi", "10.1007/978-3-319-93417-4_38", "modeling relational data with graph convolutional networks",
                          {"venue": "The Semantic Web. ESWC 2018. Lecture Notes in Computer Science, vol 10843"}),
    "wang2019han": ("doi", "10.1145/3308558.3313562", "heterogeneous graph attention network"),
    "hu2020hgt": ("doi", "10.1145/3366423.3380027", "heterogeneous graph transformer"),
    "hu2020ogb": ("arxiv", "2005.00687", "open graph benchmark",
                  {"venue": "Advances in Neural Information Processing Systems 33 (NeurIPS 2020)", "year": 2020, "pages": "22118-22133"}),
    "fey2019": ("arxiv", "1903.02428", "fast graph representation learning with pytorch geometric",
                {"venue": "ICLR 2019 Workshop on Representation Learning on Graphs and Manifolds", "year": 2019}),
    "shchur2018": ("arxiv", "1811.05868", "pitfalls of graph neural network evaluation",
                   {"venue": "Relational Representation Learning Workshop (R2L 2018), NeurIPS 2018", "year": 2018}),
    "huang2021cs": ("arxiv", "2010.13993", "combining label propagation and simple models",
                    {"venue": "9th International Conference on Learning Representations (ICLR 2021)", "year": 2021}),
    "jain2019": ("doi", "10.18653/v1/N19-1357", "attention is not explanation"),
    "wiegreffe2019": ("doi", "10.18653/v1/D19-1002", "attention is not not explanation"),
    # graph anomaly detection, cartels, self-citation
    "ding2019dominant": ("doi", "10.1137/1.9781611975673.67", "deep anomaly detection on attributed networks"),
    "kipf2016vgae": ("arxiv", "1611.07308", "variational graph auto-encoders",
                     {"venue": "NIPS 2016 Workshop on Bayesian Deep Learning", "year": 2016}),
    "liu2008iforest": ("doi", "10.1109/ICDM.2008.17", "isolation forest"),
    "ma2021survey": ("doi", "10.1109/TKDE.2021.3118815", "graph anomaly detection with deep learning"),
    "liu2022bond": ("arxiv", "2206.10071", "bond",
                    {"venue": "Advances in Neural Information Processing Systems 35 (NeurIPS 2022), Datasets and Benchmarks Track", "year": 2022, "pages": "27021-27035"}),
    "fister2016": ("doi", "10.3389/fphy.2016.00049", "citation cartels", {"pages": "49"}),
    "kojaku2021": ("doi", "10.1038/s41598-021-93572-3", "anomalous citation groups"),
    "heneberg2016": ("doi", "10.1371/journal.pone.0153730", "citation stacking"),
    "vannoorden2019": ("doi", "10.1038/d41586-019-02479-7", "self-citing"),
    "ioannidis2019": ("doi", "10.1371/journal.pbio.3000384", "standardized citation metrics author database"),
    # text representations
    "devlin2019": ("doi", "10.18653/v1/N19-1423", "bert"),
    "beltagy2019": ("doi", "10.18653/v1/D19-1371", "scibert", {"pages": "3615-3620"}),
    "cohan2020": ("doi", "10.18653/v1/2020.acl-main.207", "specter"),
    "reimers2019": ("doi", "10.18653/v1/D19-1410", "sentence-bert", {"pages": "3982-3992"}),
    "chien2022giant": ("arxiv", "2111.00064", "node feature extraction by self-supervised multi-scale neighborhood prediction",
                       {"venue": "10th International Conference on Learning Representations (ICLR 2022)", "year": 2022}),
    # data sources
    "priem2022": ("arxiv", "2205.01833", "openalex"),
    "wang2020mag": ("doi", "10.1162/qss_a_00021", "microsoft academic graph"),
    "visser2021": ("doi", "10.1162/qss_a_00112", "large-scale comparison of bibliographic data sources"),
    "culbert2025": ("doi", "10.1007/s11192-025-05293-3", "reference coverage analysis of openalex"),
    # added in the revision: evaluation of temporal graph learning, paired tests for resampled comparisons
    "peng2025leakage": ("arxiv", "2510.25348", "from leakage to fidelity"),
    "poursafaei2022": ("arxiv", "2207.10128", "towards better evaluation for dynamic link prediction",
                       {"venue": "Advances in Neural Information Processing Systems 35 (NeurIPS 2022), Datasets and "
                                 "Benchmarks Track", "year": 2022}),
    "huang2023tgb": ("arxiv", "2307.01026", "temporal graph benchmark for machine learning on temporal graphs",
                     {"venue": "Advances in Neural Information Processing Systems 36 (NeurIPS 2023), Datasets and "
                               "Benchmarks Track", "year": 2023}),
    "nadeau2003": ("doi", "10.1023/A:1024068626366", "inference for the generalization error"),
    "bouckaert2004": ("doi", "10.1007/978-3-540-24775-3_3", "evaluating the replicability of significance tests",
                      # the Crossref record names the series first and the book second
                      {"venue": "Advances in Knowledge Discovery and Data Mining. Lecture Notes in Computer Science"}),
}

NEW_IN_REVISION = ["peng2025leakage", "poursafaei2022", "huang2023tgb", "nadeau2003", "bouckaert2004"]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]+", " ", s)


def tex_escape(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s or "")  # Crossref titles may contain JATS/MathML tags
    s = re.sub(r"\s+", " ", s).strip()
    for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def crossref(client: httpx.Client, doi: str) -> dict:
    r = client.get(f"https://api.crossref.org/works/{doi}", params={"mailto": MAILTO})
    r.raise_for_status()
    m = r.json()["message"]
    year = None
    for k in ("published-print", "published-online", "issued", "published"):
        if m.get(k, {}).get("date-parts"):
            year = m[k]["date-parts"][0][0]
            break
    authors = [f"{a.get('family', '')}, {a.get('given', '')}".strip(", ") for a in m.get("author", [])]
    if not authors and m.get("editor"):
        authors = [f"{a.get('family', '')}, {a.get('given', '')}".strip(", ") for a in m["editor"]]
    # Crossref "subtitle" is sometimes a real subtitle (Yan 2011) and sometimes a teaser (Science): opt-in per key
    subtitle = html.unescape(m["subtitle"][0]) if m.get("subtitle") else None
    return {"title": html.unescape((m.get("title") or [""])[0]), "subtitle": subtitle, "authors": authors, "year": year,
            "venue": html.unescape((m.get("container-title") or [""])[0]), "volume": m.get("volume"), "issue": m.get("issue"),
            "pages": m.get("page") or m.get("article-number"), "type": m.get("type"), "doi": m.get("DOI"),
            "publisher": m.get("publisher"), "source": "crossref"}


def arxiv(client: httpx.Client, aid: str) -> dict:
    r = client.get("https://export.arxiv.org/api/query", params={"id_list": aid})
    r.raise_for_status()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    e = ET.fromstring(r.text).find("a:entry", ns)
    return {"title": " ".join(e.find("a:title", ns).text.split()),
            "authors": [" ".join(x.find("a:name", ns).text.split()) for x in e.findall("a:author", ns)],
            "year": int(e.find("a:published", ns).text[:4]), "venue": f"arXiv preprint arXiv:{aid}",
            "arxiv": aid, "source": "arxiv"}


def bib_entry(key: str, m: dict) -> str:
    authors = " and ".join(tex_escape(a) for a in m["authors"]) or "{Anonymous}"
    fields = {"author": authors, "title": "{" + tex_escape(m["title"]) + "}", "year": str(m["year"])}
    if m["source"] == "arxiv" and "venue" in m.get("overrides", []):  # peer-reviewed version of an arXiv paper
        kind = "inproceedings"
        fields.update(booktitle=tex_escape(m["venue"]), eprint=m["arxiv"], archiveprefix="arXiv")
    elif m["source"] == "arxiv":
        kind = "misc"
        fields.update(howpublished=f"arXiv preprint arXiv:{m['arxiv']}", eprint=m["arxiv"], archiveprefix="arXiv")
    elif m.get("type") == "journal-article":
        kind = "article"
        fields.update(journal=tex_escape(m["venue"]))
    elif m.get("type") in ("proceedings-article", "book-chapter"):
        kind = "inproceedings"
        fields.update(booktitle=tex_escape(m["venue"]))
    else:
        kind = "misc"
        if m.get("venue"):
            fields["howpublished"] = tex_escape(m["venue"])
    for f in ("volume", "issue", "pages"):
        if m.get(f):
            fields["number" if f == "issue" else f] = str(m[f]).replace("-", "--")
    if m.get("publisher") and kind != "article":
        fields["publisher"] = tex_escape(m["publisher"])
    if m.get("doi"):
        fields["doi"] = m["doi"]
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items())
    return f"@{kind}{{{key},\n{body}\n}}\n"


def main() -> None:
    agent = f"reference-check/0.1 (mailto:{MAILTO})" if MAILTO else "reference-check/0.1"
    client = httpx.Client(timeout=30, headers={"User-Agent": agent})
    checks, bib, bad = {}, [], []
    for key, (kind, ident, expect, *extra) in REFS.items():
        try:
            m = crossref(client, ident) if kind == "doi" else arxiv(client, ident)
        except Exception as exc:  # noqa: BLE001
            checks[key] = {"id": ident, "ok": False, "error": str(exc)}
            bad.append(key)
            continue
        ok = norm(expect) in norm(m["title"])
        if extra:
            over = dict(extra[0])
            if over.pop("subtitle", False) and m.get("subtitle"):
                m["title"] += ": " + m["subtitle"]
            m.update(over)
            m["overrides"] = sorted(extra[0])
        checks[key] = {"id": ident, "ok": ok, **m}
        if ok:
            bib.append(bib_entry(key, m))
        else:
            bad.append(key)
        time.sleep(0.4 if kind == "doi" else 3.0)
    (HERE / "references.bib").write_text("% Generated by verify_refs.py from Crossref / arXiv metadata.\n\n" + "\n".join(bib))
    (HERE / "references_check.json").write_text(json.dumps(checks, indent=1, ensure_ascii=False))
    (HERE / "new_refs.json").write_text(json.dumps(NEW_IN_REVISION, indent=1) + "\n")
    for k, c in checks.items():
        print(("OK  " if c["ok"] else "BAD ") + k, "|", c.get("year"), "|", (c.get("title") or c.get("error", ""))[:90])
    print(f"{len(REFS) - len(bad)}/{len(REFS)} verified; failed: {bad}")


if __name__ == "__main__":
    main()
