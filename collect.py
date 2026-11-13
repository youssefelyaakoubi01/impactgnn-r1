"""Data collection: PDF reading, OpenAlex matching and the additional downloads.

Steps (``python collect.py --profile small|full --step all``):

1. ``seeds``   match every PDF of the open corpus and of the lab to an OpenAlex work (manifest /
               earlier FBN matches, verified with the DOI read by PyMuPDF; arXiv DOI and title
               search for the rest) and fetch the full records of the seeds.
2. ``topics``  pick the topics of the labeled sample: half around the lab papers, half across
               the strata of the open corpus.
3. ``sample``  random sample of 2017-2022 articles per topic (OpenAlex ``sample`` + ``seed``).
4. ``refs``    lightweight metadata of every reference of the sample and of the seeds.
5. ``nodes``   choose the reference nodes of the graph (shared references, capped) and fetch
               their abstracts and keywords.
6. ``authors`` author records (counts_by_year) of the sample and seed papers.
7. ``venues``  source records (counts_by_year) of every venue seen.

Every response is cached on disk (``data/cache``), 429/5xx are retried with exponential backoff
and jitter, and the daily credit budget is read from ``x-ratelimit-remaining``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode

import httpx
import numpy as np
import pandas as pd
from dotenv import dotenv_values
from rapidfuzz import fuzz

from config import (CACHE_DIR, ENV_FILE, FBN_ARTICLES, LOCAL_ENV_FILE, LAB_PDF_DIR, MAILTO, OPEN_MANIFEST, OPEN_PDF_DIR,
                    OPENALEX_URL, Config, add_common_args, get_config, setup_logging)

logger = logging.getLogger("collect")

FULL_SELECT = ",".join([
    "id", "doi", "title", "display_name", "publication_year", "type", "language", "authorships",
    "primary_location", "primary_topic", "topics", "keywords", "cited_by_count", "counts_by_year",
    "referenced_works", "referenced_works_count", "abstract_inverted_index",
])
LIGHT_SELECT = ",".join([
    "id", "doi", "title", "publication_year", "type", "authorships", "primary_location", "primary_topic",
    "cited_by_count", "counts_by_year", "referenced_works", "referenced_works_count",
])
TEXT_SELECT = "id,abstract_inverted_index,keywords,topics"
AUTHOR_SELECT = "id,display_name,works_count,cited_by_count,counts_by_year"
SOURCE_SELECT = "id,display_name,type,works_count,cited_by_count,counts_by_year,host_organization_name"

PLACEHOLDER_DOI = re.compile(r"n{4,}|x{4,}|0{6,}", re.I)
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"<>{}|\\^`\[\]]+", re.I)


# ====================================================================== small utilities
def short_id(url: str | None) -> str:
    return (url or "").rstrip("/").rsplit("/", 1)[-1]


def normalise_doi(doi: str | None) -> str:
    if not doi:
        return ""
    doi = doi.strip().lower()
    doi = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", doi)
    return doi.rstrip(".,;:)]}")


def normalise_title(t: str | None) -> str:
    t = unicodedata.normalize("NFKD", t or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "", t.lower())


def reconstruct_abstract(inv: dict[str, list[int]] | None) -> str:
    if not inv:
        return ""
    pos = sorted((i, w) for w, idxs in inv.items() for i in idxs)
    return " ".join(w for _, w in pos)


def chunks(seq: list, n: int) -> Iterable[list]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def compact_work(w: dict[str, Any]) -> dict[str, Any]:
    """Flatten an OpenAlex work into the fields used downstream (missing fields -> defaults)."""
    auth = w.get("authorships") or []
    loc = w.get("primary_location") or {}
    src = loc.get("source") or {}
    pt = w.get("primary_topic") or {}
    cby = sorted((c["year"], c.get("cited_by_count", 0)) for c in (w.get("counts_by_year") or []))
    refs = [short_id(r) for r in (w.get("referenced_works") or [])]
    rec = {
        "id": short_id(w.get("id")),
        "doi": normalise_doi(w.get("doi")),
        "title": w.get("title") or w.get("display_name") or "",
        "year": w.get("publication_year"),
        "type": w.get("type") or "",
        "language": w.get("language") or "",
        "cited_by_count": int(w.get("cited_by_count") or 0),
        "cby_years": [int(y) for y, _ in cby],
        "cby_counts": [int(c) for _, c in cby],
        "refs": refs,
        "n_refs": int(w.get("referenced_works_count") or len(refs)),
        "venue_id": short_id(src.get("id")),
        "venue_name": src.get("display_name") or "",
        "venue_type": src.get("type") or "",
        "author_ids": [short_id((a.get("author") or {}).get("id")) for a in auth],
        "author_names": [(a.get("author") or {}).get("display_name") or "" for a in auth],
        "topic_id": short_id(pt.get("id")),
        "topic_name": pt.get("display_name") or "",
        "subfield": (pt.get("subfield") or {}).get("display_name") or "",
        "field": (pt.get("field") or {}).get("display_name") or "",
        "domain": (pt.get("domain") or {}).get("display_name") or "",
    }
    if "topics" in w:
        rec["topic_ids"] = [short_id(t.get("id")) for t in (w.get("topics") or [])]
        rec["topic_scores"] = [float(t.get("score") or 0) for t in (w.get("topics") or [])]
    if "keywords" in w:
        rec["keywords"] = [k.get("display_name") or "" for k in (w.get("keywords") or [])]
    if "abstract_inverted_index" in w:
        rec["abstract"] = reconstruct_abstract(w.get("abstract_inverted_index"))
    return rec


def compact_counts(rec: dict[str, Any], kind: str) -> dict[str, Any]:
    cby = sorted(rec.get("counts_by_year") or [], key=lambda c: c["year"])
    out = {
        "id": short_id(rec.get("id")),
        "name": rec.get("display_name") or "",
        "works_count": int(rec.get("works_count") or 0),
        "cited_by_count": int(rec.get("cited_by_count") or 0),
        "cby_years": [int(c["year"]) for c in cby],
        "cby_works": [int(c.get("works_count") or 0) for c in cby],
        "cby_cited": [int(c.get("cited_by_count") or 0) for c in cby],
    }
    if kind == "sources":
        out["type"] = rec.get("type") or ""
        out["host"] = rec.get("host_organization_name") or ""
    return out


# ====================================================================== OpenAlex client
class BudgetExhausted(RuntimeError):
    """The daily OpenAlex credit budget is used up; cached answers remain available."""


class JsonCache:
    """One JSON file per request key, written atomically (safe with concurrent threads)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        h = hashlib.sha1(key.encode()).hexdigest()
        return self.root / h[:2] / f"{h}.json"

    def get(self, key: str) -> Any | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())["value"]
        except (json.JSONDecodeError, KeyError):
            return None

    def set(self, key: str, value: Any) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({"key": key, "value": value}))
        os.replace(tmp, p)


class RateLimiter:
    """Minimum interval between request starts, shared by all threads."""

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.lock = threading.Lock()
        self.next_t = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_t)
            self.next_t = t + self.interval
        if t > now:
            time.sleep(t - now)


def load_api_keys() -> list[str]:
    """OpenAlex keys in order of use: this project's .env (OPENALEX_API_KEY, OPENALEX_API_KEY_2, ...),
    then the FBN .env, then the environment; duplicates removed. Keys are never logged."""
    keys: list[str] = []
    for f in (LOCAL_ENV_FILE, ENV_FILE):
        vals = dotenv_values(f) if f.exists() else {}
        for name in sorted(vals, key=lambda n: (len(n), n)):
            if name.startswith("OPENALEX_API_KEY") and vals[name]:
                keys.append(vals[name].strip())
    if os.environ.get("OPENALEX_API_KEY"):
        keys.append(os.environ["OPENALEX_API_KEY"].strip())
    return list(dict.fromkeys(keys))


class OpenAlex:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.keys = load_api_keys()
        self.key_idx = 0
        if not self.keys:
            logger.warning("No OpenAlex API key found: the daily budget is 10x lower")
        logger.info("OpenAlex: %d API key(s) available; switching automatically when one is exhausted", len(self.keys))
        self.cache = JsonCache(CACHE_DIR / "openalex")
        self.limiter = RateLimiter(cfg.min_interval)
        agent = f"research-pipeline/0.1 (mailto:{MAILTO})" if MAILTO else "research-pipeline/0.1"
        self.client = httpx.Client(timeout=90, headers={"User-Agent": agent},
                                   limits=httpx.Limits(max_connections=cfg.workers + 2))
        self.lock = threading.Lock()
        self.remaining: int | None = None
        self.step = "init"
        self.stats: dict[str, Counter] = defaultdict(Counter)

    def _next_key(self, used_idx: int) -> bool:
        """Move to the next key after key ``used_idx`` is exhausted; False when no key is left."""
        with self.lock:
            if self.key_idx != used_idx:  # another thread already switched
                return True
            if self.key_idx + 1 >= len(self.keys):
                return False
            self.key_idx += 1
            self.remaining = None
            logger.warning("OpenAlex key %d/%d exhausted: switching to key %d/%d", used_idx + 1, len(self.keys),
                           self.key_idx + 1, len(self.keys))
            return True

    def _ledger(self, resp: httpx.Response, key_idx: int) -> None:
        """Append one line per request actually sent (never the key itself)."""
        rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "step": self.step, "status": resp.status_code,
               "credits": resp.headers.get("x-ratelimit-credits-used"),
               "remaining": resp.headers.get("x-ratelimit-remaining"), "key": key_idx + 1}
        with self.lock, open(self.cfg.raw_dir / "requests_ledger.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _count(self, field: str, n: int = 1) -> None:
        with self.lock:
            self.stats[self.step][field] += n

    def get(self, path: str, params: dict[str, Any], cache: bool = True) -> dict[str, Any]:
        """``cache=False`` skips the response cache (used for large paged listings that are stored compacted)."""
        key = path + "?" + urlencode(sorted(params.items()))
        cached = self.cache.get(key) if cache else None
        if cached is not None:
            self._count("cache_hits")
            return cached
        while self.remaining is not None and self.remaining <= self.cfg.budget_reserve:
            if not self._next_key(self.key_idx):
                raise BudgetExhausted(f"{self.remaining} credits left on the last key (reserve {self.cfg.budget_reserve})")
        for attempt in range(8):
            used_idx = self.key_idx
            q = dict(params, mailto=MAILTO) if MAILTO else dict(params)
            if self.keys:
                q["api_key"] = self.keys[used_idx]
            self.limiter.wait()
            try:
                resp = self.client.get(OPENALEX_URL + path, params=q)
            except httpx.HTTPError as exc:
                logger.warning("OpenAlex %s: %s (attempt %d)", path, type(exc).__name__, attempt + 1)
                time.sleep(min(60, 2 ** attempt) + random.random())
                continue
            self._count("requests")
            self._ledger(resp, used_idx)
            rem = resp.headers.get("x-ratelimit-remaining")
            used = resp.headers.get("x-ratelimit-credits-used")
            with self.lock:
                if rem is not None and rem.lstrip("-").isdigit() and used_idx == self.key_idx:
                    self.remaining = int(rem)
                if used is not None and used.isdigit():
                    self.stats[self.step]["credits"] += int(used)
                    self.stats[self.step][f"credits_key{used_idx + 1}"] += int(used)
            if resp.status_code == 200:
                data = resp.json()
                if cache:
                    self.cache.set(key, data)
                return data
            if resp.status_code == 429:
                self._count("http_429")
                low = rem is not None and rem.lstrip("-").isdigit() and int(rem) <= self.cfg.budget_reserve
                if low:
                    if self._next_key(used_idx):
                        continue
                    raise BudgetExhausted("daily budget exhausted on every key (HTTP 429)")
                retry_after = resp.headers.get("retry-after")
                wait = float(retry_after) if retry_after and retry_after.replace(".", "").isdigit() else 0.0
                time.sleep(max(wait, min(120, 2 ** (attempt + 1))) + random.random())
                continue
            if resp.status_code >= 500:
                self._count("http_5xx")
                time.sleep(min(60, 2 ** attempt) + random.random())
                continue
            # 400/403/404: cache the empty answer so the request is not repeated
            self._count(f"http_{resp.status_code}")
            logger.warning("OpenAlex HTTP %d for %s", resp.status_code, key[:160])
            empty = {"results": [], "status": resp.status_code}
            if cache:
                self.cache.set(key, empty)
            return empty
        raise RuntimeError(f"OpenAlex request failed after retries: {key[:160]}")

    # ------------------------------------------------------------------ batch lookups
    def by_ids(self, entity: str, ids: list[str], select: str, field: str = "ids.openalex") -> list[dict]:
        """Fetch records in batches of 50 with ``filter=<field>:a|b|...``, several requests in flight."""
        ids = [i for i in dict.fromkeys(ids) if i]
        plain = [i for i in ids if not re.search(r"[,|<>;#?&\s()]", i)]
        odd = [i for i in ids if re.search(r"[,|<>;#?&\s()]", i)]
        batches = list(chunks(sorted(plain), 50)) + [[i] for i in odd]

        def one(batch: list[str]) -> list[dict]:
            data = self.get(f"/{entity}", {"filter": f"{field}:{'|'.join(batch)}", "per-page": 50,
                                           "select": select})
            return data.get("results", [])

        out: list[dict] = []
        done = 0
        with ThreadPoolExecutor(self.cfg.workers) as pool:
            for res in pool.map(one, batches):
                out.extend(res)
                done += 1
                if done % 200 == 0:
                    logger.info("  %s: %d/%d batches, %s credits left", entity, done, len(batches), self.remaining)
        return out

    def title_match(self, title: str, year: int | None) -> dict | None:
        norm = normalise_title(title)
        if len(norm) < 15:
            return None
        q = " ".join(re.sub(r"[^\w\s]", " ", title).split()[:30])
        filt = f"title.search:{q}"
        if year:
            filt += f",publication_year:{year - 1}-{year + 1}"
        data = self.get("/works", {"filter": filt, "per-page": 10, "select": FULL_SELECT})
        best, best_score = None, 0.0
        for w in data.get("results", []):
            score = fuzz.ratio(norm, normalise_title(w.get("title") or w.get("display_name")))
            if score > best_score:
                best, best_score = w, score
        if best is not None and best_score >= self.cfg.title_threshold:
            return dict(best, _match_score=best_score)
        return None


# ====================================================================== PDF reading
def pdf_text_dois(path: Path, pages: int = 2) -> tuple[list[str], str]:
    """DOIs found on the first pages of a PDF (in order) and the normalised text of those pages."""
    import pymupdf

    try:
        with pymupdf.open(path) as doc:
            text = " ".join(doc[i].get_text() for i in range(min(pages, doc.page_count)))
    except Exception as exc:  # corrupt PDF
        logger.warning("PyMuPDF failed on %s: %s", path.name, exc)
        return [], ""
    dois = [normalise_doi(m.group(0)) for m in DOI_RE.finditer(text)]
    return list(dict.fromkeys(d for d in dois if len(d) > 7)), normalise_title(text)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ====================================================================== steps
def step_seeds(cfg: Config, oa: OpenAlex) -> pd.DataFrame:
    """Match every PDF to OpenAlex and fetch the full records of the matched seeds."""
    oa.step = "seeds"
    man = pd.read_csv(OPEN_MANIFEST, dtype=str).fillna("")
    man = man[man.status == "accepted"].copy()
    fbn = pd.read_parquet(FBN_ARTICLES) if FBN_ARTICLES.exists() else pd.DataFrame(columns=["doc_id"])
    fbn_oa = dict(zip(fbn.doc_id, fbn.get("openalex_id", pd.Series(dtype=str)).fillna("")))

    rows = []
    # ---- open corpus: manifest openalex_id (OpenAlex-sourced PDFs) or earlier FBN match
    for r in man.itertuples():
        path = OPEN_PDF_DIR / Path(r.file or f"{r.doc_id}.pdf").name
        oid, method = "", ""
        if r.openalex_id:
            oid, method = r.openalex_id, "manifest_openalex_id"
        elif fbn_oa.get(r.doc_id):
            oid, method = fbn_oa[r.doc_id], "fbn_match"
        rows.append(dict(doc_id=r.doc_id, corpus="open", stratum=r.stratum, pdf=str(path), title_hint=r.title,
                         year_hint=int(float(r.year)) if r.year else None, doi_hint=normalise_doi(r.doi),
                         arxiv_id=r.arxiv_id, openalex_id=oid, match_method=method))
    # ---- lab: all 40 files, exact duplicates removed by sha256; FBN matches reused
    lab_files = sorted(LAB_PDF_DIR.glob("*.pdf"))
    fbn_lab = fbn[fbn.get("corpus", pd.Series(dtype=str)) == "lab"] if len(fbn) else fbn
    fbn_lab_by_file = {Path(p).name: (d, o) for p, d, o in zip(fbn_lab.pdf_path, fbn_lab.doc_id,
                                                               fbn_lab.openalex_id.fillna(""))}
    seen_sha: dict[str, str] = {}
    lab_report = []
    for path in lab_files:
        h = sha256(path)
        if h in seen_sha:
            lab_report.append(dict(file=path.name, status="duplicate", of=seen_sha[h]))
            continue
        seen_sha[h] = path.name
        doc_id, oid = fbn_lab_by_file.get(path.name, (f"lab_{normalise_title(path.stem)[:60]}", ""))
        rows.append(dict(doc_id=doc_id, corpus="lab", stratum="lab", pdf=str(path), title_hint="", year_hint=None,
                         doi_hint="", arxiv_id="", openalex_id=oid, match_method="fbn_match" if oid else ""))
        lab_report.append(dict(file=path.name, status="kept"))
    seeds = pd.DataFrame(rows)

    # ---- PyMuPDF: DOIs on the first two pages (verification and fallback)
    logger.info("Reading %d PDFs with PyMuPDF", len(seeds))
    pdf_info = [pdf_text_dois(Path(p)) for p in seeds.pdf]
    seeds["pdf_dois"] = [d for d, _ in pdf_info]
    pdf_text = dict(zip(seeds.doc_id, (t for _, t in pdf_info)))

    def verified(r, w: dict) -> bool:
        """A match is accepted if the titles agree, or the OpenAlex DOI or title is printed in the PDF."""
        oa_title = normalise_title(w.get("title") or w.get("display_name"))
        doi = normalise_doi(w.get("doi"))
        # template DOIs such as 10.1145/nnnnnnn.nnnnnnn exist both in PDFs and in OpenAlex
        doi_in_pdf = bool(doi) and not PLACEHOLDER_DOI.search(doi) and doi in r.pdf_dois
        if r.title_hint:
            ratio = fuzz.ratio(normalise_title(r.title_hint), oa_title)
            return ratio >= cfg.title_threshold or (doi_in_pdf and ratio >= 70)
        return doi_in_pdf or (len(oa_title) >= 15 and oa_title in pdf_text[r.doc_id])

    # ---- verify the inherited matches (some earlier title matches of arXiv PDFs were wrong)
    recs_raw = {short_id(w["id"]): w for w in
                oa.by_ids("works", seeds.loc[seeds.openalex_id != "", "openalex_id"].tolist(), FULL_SELECT)}
    rejected = []
    for r in seeds[seeds.openalex_id != ""].itertuples():
        if r.match_method == "manifest_openalex_id":  # the PDF was downloaded from this OpenAlex record
            continue
        w = recs_raw.get(r.openalex_id)
        if w is None or not verified(r, w):
            rejected.append((r.doc_id, r.openalex_id))
            seeds.loc[seeds.doc_id == r.doc_id, ["openalex_id", "match_method"]] = ["", ""]
    logger.info("Inherited matches rejected by verification: %d", len(rejected))

    # ---- fallbacks for unmatched PDFs: DOI hints, arXiv DOI, PDF DOIs, then title search
    todo = seeds[seeds.openalex_id == ""]
    cand_dois: dict[str, list[str]] = {}
    for r in todo.itertuples():
        c = [r.doi_hint] if r.doi_hint else []
        if r.arxiv_id:
            c.append(f"10.48550/arxiv.{r.arxiv_id.lower()}")
        c += list(r.pdf_dois[:3])
        cand_dois[r.doc_id] = [d for d in c if d]
    found = {normalise_doi(w.get("doi")): w for w in
             oa.by_ids("works", sorted({d for v in cand_dois.values() for d in v}), FULL_SELECT, field="doi")}
    n_search = 0
    for r in todo.itertuples():
        # a DOI read in the PDF may belong to a cited paper: keep only verified candidates
        hit = next((found[d] for d in cand_dois[r.doc_id] if d in found and verified(r, found[d])), None)
        method = "doi_lookup"
        if hit is None and r.title_hint and n_search < cfg.max_title_searches:
            n_search += 1
            hit, method = oa.title_match(r.title_hint, r.year_hint), "title_search"
        if hit is not None:
            recs_raw[short_id(hit["id"])] = hit
            seeds.loc[seeds.doc_id == r.doc_id, ["openalex_id", "match_method"]] = [short_id(hit["id"]), method]
    seeds["rejected_match"] = seeds.doc_id.isin({d for d, _ in rejected})

    # ---- full records of all matched seeds (fetched above; title-search hits carry the full select)
    recs = {k: compact_work(w) for k, w in recs_raw.items() if k in set(seeds.openalex_id)}
    seeds["oa_doi"] = seeds.openalex_id.map(lambda i: recs.get(i, {}).get("doi", ""))
    seeds["doi_in_pdf"] = [bool(d) and d in pd_ for d, pd_ in zip(seeds.oa_doi, seeds.pdf_dois)]
    seeds["has_record"] = seeds.openalex_id.isin(recs)
    # the same work can be matched by two PDFs (e.g. preprint and published copy): keep the first
    seeds["dup_work"] = seeds.openalex_id.ne("") & seeds.duplicated("openalex_id")
    seeds.to_parquet(cfg.raw_dir / "seeds.parquet")
    works = pd.DataFrame(list(recs.values()))
    works.to_parquet(cfg.raw_dir / "seed_works.parquet")
    (cfg.raw_dir / "lab_files.json").write_text(json.dumps(lab_report, indent=1))
    logger.info("Seeds: %d PDFs, %d matched (%s), %d records; DOI found in PDF for %d",
                len(seeds), (seeds.openalex_id != "").sum(), seeds.match_method.value_counts().to_dict(),
                len(recs), seeds.doi_in_pdf.sum())
    return seeds


def step_topics(cfg: Config) -> list[dict]:
    """Half the topics around the lab papers, half across the open-corpus strata."""
    seeds = pd.read_parquet(cfg.raw_dir / "seeds.parquet")
    works = pd.read_parquet(cfg.raw_dir / "seed_works.parquet").set_index("id")
    seeds = seeds[seeds.has_record & ~seeds.dup_work]
    names: dict[str, str] = {}
    chosen: list[dict] = []

    # lab topics: all topics of the lab papers, primary topic weighted 2
    score: Counter = Counter()
    for oid in seeds.loc[seeds.corpus == "lab", "openalex_id"]:
        w = works.loc[oid]
        for j, t in enumerate(w.topic_ids):
            score[t] += 2 if j == 0 else 1
        names[w.topic_id] = w.topic_name
    for t, s in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))[:cfg.n_lab_topics]:
        chosen.append(dict(topic_id=t, group="lab", weight=s))

    # open topics: most frequent primary topics per stratum, round robin over strata
    per_stratum: dict[str, list[str]] = {}
    for stratum, g in seeds[seeds.corpus == "open"].groupby("stratum"):
        cnt = Counter(works.loc[g.openalex_id, "topic_id"])
        cnt.pop("", None)
        per_stratum[stratum] = [t for t, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))]
        for oid in g.openalex_id:
            names[works.loc[oid, "topic_id"]] = works.loc[oid, "topic_name"]
    taken = {c["topic_id"] for c in chosen}
    # strata with more labelable seeds (year <= last sample year) pick first in each round
    n_lab = seeds[seeds.corpus == "open"].assign(y=lambda d: d.openalex_id.map(works.year)).query(
        f"y <= {cfg.sample_years[1]}").groupby("stratum").size()
    order = sorted(per_stratum, key=lambda s: (-int(n_lab.get(s, 0)), s))
    rank = 0
    while sum(c["group"] == "open" for c in chosen) < cfg.n_open_topics and rank < 50:
        for stratum in order:
            lst = [t for t in per_stratum[stratum] if t not in taken]
            if rank < len(per_stratum[stratum]) and lst and sum(c["group"] == "open" for c in chosen) < cfg.n_open_topics:
                t = lst[0]
                chosen.append(dict(topic_id=t, group="open", stratum=stratum))
                taken.add(t)
        rank += 1
    for c in chosen:
        c["name"] = names.get(c["topic_id"], "")
    (cfg.raw_dir / "topics.json").write_text(json.dumps(chosen, indent=1))
    logger.info("Topics: %d lab + %d open", sum(c["group"] == "lab" for c in chosen),
                sum(c["group"] == "open" for c in chosen))
    return chosen


def step_sample(cfg: Config, oa: OpenAlex) -> pd.DataFrame:
    """Random 2017-2022 articles per topic (not ranked by citations, to avoid selection bias)."""
    oa.step = "sample"
    topics = json.loads((cfg.raw_dir / "topics.json").read_text())
    y0, y1 = cfg.sample_years
    rows = []

    def one(t: dict) -> list[dict]:
        filt = (f"primary_topic.id:{t['topic_id']},publication_year:{y0}-{y1},type:article,language:en,"
                f"referenced_works_count:>4")
        out = []
        for page in range(1, (cfg.papers_per_topic - 1) // 200 + 2):
            data = oa.get("/works", {"filter": filt, "sample": cfg.papers_per_topic, "seed": cfg.sample_seed,
                                     "per-page": 200, "page": page, "select": FULL_SELECT})
            res = data.get("results", [])
            out.extend(res)
            if len(res) < 200:
                break
        return out

    with ThreadPoolExecutor(cfg.workers) as pool:
        for t, res in zip(topics, pool.map(one, topics)):
            for w in res:
                rec = compact_work(w)
                rec.update(sampled_topic=t["topic_id"], topic_group=t["group"])
                rows.append(rec)
            t["n_sampled"] = len(res)
    df = pd.DataFrame(rows).drop_duplicates("id")
    df.to_parquet(cfg.raw_dir / "sample.parquet")
    (cfg.raw_dir / "topics.json").write_text(json.dumps(topics, indent=1))
    logger.info("Sample: %d unique papers from %d topics", len(df), len(topics))
    return df


def _citing_works(cfg: Config) -> pd.DataFrame:
    seeds = pd.read_parquet(cfg.raw_dir / "seeds.parquet")
    sw = pd.read_parquet(cfg.raw_dir / "seed_works.parquet")
    lab_ids = set(seeds.loc[seeds.corpus == "lab", "openalex_id"])
    if not cfg.refs_of_open_seeds:
        sw = sw[sw.id.isin(lab_ids)]
    sample = pd.read_parquet(cfg.raw_dir / "sample.parquet")
    return pd.concat([sample, sw], ignore_index=True).drop_duplicates("id")


def step_refs(cfg: Config, oa: OpenAlex) -> pd.DataFrame:
    """Lightweight metadata of every reference of the sample and seed papers."""
    oa.step = "refs"
    citing = _citing_works(cfg)
    have = set(pd.read_parquet(cfg.raw_dir / "seed_works.parquet").id) | set(
        pd.read_parquet(cfg.raw_dir / "sample.parquet").id)
    ref_ids = sorted({r for refs in citing.refs for r in refs} - have)
    logger.info("References: %d unique to fetch (%d batches)", len(ref_ids), (len(ref_ids) + 49) // 50)
    recs = [compact_work(w) for w in oa.by_ids("works", ref_ids, LIGHT_SELECT)]
    df = pd.DataFrame(recs).drop_duplicates("id")
    df.to_parquet(cfg.raw_dir / "refs.parquet")
    logger.info("References: %d records (%.1f%% of requested)", len(df), 100 * len(df) / max(1, len(ref_ids)))
    return df


def step_nodes(cfg: Config, oa: OpenAlex) -> list[str]:
    """Reference nodes of the graph: all references of lab papers, then shared references."""
    oa.step = "nodes"
    seeds = pd.read_parquet(cfg.raw_dir / "seeds.parquet")
    sw = pd.read_parquet(cfg.raw_dir / "seed_works.parquet")
    sample = pd.read_parquet(cfg.raw_dir / "sample.parquet")
    refs = pd.read_parquet(cfg.raw_dir / "refs.parquet")
    base = set(sample.id) | set(sw.id)
    lab_ids = set(seeds.loc[seeds.corpus == "lab", "openalex_id"])
    known = set(refs.id)
    citers: Counter = Counter()
    for refs_ in pd.concat([sample.refs, sw.refs]):
        citers.update(set(refs_))
    lab_refs = {r for refs_ in sw.loc[sw.id.isin(lab_ids), "refs"] for r in refs_} & known
    budget = cfg.max_paper_nodes - len(base)
    chosen = sorted(lab_refs - base)[:max(0, budget)]
    shared = [r for r, c in sorted(citers.items(), key=lambda kv: (-kv[1], kv[0]))
              if c >= cfg.min_ref_citers and r in known and r not in base and r not in lab_refs]
    chosen += shared[:max(0, budget - len(chosen))]
    logger.info("Nodes: %d base papers + %d lab refs + %d shared refs (cap %d; %d shared candidates)",
                len(base), len(lab_refs - base), len(chosen) - len(lab_refs - base), cfg.max_paper_nodes,
                len(shared))
    texts = [compact_work(w) for w in oa.by_ids("works", chosen, TEXT_SELECT)]
    pd.DataFrame(texts)[["id", "abstract", "keywords", "topic_ids", "topic_scores"]].to_parquet(
        cfg.raw_dir / "node_ref_texts.parquet")
    (cfg.raw_dir / "node_refs.json").write_text(json.dumps(chosen))
    return chosen


def step_authors(cfg: Config, oa: OpenAlex) -> pd.DataFrame:
    oa.step = "authors"
    sw = pd.read_parquet(cfg.raw_dir / "seed_works.parquet")
    sample = pd.read_parquet(cfg.raw_dir / "sample.parquet")
    ids = sorted({a for lst in pd.concat([sample.author_ids, sw.author_ids]) for a in lst if a})
    logger.info("Authors: %d unique (%d batches)", len(ids), (len(ids) + 49) // 50)
    df = pd.DataFrame([compact_counts(r, "authors") for r in oa.by_ids("authors", ids, AUTHOR_SELECT)])
    df.drop_duplicates("id").to_parquet(cfg.raw_dir / "authors.parquet")
    return df


def step_venues(cfg: Config, oa: OpenAlex) -> pd.DataFrame:
    oa.step = "venues"
    frames = [pd.read_parquet(cfg.raw_dir / f, columns=["venue_id"])
              for f in ("seed_works.parquet", "sample.parquet", "refs.parquet")]
    ids = sorted(set(pd.concat(frames).venue_id) - {""})
    logger.info("Venues: %d unique (%d batches)", len(ids), (len(ids) + 49) // 50)
    df = pd.DataFrame([compact_counts(r, "sources") for r in oa.by_ids("sources", ids, SOURCE_SELECT)])
    df.drop_duplicates("id").to_parquet(cfg.raw_dir / "venues.parquet")
    return df


STEPS = ["seeds", "topics", "sample", "refs", "nodes", "authors", "venues"]


def write_report(cfg: Config, oa: OpenAlex, t_steps: dict[str, float]) -> None:
    """Per step: ``calls`` = distinct API calls the step needs (cache hits + requests of the last run);
    ``requests_sent`` and ``credits`` accumulate over runs (every sent request is also in requests_ledger.jsonl)."""
    path = cfg.raw_dir / "collect_report.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    steps = old.get("steps", {})
    for s, c in oa.stats.items():
        prev = steps.get(s, {})
        new = {"calls": int(c.get("cache_hits", 0) + c.get("requests", 0)),
               "requests_sent": int(prev.get("requests_sent", 0)) + int(c.get("requests", 0)),
               "credits": int(prev.get("credits", 0)) + int(c.get("credits", 0)),
               "seconds_last_run": round(t_steps.get(s, 0), 1)}
        new.update({k: int(prev.get(k, 0)) + int(v) for k, v in c.items() if k.startswith(("http_", "credits_key"))})
        steps[s] = new
    report = {"profile": cfg.profile, "steps": steps, "credits_remaining_at_end": oa.remaining,
              "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    path.write_text(json.dumps(report, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--step", default="all", help=f"all or comma list of {STEPS}")
    args = ap.parse_args()
    cfg = get_config(args.profile)
    setup_logging("collect", cfg)
    steps = STEPS if args.step == "all" else args.step.split(",")
    oa = OpenAlex(cfg)
    t_steps: dict[str, float] = {}
    try:
        for s in steps:
            t = time.time()
            logger.info("=== step %s (profile %s)", s, cfg.profile)
            if s == "topics":
                step_topics(cfg)
            else:
                globals()[f"step_{s}"](cfg, oa)
            t_steps[s] = time.time() - t
            logger.info("=== step %s done in %.0f s; credits left: %s", s, t_steps[s], oa.remaining)
    except BudgetExhausted as exc:
        logger.error("Stopped: %s. Re-run the same command after the daily reset; cached answers are reused.", exc)
        raise SystemExit(2)
    finally:
        write_report(cfg, oa, t_steps)


if __name__ == "__main__":
    main()
