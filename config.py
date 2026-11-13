"""Central configuration: paths, years, horizons, seeds, text models and run profiles.

Every script takes ``--profile small|full``. ``small`` is the end-to-end pilot (few topics,
few nodes, one seed); ``full`` is the real run. Results of ``full`` go to ``results/``, those of
``small`` to ``results/small/``.
"""

from __future__ import annotations

import json
import logging
import os
import platform
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHROMASEM = ROOT.parent

# ---------------------------------------------------------------- input data (read only)
OPEN_PDF_DIR = CHROMASEM / "DataOpenCorpus" / "pdf"
OPEN_MANIFEST = CHROMASEM / "DataOpenCorpus" / "manifest.csv"
LAB_PDF_DIR = CHROMASEM / "DataOurLabo"
# Optional directory of an earlier project of the authors (IMPACTGNN_PRIOR_DIR): earlier PDF->OpenAlex matches, which
# are checked again before use, and a fallback .env. Nothing in it is required; without it every PDF is matched anew.
FBN_ROOT = Path(os.environ.get("IMPACTGNN_PRIOR_DIR", Path.home() / "fbn-reference-relevance"))
FBN_ARTICLES = FBN_ROOT / "data" / "processed" / "full" / "articles.parquet"  # earlier PDF->OpenAlex matches
ENV_FILE = FBN_ROOT / ".env"  # holds OPENALEX_API_KEY; read only, never copied or printed
LOCAL_ENV_FILE = ROOT / ".env"  # optional OPENALEX_API_KEY of this project (git-ignored); takes precedence

# ---------------------------------------------------------------- OpenAlex
OPENALEX_URL = "https://api.openalex.org"
MAILTO = os.environ.get("OPENALEX_MAILTO", "")  # optional contact address for the OpenAlex polite pool; never hard-coded
CACHE_DIR = ROOT / "data" / "cache"  # shared by all profiles: a response is never downloaded twice

TEXT_MODELS = {
    "specter": "sentence-transformers/allenai-specter",
    "scibert": "allenai/scibert_scivocab_uncased",
}


@dataclass(frozen=True)
class Config:
    profile: str = "full"

    # years
    t0: int = 2017                      # first labeled publication year; every target is >= t0
    sample_years: tuple[int, int] = (2017, 2022)
    last_complete_year: int = 2025      # last calendar year with complete citation counts
    cby_first_year: int = 2012          # first year covered by counts_by_year (checked on 2026-10-24)
    horizons: tuple[int, ...] = (3, 5)
    label_inclusive: bool = True        # label window = years y .. y+k (inclusive)
    top_share: float = 0.10             # "high potential" = top 10 % of the publication-year cohort
    coldstart_test_start: dict = field(default_factory=lambda: {3: 2021, 5: 2020})
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    split: tuple[float, float, float] = (0.70, 0.15, 0.15)

    # collection
    n_lab_topics: int = 24
    n_open_topics: int = 24
    papers_per_topic: int = 250
    sample_seed: int = 20261024
    refs_of_open_seeds: bool = True     # fetch reference metadata of the open-corpus seeds too
    max_paper_nodes: int = 60000        # cap on paper nodes of the graph
    min_ref_citers: int = 2             # a non-sampled reference becomes a node if cited >= this
    workers: int = 4                    # concurrent OpenAlex requests
    min_interval: float = 0.12          # seconds between request starts (<= ~8 req/s overall)
    budget_reserve: int = 150           # stop calling OpenAlex when fewer daily credits remain
    title_threshold: float = 90.0       # rapidfuzz ratio for title matching
    max_title_searches: int = 60

    # text
    default_text: str = "specter"
    embed_batch: int = 256
    embed_max_len: int = 320

    # training
    device: str = "cuda"
    hidden: int = 128
    heads: int = 4
    dropout: float = 0.3
    lr: float = 3e-3
    weight_decay: float = 1e-4
    epochs: int = 400
    patience: int = 40
    amp: bool = True
    jobs: int = 3                       # concurrent training processes on the GPU
    metapath_k: int = 20                # HAN: neighbours kept per target and metapath

    # anomalies (sub-topic 4)
    venue_min_out: int = 10
    venue_min_links: int = 30
    n_cartels: int = 5
    cartel_size: int = 4
    n_selfcit: int = 10
    intensities: dict = field(default_factory=lambda: {"mild": 0.10, "medium": 0.25, "strong": 0.50})
    injection_draws: int = 3

    # ------------------------------------------------------------ derived paths
    @property
    def raw_dir(self) -> Path:
        return ROOT / "data" / "raw" / self.profile

    @property
    def processed_dir(self) -> Path:
        return ROOT / "data" / "processed" / self.profile

    @property
    def results_dir(self) -> Path:
        return ROOT / "results" if self.profile == "full" else ROOT / "results" / self.profile

    def ensure_dirs(self) -> None:
        for p in (self.raw_dir, self.processed_dir, self.results_dir, CACHE_DIR):
            p.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["coldstart_test_start"] = {str(k): v for k, v in self.coldstart_test_start.items()}
        return d


PROFILES = {
    "full": {},
    "small": dict(
        n_lab_topics=2, n_open_topics=2, papers_per_topic=100, refs_of_open_seeds=False,
        max_paper_nodes=5000, seeds=(0,), epochs=80, patience=20, injection_draws=1, jobs=2,
        max_title_searches=5,
    ),
}


def get_config(profile: str = "full", **overrides) -> Config:
    cfg = replace(Config(profile=profile), **PROFILES[profile])
    if overrides:
        cfg = replace(cfg, **{k: v for k, v in overrides.items() if v is not None})
    cfg.ensure_dirs()
    return cfg


def setup_logging(name: str, cfg: Config) -> logging.Logger:
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / f"{name}_{cfg.profile}.log")],
        force=True,
    )
    for noisy in ("httpx", "httpx2", "httpcore", "urllib3", "huggingface_hub", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(name)


def resolve_device(cfg: Config):
    """Return the torch device. CUDA is required unless ``--device cpu`` is given explicitly."""
    import torch

    if cfg.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available; pass --device cpu to run on CPU explicitly")
        return torch.device(cfg.device)
    return torch.device("cpu")


def write_env_json(cfg: Config) -> dict:
    """Record hardware and library versions in results/env.json."""
    import torch
    import torch_geometric

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "torch_geometric": torch_geometric.__version__,
        "cpu_count": os.cpu_count(),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, gpu_memory_gb=round(p.total_memory / 2**30, 1))
    for mod in ("pyg_lib", "torch_scatter", "torch_sparse", "sentence_transformers", "transformers",
                "sklearn", "pymupdf", "numpy", "pandas", "scipy"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", getattr(m, "VersionBind", "?"))
        except Exception:
            info[mod] = None
    path = cfg.results_dir / "env.json"
    path.write_text(json.dumps(info, indent=2))
    return info


def add_common_args(parser) -> None:
    parser.add_argument("--profile", default="full", choices=sorted(PROFILES))
    parser.add_argument("--device", default=None, help="cuda (default) or cpu")
    parser.add_argument("--jobs", type=int, default=None)
