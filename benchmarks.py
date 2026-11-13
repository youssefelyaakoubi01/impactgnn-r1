"""Sanity check of the model implementations on ogbn-arxiv (official split, OGB evaluator).

``python benchmarks.py [--seeds 3] [--epochs 500]`` -> ``results/ogbn_arxiv.json``

The reference values are those of the OGB paper (Hu et al., NeurIPS 2020; Table 6 of arXiv:2005.00687v7): MLP 55.50 ± 0.23,
GCN 71.74 ± 0.29, GraphSAGE 71.49 ± 0.27 test accuracy. Our GAT/GraphSAGE add residual connections and
use our default architecture, so this is a check that the code learns sensibly, not a reproduction.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected

from config import ROOT, get_config, resolve_device
from models import GAT, MLP, SAGE

PUBLISHED = {"source": "Hu et al. 2020, Open Graph Benchmark, arXiv:2005.00687v7, Table 6 (test accuracy %)",
             "mlp": [55.50, 0.23], "gcn": [71.74, 0.29], "sage": [71.49, 0.27]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = get_config("full", device=args.device)
    device = resolve_device(cfg)
    _load = torch.load
    torch.load = lambda *a, **k: _load(*a, **{**k, "weights_only": False})  # ogb pickles its processed files
    from ogb.nodeproppred import Evaluator, PygNodePropPredDataset

    ds = PygNodePropPredDataset("ogbn-arxiv", root=str(ROOT / "data" / "ogb"))
    torch.load = _load
    data = ds[0]
    split = ds.get_idx_split()
    ev = Evaluator("ogbn-arxiv")
    x = data.x.to(device)
    y = data.y.squeeze(1).to(device)
    ei = to_undirected(data.edge_index).to(device)
    idx = {k: v.to(device) for k, v in split.items()}
    out = {"published": PUBLISHED, "runs": {}, "device": str(device)}
    for name in ("mlp", "sage", "gat"):
        accs = []
        t = time.time()
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            if name == "mlp":
                model = MLP(x.shape[1], 256, 40, dropout=0.5)
            elif name == "sage":
                model = SAGE(x.shape[1], 256, 40, dropout=0.5)
            else:
                model = GAT(x.shape[1], 256, 40, heads=4, dropout=0.5)
            model = model.to(device)
            opt = torch.optim.Adam(model.parameters(), lr=0.005)
            best_val, best_test = 0.0, 0.0
            for _ in range(args.epochs):
                model.train()
                opt.zero_grad()
                loss = F.cross_entropy(model(x, ei)[idx["train"]], y[idx["train"]])
                loss.backward()
                opt.step()
                model.eval()
                with torch.no_grad():
                    pred = model(x, ei).argmax(1, keepdim=True)
                acc = {k: ev.eval({"y_true": y[idx[k]].unsqueeze(1).cpu(), "y_pred": pred[idx[k]].cpu()})["acc"]
                       for k in ("valid", "test")}
                if acc["valid"] > best_val:
                    best_val, best_test = acc["valid"], acc["test"]
            accs.append(100 * best_test)
        out["runs"][name] = {"test_acc": [float(a) for a in accs], "mean": float(np.mean(accs)),
                             "std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                             "seconds": round(time.time() - t, 1)}
        print(name, out["runs"][name])
    (cfg.results_dir / "ogbn_arxiv.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
