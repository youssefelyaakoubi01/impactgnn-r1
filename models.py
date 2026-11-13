"""Models: MLP, GAT, GraphSAGE (+ edge-aware variant), HAN, HGT, and the GNN anomaly detectors
(DOMINANT, graph autoencoder). All node models output one value per paper (regression target or
classification logit); every GNN has a residual path so that papers without in-graph references
keep their own features.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, HANConv, HGTConv, SAGEConv
from torch_geometric.nn.conv import MessagePassing


class MLP(nn.Module):
    """``n_layers`` hidden layers (two in the submitted configuration)."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int = 1, dropout: float = 0.3, n_layers: int = 2) -> None:
        super().__init__()
        layers, d = [], in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.net = nn.Sequential(*layers, nn.Linear(d, out_dim))

    def forward(self, x, edge_index=None, edge_attr=None):
        return self.net(x)


class GAT(nn.Module):
    """GATConv layers (multi-head, optional edge features) with residual connections."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int = 1, heads: int = 4, dropout: float = 0.3,
                 edge_dim: int | None = None, n_layers: int = 2) -> None:
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([GATConv(hidden, hidden // heads, heads=heads, dropout=dropout, edge_dim=edge_dim,
                                            fill_value="mean") for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
        self.dropout = dropout
        self.use_edge = edge_dim is not None

    def forward(self, x, edge_index, edge_attr=None, return_attention: bool = False):
        ea = edge_attr if self.use_edge else None
        h = F.dropout(F.relu(self.inp(x)), self.dropout, self.training)
        att = []
        for i, conv in enumerate(self.convs):
            z = h if i == 0 else F.dropout(h, self.dropout, self.training)
            o, (ei2, a) = conv(z, edge_index, ea, return_attention_weights=True)
            h = F.elu(o) + h
            att.append(a)
        out = self.head(h)
        if return_attention:
            return out, (ei2, att[0], att[-1])
        return out


class EdgeSAGEConv(MessagePassing):
    """GraphSAGE (mean) whose messages also see the edge features: m_ij = W [x_j || e_ij]."""

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int) -> None:
        super().__init__(aggr="mean")
        self.lin_msg = nn.Linear(in_dim + edge_dim, out_dim)
        self.lin_root = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr) + self.lin_root(x)

    def message(self, x_j, edge_attr):
        return self.lin_msg(torch.cat([x_j, edge_attr], -1))


class SAGE(nn.Module):
    """GraphSAGE layers (mean aggregation, inductive), optionally edge-aware."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int = 1, dropout: float = 0.3,
                 edge_dim: int | None = None, n_layers: int = 2) -> None:
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([EdgeSAGEConv(hidden, hidden, edge_dim) if edge_dim else SAGEConv(hidden, hidden)
                                    for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
        self.dropout = dropout
        self.use_edge = bool(edge_dim)

    def forward(self, x, edge_index, edge_attr=None):
        args = (edge_attr,) if self.use_edge else ()
        h = F.dropout(F.relu(self.inp(x)), self.dropout, self.training)
        for i, conv in enumerate(self.convs):
            z = h if i == 0 else F.dropout(h, self.dropout, self.training)
            h = F.relu(conv(z, edge_index, *args)) + h
        return self.head(h)


class HAN(nn.Module):
    """HANConv layers over paper->paper metapath relations, residual input path."""

    def __init__(self, in_dim: int, hidden: int, metadata, out_dim: int = 1, heads: int = 4,
                 dropout: float = 0.3, n_layers: int = 2) -> None:
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([HANConv(hidden, hidden, metadata, heads=heads, dropout=dropout)
                                    for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
        self.dropout = dropout

    def forward(self, x_dict, edge_index_dict):
        h = F.dropout(F.relu(self.inp(x_dict["paper"])), self.dropout, self.training)
        for i, conv in enumerate(self.convs):
            z = h if i == 0 else F.dropout(h, self.dropout, self.training)
            o = conv({"paper": z}, edge_index_dict)["paper"]
            h = (o if o is not None else 0) + h
        return self.head(h)


class HGT(nn.Module):
    """Per-type input projections, HGTConv layers (gated residual inside HGTConv)."""

    def __init__(self, in_dims: dict[str, int], hidden: int, metadata, out_dim: int = 1, heads: int = 4,
                 dropout: float = 0.3, n_layers: int = 2) -> None:
        super().__init__()
        self.inp = nn.ModuleDict({t: nn.Linear(d, hidden) for t, d in in_dims.items()})
        self.convs = nn.ModuleList([HGTConv(hidden, hidden, metadata, heads=heads) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
        self.dropout = dropout

    def forward(self, x_dict, edge_index_dict):
        h = {t: F.dropout(F.relu(self.inp[t](x)), self.dropout, self.training) for t, x in x_dict.items()}
        for conv in self.convs:
            o = conv(h, edge_index_dict)
            h = {t: (F.dropout(o[t], self.dropout, self.training) + h[t]) if o.get(t) is not None else h[t]
                 for t in h}
        return self.head(h["paper"])


# ====================================================================== anomaly detectors (dense, small graphs)
def normalise_adj(A: torch.Tensor) -> torch.Tensor:
    A = A + torch.eye(A.shape[0], device=A.device)
    d = A.sum(1).pow(-0.5)
    return d[:, None] * A * d[None, :]


class DenseGCN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, A_hat):
        return A_hat @ self.lin(x)


class Dominant(nn.Module):
    """DOMINANT (Ding et al., SDM 2019): GCN encoder, GCN attribute decoder, inner-product structure decoder."""

    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.enc = nn.ModuleList([DenseGCN(in_dim, hidden), DenseGCN(hidden, hidden)])
        self.attr_dec = nn.ModuleList([DenseGCN(hidden, hidden), DenseGCN(hidden, in_dim)])
        self.struct_dec = DenseGCN(hidden, hidden)
        self.dropout = dropout

    def forward(self, x, A_hat):
        z = x
        for layer in self.enc:
            z = F.dropout(F.relu(layer(z, A_hat)), self.dropout, self.training)
        xa = F.relu(self.attr_dec[0](z, A_hat))
        x_hat = self.attr_dec[1](xa, A_hat)
        s = F.relu(self.struct_dec(z, A_hat))
        return x_hat, torch.sigmoid(s @ s.T)


class GAE(nn.Module):
    """Graph autoencoder: GCN encoder, inner-product decoder (structure only)."""

    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.enc = nn.ModuleList([DenseGCN(in_dim, hidden), DenseGCN(hidden, hidden)])
        self.dropout = dropout

    def forward(self, x, A_hat):
        z = F.dropout(F.relu(self.enc[0](x, A_hat)), self.dropout, self.training)
        z = self.enc[1](z, A_hat)
        return None, torch.sigmoid(z @ z.T)


def fit_detector(kind: str, x: torch.Tensor, A: torch.Tensor, seed: int, epochs: int = 300, alpha: float = 0.5,
                 lr: float = 5e-3) -> torch.Tensor:
    """Train a reconstruction detector on (x, A in [0,1]); return one anomaly score per node."""
    torch.manual_seed(seed)
    A_hat = normalise_adj(A)
    model = (Dominant if kind == "dominant" else GAE)(x.shape[1]).to(x.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        x_hat, a_hat = model(x, A_hat)
        s_err = ((a_hat - A) ** 2).sum(1)
        loss = s_err.mean() if x_hat is None else alpha * ((x_hat - x) ** 2).sum(1).mean() + (1 - alpha) * s_err.mean()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        x_hat, a_hat = model(x, A_hat)
        s = ((a_hat - A) ** 2).sum(1).sqrt()
        if x_hat is None:
            return s
        return alpha * ((x_hat - x) ** 2).sum(1).sqrt() + (1 - alpha) * s
