""" Components of the model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class Encoder(nn.Module):
    def __init__(self, in_dim, hid, out_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hid, out_dim), nn.LayerNorm(out_dim))
    def forward(self, x):
        return self.net(x)


class EDLHead(nn.Module):
    def __init__(self, in_dim, K, dropout=0.1):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, K), nn.Softplus())
    def forward(self, x):
        # Evidence-based classification head.
        evi = self.net(x)
        a = evi + 1
        S = a.sum(1, keepdim=True)
        return {'evidence': evi, 'belief': a / S, 'uncertainty': self.K / S}


class QualityEst(nn.Module):
    def __init__(self, V, K):
        super().__init__()
        self.V, self.K = V, K
        self.nets = nn.ModuleList([nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1)) for _ in range(V)])
        self.temp = nn.Parameter(torch.tensor(1.0))
    def forward(self, evi_l, bel_l, unc_l):
        # Estimate per-view confidence weights.
        sc = []
        for m in range(self.V):
            dev = evi_l[m].device
            logK = torch.log(torch.tensor(float(self.K), device=dev)) if self.K > 1 else torch.tensor(1.0, device=dev)
            f = torch.cat([
                torch.log1p(evi_l[m].sum(1, keepdim=True)), unc_l[m],
                -(bel_l[m] * torch.log(bel_l[m] + 1e-8)).sum(1, keepdim=True) / logK,
                bel_l[m].max(1, keepdim=True)[0]], 1)
            sc.append(self.nets[m](f).squeeze(-1))
        return F.softmax(torch.stack(sc, 1) / self.temp.abs().clamp(min=0.5), 1)


class OmicsFusion(nn.Module):
    def __init__(self, dim_list, hid=128, dropout=0.1):
        super().__init__()
        self.V = len(dim_list)
        self.enc = nn.ModuleList([Encoder(d, hid * 2, hid, dropout) for d in dim_list])
        self.emb = nn.Parameter(torch.randn(self.V, hid) * 0.02)
        self.attn = nn.MultiheadAttention(hid, 4, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hid, hid * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(hid * 4, hid))
        self.n1 = nn.LayerNorm(hid)
        self.n2 = nn.LayerNorm(hid)
        self.gate = nn.Sequential(nn.Linear(hid + 1, hid), nn.Sigmoid())
    def forward(self, data, conf):
        # Fuse omics features with attention and confidence gating.
        vf = torch.stack([self.enc[m](data[m]) + self.emb[m] for m in range(self.V)], 1)
        h, _ = self.attn(vf, vf, vf)
        vf = self.n1(vf + h)
        vf = self.n2(vf + self.ffn(vf))
        g = self.gate(torch.cat([vf, conf.unsqueeze(-1)], -1))
        return (vf * g * conf.unsqueeze(-1)).sum(1)


class GraphFusion(nn.Module):
    def __init__(self, V, strategy="intersection"):
        super().__init__()
        self.V, self.strategy = V, strategy
    def forward(self, adj_list, N):
        # Merge per-view graphs into one fused graph.
        dev = adj_list[0].device if adj_list else torch.device('cpu')
        sl = torch.arange(N, device=dev)
        edges = [e for e in adj_list if e.numel() > 0]
        if not edges or N == 0:
            return torch.stack([sl, sl])
        merged = torch.cat(edges, 1)
        try:
            ei, cnt = torch.unique(merged, dim=1, return_counts=True)
        except NotImplementedError:
            ei, cnt = torch.unique(merged.cpu(), dim=1, return_counts=True)
            ei, cnt = ei.to(dev), cnt.to(dev)
        if self.strategy == "intersection":
            mask = cnt == len(edges)
            ei = ei[:, mask] if mask.any() else ei[:, :0]
        return torch.cat([ei, torch.stack([sl, sl])], 1)


class MRF(nn.Module):
    def __init__(self, V, K, dim_list, dropout=0.7):
        super().__init__()
        self.V, self.K = V, K
        self.enc = nn.ModuleList([Encoder(d, 256, 128, dropout) for d in dim_list])
        self.edl = nn.ModuleList([EDLHead(128, K, dropout * 0.5) for _ in range(V)])
        self.qual = QualityEst(V, K)

    def forward(self, data):
        # Run per-view encoding and confidence estimation.
        evi_l, bel_l, unc_l = [], [], []
        for m in range(self.V):
            out = self.edl[m](self.enc[m](data[m]))
            evi_l.append(out['evidence'])
            bel_l.append(out['belief'])
            unc_l.append(out['uncertainty'])
        conf = self.qual(evi_l, bel_l, unc_l)
        return {'evidence_list': evi_l, 'belief_list': bel_l,
                'uncertainty_list': unc_l, 'classification_confidence': conf}


class GNNStage(nn.Module):
    def __init__(self, V, K, dim_list, hid=128, gnn_hid=(128, 64), n_layers=2,
                 dropout=0.7, strategy="intersection"):
        super().__init__()
        self.dropout = dropout
        self.fus = OmicsFusion(dim_list, hid, dropout)
        self.gf = GraphFusion(V, strategy)
        gnn_hid_list = [gnn_hid] * n_layers if isinstance(gnn_hid, int) else list(gnn_hid)
        self.gnns = nn.ModuleList()
        self.norms = nn.ModuleList()
        d = hid
        for h in gnn_hid_list:
            self.gnns.append(SAGEConv(d, h, normalize=True))
            self.norms.append(nn.LayerNorm(h))
            d = h
        self.clf = nn.Linear(d, K)
        self.res = nn.Linear(hid, d)

    def forward(self, data, adj_list, conf):
        # Build node features, run GraphSAGE, and classify nodes.
        fused = self.fus(data, conf)
        N = data[0].shape[0]
        edge_index = self.gf(adj_list, N)
        x, x0 = fused, fused
        for i, (g, n) in enumerate(zip(self.gnns, self.norms)):
            x = F.gelu(n(g(x, edge_index)))
            if i < len(self.gnns) - 1:
                x = F.dropout(x, self.dropout, training=self.training)
        emb = x + self.res(x0)
        logits = self.clf(emb)
        return {'logits': logits, 'prob': F.softmax(logits, 1), 'embeddings': emb,
                'fused_edge_index': edge_index}


def init_model_dict(V, K, dim_list, cls_temp=1.0, strategy="intersection"):
    # Build the two-stage model dictionary.
    md = {}
    md["MRF"] = MRF(V, K, dim_list, dropout=0.7)
    md["MRF"].qual.temp.data.fill_(cls_temp)
    md["GNN"] = GNNStage(V, K, dim_list, hid=128, gnn_hid=[128, 64], n_layers=2,
                         dropout=0.7, strategy=strategy)
    return md
