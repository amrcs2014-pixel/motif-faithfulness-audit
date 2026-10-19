"""CCMG-class backbone (MotifCellGNN: molecular graph + motif nodes + multi-omics cell node in one hierarchical graph)
and the late-fusion baseline. Re-implemented from the CCMG abstract and standard practice; no public CCMG code
was available (checked 16 Oct 2026).

Drug side (shared by every pair): a GINE message-passing network over the atom graph; motif nodes are initialised
from the mean of their atoms' embeddings plus a motif-vocabulary embedding and a size feature.
Cell side: three omics towers (expression, mutation, copy number) fused into one cell embedding.
Unified motif-cell graph (per pair): motif nodes connected by the cut bonds, plus one cell node connected to every
motif. Each layer (i) updates motifs from their motif neighbours AND the cell node (cell conditioning), then
(ii) updates the cell node by attention over the motifs (the model's own motif-cell attention).
Readout: [cell node, attention-pooled motifs, atom-level drug pooling, cell embedding] -> MLP -> LN IC50.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch_geometric.utils import scatter


class Bank:
    """A list of graph dicts (chem.build_graph output, or counterfactual edits) batched into flat GPU tensors."""

    def __init__(self, graphs, device):
        xs, eis, eas, atom_drug, atom_motif, motif_drug, motif_vid, motif_size = [], [], [], [], [], [], [], []
        D = len(graphs)
        self.M = max(int(g["n_motifs"]) for g in graphs)
        adj = torch.zeros(D, self.M, self.M)
        mask = torch.zeros(D, self.M, dtype=torch.bool)
        pad_index = torch.full((D, self.M), -1, dtype=torch.long)
        na = nm = 0
        for d, g in enumerate(graphs):
            n_at, n_mo = int(g["n_atoms"]), int(g["n_motifs"])
            xs.append(torch.as_tensor(np.asarray(g["x"], np.float32)))
            ei = torch.as_tensor(np.asarray(g["edge_index"], np.int64)).reshape(2, -1)
            eis.append(ei + na)
            eas.append(torch.as_tensor(np.asarray(g["edge_attr"], np.float32)).reshape(-1, 6))
            atom_drug.append(torch.full((n_at,), d, dtype=torch.long))
            a2m = np.asarray(g["a2m"], np.int64)
            atom_motif.append(torch.as_tensor(a2m) + nm)
            motif_drug.append(torch.full((n_mo,), d, dtype=torch.long))
            motif_vid.append(torch.as_tensor(np.asarray(g["motif_vid"], np.int64)))
            motif_size.append(torch.as_tensor(np.bincount(a2m, minlength=n_mo).astype(np.float32)))
            pad_index[d, :n_mo] = torch.arange(nm, nm + n_mo)
            mask[d, :n_mo] = True
            for a, b in g["motif_edges"]:
                adj[d, a, b] = 1.0; adj[d, b, a] = 1.0
            na += n_at; nm += n_mo
        self.D, self.Na, self.Nm = D, na, nm
        self.x = torch.cat(xs).to(device)
        self.edge_index = torch.cat(eis, 1).to(device)
        self.edge_attr = torch.cat(eas).to(device)
        self.atom_drug = torch.cat(atom_drug).to(device)
        self.atom_motif = torch.cat(atom_motif).to(device)
        self.motif_drug = torch.cat(motif_drug).to(device)
        self.motif_vid = torch.cat(motif_vid).to(device)
        self.motif_size = torch.cat(motif_size).to(device)
        self.adj = adj.to(device)
        self.mask = mask.to(device)
        self.pad_index = pad_index.to(device)
        self.n_motifs = mask.sum(1).to(device)


class AtomGNN(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden, n_layers=3, dropout=0.1):
        super().__init__()
        self.lin_in = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim))
            self.norms.append(nn.LayerNorm(hidden))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        h = F.relu(self.lin_in(x))
        for conv, norm in zip(self.convs, self.norms):
            h = h + F.dropout(F.relu(norm(conv(h, edge_index, edge_attr))), self.dropout, self.training)
        return h


def tower(d_in, d_mid, d_out, dropout):
    return nn.Sequential(nn.Linear(d_in, d_mid), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_mid, d_out), nn.ReLU())


class CellEncoder(nn.Module):
    """Multi-omics cell embedding from [expression | mutation | copy number] (each G genes)."""

    def __init__(self, G, hidden, dropout=0.1):
        super().__init__()
        self.G = G
        self.expr = tower(G, 512, hidden, dropout)
        self.mut = tower(G, 256, hidden, dropout)
        self.cnv = tower(G, 256, hidden, dropout)
        self.fuse = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU())

    def forward(self, x):
        G = self.G
        return self.fuse(torch.cat([self.expr(x[:, :G]), self.mut(x[:, G:2 * G]), self.cnv(x[:, 2 * G:])], -1))


class MotifCellLayer(nn.Module):
    def __init__(self, H, dropout=0.1, n_heads=1):
        super().__init__()
        assert H % n_heads == 0
        self.msg = nn.Sequential(nn.Linear(3 * H, H), nn.ReLU(), nn.Linear(H, H))
        self.norm_m = nn.LayerNorm(H)
        self.q = nn.Linear(H, H); self.k = nn.Linear(H, H); self.v = nn.Linear(H, H); self.o = nn.Linear(H, H)
        self.norm_c = nn.LayerNorm(H)
        self.ff_c = nn.Sequential(nn.Linear(H, H), nn.ReLU(), nn.Linear(H, H))
        self.H = H; self.dropout = dropout; self.n_heads = n_heads

    def forward(self, Hm, hc, adj, mask, logmask):
        """Hm [B,M,H] motif states, hc [B,H] cell node, adj [B,M,M], mask [B,M] in [0,1], logmask [B,M] additive.
        With n_heads > 1 the cell node attends with several heads; the returned alpha is the head average (the
        single-head case is numerically identical to the original single-head layer)."""
        adj_m = adj * mask.unsqueeze(1)                                     # messages from masked motifs are dropped
        nb = torch.bmm(adj_m, Hm) / adj_m.sum(-1, keepdim=True).clamp(min=1.0)
        upd = self.msg(torch.cat([Hm, nb, hc.unsqueeze(1).expand_as(Hm)], -1))
        Hm = Hm + F.dropout(self.norm_m(upd), self.dropout, self.training) * mask.unsqueeze(-1)
        B, M, _ = Hm.shape; h = self.n_heads; d = self.H // h
        q = self.q(hc).view(B, h, 1, d)
        k = self.k(Hm).view(B, M, h, d).transpose(1, 2)                     # [B,h,M,d]
        logits = (k * q).sum(-1) / math.sqrt(d) + logmask.unsqueeze(1)      # [B,h,M]
        alpha_h = torch.softmax(logits, -1)
        v = self.v(Hm).view(B, M, h, d).transpose(1, 2)
        ctx = (alpha_h.unsqueeze(-1) * v).sum(2).reshape(B, self.H)
        hc = hc + F.dropout(self.o(ctx), self.dropout, self.training)
        hc = hc + self.ff_c(self.norm_c(hc))
        return Hm, hc, alpha_h.mean(1)


class MotifCellGNN(nn.Module):
    def __init__(self, atom_dim, bond_dim, n_vocab, G, hidden=128, n_atom_layers=3, n_mc_layers=2, dropout=0.1, n_heads=1):
        super().__init__()
        self.hidden = hidden
        self.atom_gnn = AtomGNN(atom_dim, bond_dim, hidden, n_atom_layers, dropout)
        self.vocab_emb = nn.Embedding(n_vocab, hidden)
        self.pool_lin = nn.Linear(hidden, hidden)
        self.size_lin = nn.Linear(1, hidden)
        self.cell = CellEncoder(G, hidden, dropout)
        self.layers = nn.ModuleList([MotifCellLayer(hidden, dropout, n_heads) for _ in range(n_mc_layers)])
        self.head = nn.Sequential(nn.Linear(4 * hidden, 2 * hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def encode_drugs(self, bank):
        h = self.atom_gnn(bank.x, bank.edge_index, bank.edge_attr)
        g = scatter(h, bank.atom_drug, dim=0, dim_size=bank.D, reduce="mean")
        mp = scatter(h, bank.atom_motif, dim=0, dim_size=bank.Nm, reduce="mean")
        h0 = self.pool_lin(mp) + self.vocab_emb(bank.motif_vid) + self.size_lin(torch.log(bank.motif_size).unsqueeze(-1))
        h0 = torch.cat([h0, torch.zeros(1, self.hidden, device=h0.device, dtype=h0.dtype)], 0)
        idx = bank.pad_index.clone(); idx[idx < 0] = bank.Nm
        H0 = torch.index_select(h0, 0, idx.reshape(-1)).view(bank.D, bank.M, self.hidden)
        return {"g": g, "H0": H0, "mask": bank.mask.float(), "adj": bank.adj}

    def forward_pairs(self, enc, d_idx, xcell, H0=None, mmask=None, return_attn=False, M=None):
        """enc: encode_drugs output. d_idx [B] drug indices into the bank. xcell [B, 3G] standardised omics.
        H0: optional override of the motif inputs [B,M,H] (attribution). mmask: optional soft motif mask [B,M].
        M: optional truncation of the motif axis (exact when every drug in the batch has <= M motifs)."""
        Hm = torch.index_select(enc["H0"], 0, d_idx) if H0 is None else H0
        mask = torch.index_select(enc["mask"], 0, d_idx)
        if mmask is not None:
            mask = mask * mmask
        adj = torch.index_select(enc["adj"], 0, d_idx)
        if M is not None and M < Hm.shape[1]:
            Hm, mask, adj = Hm[:, :M], mask[:, :M], adj[:, :M, :M]
        logmask = torch.log(mask.clamp_min(1e-12))
        Hm = Hm * mask.unsqueeze(-1)
        hc0 = self.cell(xcell)
        hc = hc0
        alphas = []
        for layer in self.layers:
            Hm, hc, alpha = layer(Hm, hc, adj, mask, logmask)
            alphas.append(alpha)
        ctx = (alphas[-1].unsqueeze(-1) * Hm).sum(1)
        y = self.head(torch.cat([hc, ctx, torch.index_select(enc["g"], 0, d_idx), hc0], -1)).squeeze(-1)
        return (y, alphas) if return_attn else y


class LateFusion(nn.Module):
    """Drug GNN pooled embedding || cell embedding -> MLP (no motif nodes, no cross-attention)."""

    def __init__(self, atom_dim, bond_dim, n_vocab, G, hidden=128, n_atom_layers=3, dropout=0.1, **kw):
        super().__init__()
        self.hidden = hidden
        self.atom_gnn = AtomGNN(atom_dim, bond_dim, hidden, n_atom_layers, dropout)
        self.cell = CellEncoder(G, hidden, dropout)
        self.head = nn.Sequential(nn.Linear(2 * hidden, 2 * hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def encode_drugs(self, bank):
        h = self.atom_gnn(bank.x, bank.edge_index, bank.edge_attr)
        return {"g": scatter(h, bank.atom_drug, dim=0, dim_size=bank.D, reduce="mean")}

    def forward_pairs(self, enc, d_idx, xcell, H0=None, mmask=None, return_attn=False, M=None):
        hc = self.cell(xcell)
        y = self.head(torch.cat([torch.index_select(enc["g"], 0, d_idx), hc], -1)).squeeze(-1)
        return (y, []) if return_attn else y


def build_model(kind, atom_dim, bond_dim, n_vocab, G, hidden=128, dropout=0.1, n_heads=1):
    if kind == "ccmg":
        return MotifCellGNN(atom_dim, bond_dim, n_vocab, G, hidden=hidden, dropout=dropout, n_heads=n_heads)
    if kind == "latefusion":
        return LateFusion(atom_dim, bond_dim, n_vocab, G, hidden=hidden, dropout=dropout)
    raise ValueError(kind)


def attention_entropy(alpha, mask):
    """mean normalised entropy of the last-layer motif-cell attention over a batch ([B,M] alpha and 0/1 mask)."""
    n = mask.sum(1).clamp(min=2)
    ent = -(alpha * torch.log(alpha.clamp_min(1e-12)) * mask).sum(1) / torch.log(n)
    return ent.mean()


BUCKETS = [(0, 6), (7, 10), (11, 10 ** 6)]


def forward_bucketed(model, enc, bank, d_idx, xcell, return_entropy=False):
    """Exact forward that pads each motif-count bucket only to its own maximum (about 2x faster than padding
    every pair to the largest drug). With return_entropy the batch-mean normalised entropy of the last-layer
    attention is returned as well (zero for models without attention)."""
    nm = bank.n_motifs[d_idx]
    out = torch.empty(len(d_idx), device=xcell.device, dtype=xcell.dtype)
    ent, n_ent = 0.0, 0
    for lo, hi in BUCKETS:
        sel = (nm >= lo) & (nm <= hi)
        if sel.any():
            if return_entropy:
                y, alphas = model.forward_pairs(enc, d_idx[sel], xcell[sel], M=min(hi, bank.M), return_attn=True)
                if alphas:
                    m = torch.index_select(enc["mask"], 0, d_idx[sel])[:, :alphas[-1].shape[1]]
                    ent = ent + attention_entropy(alphas[-1], m) * int(sel.sum()); n_ent += int(sel.sum())
                out[sel] = y
            else:
                out[sel] = model.forward_pairs(enc, d_idx[sel], xcell[sel], M=min(hi, bank.M))
    if return_entropy:
        return out, (ent / n_ent if n_ent else torch.zeros((), device=xcell.device))
    return out
