"""Explanations, perturbation variants and input-space counterfactuals for one trained CCMG run on the probe pairs.
Usage: python explain.py --run ccmg_brics_random_ss0_s0
Output: results/explain/<run>/scores.npz, summary.json

Motif-level explainers (scores [P, M], NaN on padding): attn (last-layer motif-cell attention), rollout (attention
rollout through the motif-cell layers), ig (integrated gradients w.r.t. the motif node inputs), mask (occlusion of
one motif node in the model), gnnexp (GNNExplainer-style learned soft node mask). Controls: size (atom count), random.
Gene-level explainers (scores [P, G]): ig, sal (gradient x input), mask (occlusion of the gene's three omics
entries), gnnexp (learned gene mask).
Input-perturbation variant: every explainer recomputed after Gaussian noise on the expression and copy-number inputs.
Counterfactuals: delta_del[p, m] = f(drug without motif m, cell) - f(drug, cell) (atoms removed, re-featurised with
RDKit when the remainder sanitises); delta_swap[p, m] = mean over donors of f(motif m replaced by a size-matched motif
from another drug) - f; delta_kd[p, g] = f after knocking the expression of gene g down to -2 SD - f.
"""
import argparse, os, pickle, signal, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from model import Bank
from chem import delete_motif, swap_motif
from train import load_data, make_model


import re


def parse_run(name):
    """model, vocab, split, split_seed, seed, tag from a run name such as ccmg_brics_unseen_drug_ss0_s3[_h4]."""
    m = re.match(r"^(ccmg|latefusion)_(brics|recap|none)_(random|unseen_drug|unseen_cell)_ss(\d+)_s(\d+)(?:_(.+))?$", name)
    if not m:
        raise ValueError(name)
    return m.group(1), m.group(2), m.group(3), int(m.group(4)), int(m.group(5)), m.group(6) or ""


class Explainer:
    def __init__(self, model, data, enc, d_idx, bank):
        self.m, self.data, self.enc, self.d_idx, self.bank = model, data, enc, d_idx, bank
        self.P, self.M = len(d_idx), bank.M
        self.valid = torch.index_select(enc["mask"], 0, d_idx)              # [P, M]
        self.nvalid = self.valid.sum(1)
        self.G = data["n_genes"]

    def f(self, X, chunk=1250, d_idx=None, **kw):
        """prediction (and attention per layer) for rows of X; d_idx defaults to the probe drugs (one row per pair)."""
        d_idx = self.d_idx if d_idx is None else d_idx
        n = X.shape[0]
        out, att = [], []
        for i in range(0, n, chunk):
            sl = slice(i, i + chunk)
            kw_i = {k: (v[sl] if torch.is_tensor(v) and v.shape[0] == n else v) for k, v in kw.items()}
            y, al = self.m.forward_pairs(self.enc, d_idx[sl], X[sl], return_attn=True, **kw_i)
            out.append(y); att.append(al)
        alphas = [torch.cat([a[l] for a in att]) for l in range(len(att[0]))] if att[0] else []
        return torch.cat(out), alphas

    def nanpad(self, s):
        s = s.clone(); s[self.valid == 0] = float("nan"); return s.cpu().numpy().astype(np.float32)

    # ---------------------------------------------------------------- motif level
    @torch.no_grad()
    def rollout(self, alphas):
        P, M = self.P, self.M
        adj = torch.index_select(self.enc["adj"], 0, self.d_idx) * self.valid.unsqueeze(1) * self.valid.unsqueeze(2)
        A_hat = torch.eye(M + 1, device=adj.device).expand(P, M + 1, M + 1).clone()
        for al in alphas:
            A = torch.zeros(P, M + 1, M + 1, device=adj.device)
            A[:, :M, :M] = adj
            A[:, :M, M] = self.valid                                         # every motif receives the cell node
            A[:, M, :M] = al
            rs = A.sum(-1, keepdim=True).clamp(min=1e-9)
            A = A / rs
            A = 0.5 * torch.eye(M + 1, device=adj.device) + 0.5 * A
            A_hat = torch.bmm(A, A_hat)
        return A_hat[:, M, :M]

    def ig_motif(self, X, y0, steps=IG_STEPS):
        H = torch.index_select(self.enc["H0"], 0, self.d_idx)
        total = torch.zeros_like(H)
        for k in range(steps):
            a = (k + 0.5) / steps
            Hk = (a * H).detach().requires_grad_(True)
            y, _ = self.f(X, H0=Hk)
            g, = torch.autograd.grad(y.sum(), Hk)
            total += g
        return (H * total / steps).sum(-1).detach()

    @torch.no_grad()
    def mask_motif(self, X, y0):
        out = torch.zeros(self.P, self.M, device=X.device)
        for m in range(self.M):
            mm = torch.ones(self.P, self.M, device=X.device); mm[:, m] = 0.0
            y, _ = self.f(X, mmask=mm)
            out[:, m] = y0 - y
        return out

    def gnnexp_motif(self, X, y0, steps=GNNEXP_STEPS, lam_size=0.1, lam_ent=0.05, lr=0.05):
        theta = torch.full((self.P, self.M), 2.0, device=X.device, requires_grad=True)
        opt = torch.optim.Adam([theta], lr=lr)
        for _ in range(steps):
            mk = torch.sigmoid(theta)
            y, _ = self.f(X, mmask=mk)
            mkv = mk * self.valid
            ent = -(mkv * torch.log(mk + 1e-9) + (1 - mk) * torch.log(1 - mk + 1e-9)) * self.valid
            loss = ((y - y0) ** 2).mean() + lam_size * (mkv.sum(1) / self.nvalid).mean() + lam_ent * (ent.sum(1) / self.nvalid).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return torch.sigmoid(theta).detach()

    def gnnexp2_motif(self, X, y0, steps=GNNEXP_STEPS, lam_size=0.02, lr=0.05):
        """variant: the learned mask scales only the motif node inputs (no attention or adjacency gating), with a
        weaker size penalty and no entropy term."""
        H = torch.index_select(self.enc["H0"], 0, self.d_idx)
        theta = torch.full((self.P, self.M), 2.0, device=X.device, requires_grad=True)
        opt = torch.optim.Adam([theta], lr=lr)
        for _ in range(steps):
            mk = torch.sigmoid(theta)
            y, _ = self.f(X, H0=H * mk.unsqueeze(-1))
            loss = ((y - y0) ** 2).mean() + lam_size * ((mk * self.valid).sum(1) / self.nvalid).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return torch.sigmoid(theta).detach()

    def motif_all(self, X, rng):
        with torch.no_grad():
            y0, alphas = self.f(X)
        out = {"attn": alphas[-1], "attn_mean": torch.stack(alphas).mean(0), "rollout": self.rollout(alphas),
               "ig": self.ig_motif(X, y0), "mask": self.mask_motif(X, y0), "gnnexp": self.gnnexp_motif(X, y0),
               "gnnexp2": self.gnnexp2_motif(X, y0)}
        out = {k: self.nanpad(v) for k, v in out.items()}
        out["random"] = self.nanpad(torch.as_tensor(rng.random((self.P, self.M)), device=X.device, dtype=torch.float32))
        return y0.detach().cpu().numpy(), out

    # ---------------------------------------------------------------- gene level
    def _per_gene(self, s):
        G = self.G
        return (s[:, :G] + s[:, G:2 * G] + s[:, 2 * G:]).detach().cpu().numpy().astype(np.float32)

    def ig_gene(self, X, steps=IG_STEPS):
        total = torch.zeros_like(X)
        for k in range(steps):
            a = (k + 0.5) / steps
            Xk = (a * X).detach().requires_grad_(True)
            y, _ = self.f(Xk)
            g, = torch.autograd.grad(y.sum(), Xk)
            total += g
        return self._per_gene(X * total / steps)

    def sal_gene(self, X):
        Xk = X.detach().clone().requires_grad_(True)
        y, _ = self.f(Xk)
        g, = torch.autograd.grad(y.sum(), Xk)
        return self._per_gene(X * g)

    @torch.no_grad()
    def mask_gene(self, X, y0, value=None, expr_only=False, genes_per_batch=8):
        """Occlusion: set the gene's omics entries to 0 (training mean / no mutation); value=-2 & expr_only -> knockdown.
        Several genes are evaluated per forward pass (rows = genes x pairs)."""
        G = self.G
        out = np.zeros((self.P, G), np.float32)
        y0 = y0.reshape(-1)
        for g0 in range(0, G, genes_per_batch):
            gs = list(range(g0, min(G, g0 + genes_per_batch)))
            Xr = X.repeat(len(gs), 1)
            for j, g in enumerate(gs):
                cols = [g] if expr_only else [g, G + g, 2 * G + g]
                rows = slice(j * self.P, (j + 1) * self.P)
                Xr[rows, cols] = 0.0 if value is None else value
            y, _ = self.f(Xr, chunk=5000, d_idx=self.d_idx.repeat(len(gs)))
            y = y.view(len(gs), self.P)
            d = (y0.unsqueeze(0) - y) if value is None else (y - y0.unsqueeze(0))
            out[:, gs] = d.t().cpu().numpy()
        return out

    def gnnexp_gene(self, X, y0, steps=GNNEXP_STEPS, lam_size=0.1, lam_ent=0.05, lr=0.05):
        G = self.G
        theta = torch.full((self.P, G), 2.0, device=X.device, requires_grad=True)
        opt = torch.optim.Adam([theta], lr=lr)
        for _ in range(steps):
            mk = torch.sigmoid(theta)
            Xm = X * mk.repeat(1, 3)
            y, _ = self.f(Xm)
            ent = -(mk * torch.log(mk + 1e-9) + (1 - mk) * torch.log(1 - mk + 1e-9))
            loss = ((y - y0) ** 2).mean() + lam_size * mk.mean() + lam_ent * ent.mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return torch.sigmoid(theta).detach().cpu().numpy().astype(np.float32)

    def gene_all(self, X, y0):
        y0t = torch.as_tensor(y0, device=X.device)
        return {"ig": self.ig_gene(X), "sal": self.sal_gene(X), "mask": self.mask_gene(X, y0t), "gnnexp": self.gnnexp_gene(X, y0t)}


def build_counterfactuals(graphs, probe_drugs, rng):
    """Edited graphs for every (probe drug, motif): one deletion and N_SWAP_DONORS size-matched swaps."""
    sizes = []
    for di, g in enumerate(graphs):
        cnt = np.bincount(g["a2m"], minlength=g["n_motifs"])
        for m in range(g["n_motifs"]):
            sizes.append((di, m, int(cnt[m])))
    edited, index = [], {}          # index[(d, m)] = {"del": k, "swap": [k1, k2, k3], "chem_valid": bool}
    for d in probe_drugs:
        g = graphs[d]
        cnt = np.bincount(g["a2m"], minlength=g["n_motifs"])
        for m in range(g["n_motifs"]):
            gd = delete_motif(g, m)
            rec = {"del": len(edited), "chem_valid": bool(gd["chem_valid"]), "swap": []}
            edited.append(gd)
            for tol in (2, 5, 10 ** 6):
                cands = [(di, mi) for di, mi, s in sizes if di != d and abs(s - cnt[m]) <= tol]
                if len(cands) >= N_SWAP_DONORS:
                    break
            pick = rng.choice(len(cands), N_SWAP_DONORS, replace=False)
            for k in pick:
                di, mi = cands[k]
                rec["swap"].append(len(edited)); edited.append(swap_motif(g, m, graphs[di], mi))
            index[(d, m)] = rec
    return edited, index


def main():
    for s in ("SIGINT", "SIGBREAK"):
        if hasattr(signal, s):
            signal.signal(getattr(signal, s), signal.SIG_IGN)
    p = argparse.ArgumentParser(); p.add_argument("--run", required=True); a = p.parse_args()
    t0 = time.time()
    kind, vocab, split, ss, seed, tag = parse_run(a.run)
    assert kind == "ccmg"
    out_dir = os.path.join(EXPL, a.run); os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(vocab, split, ss, device)
    ck = torch.load(os.path.join(RUNS, a.run, "model.pt"), map_location=device, weights_only=False)
    ca = ck["args"]
    model = make_model("ccmg", data, ca["hidden"], device, ca.get("dropout", 0.1), ca.get("heads", 1))
    model.load_state_dict(ck["state"]); model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    probe = data["probe"]
    d_np, c_np = data["pair_drug"][probe], data["pair_cell"][probe]
    d_idx = torch.as_tensor(d_np, device=device); c_idx = torch.as_tensor(c_np, device=device)
    X = data["X"][c_idx]
    with torch.no_grad():
        enc = {k: v.detach() for k, v in model.encode_drugs(data["bank"]).items()}
    ex = Explainer(model, data, enc, d_idx, data["bank"])
    rng = np.random.default_rng(seed)
    print(f"{a.run}: {ex.P} probe pairs, M={ex.M}, G={ex.G}", flush=True)

    y0, motif_clean = ex.motif_all(X, rng); print(f"motif explainers done {time.time() - t0:.0f}s", flush=True)
    gene_clean = ex.gene_all(X, y0); print(f"gene explainers done {time.time() - t0:.0f}s", flush=True)
    # input perturbation (expression and copy-number blocks)
    G = ex.G
    noise = torch.as_tensor(rng.normal(0, PERTURB_SIGMA, size=(ex.P, 3 * G)), device=device, dtype=torch.float32)
    noise[:, G:2 * G] = 0.0
    Xp = X + noise
    yp, motif_pert = ex.motif_all(Xp, np.random.default_rng(seed + 100))
    gene_pert = ex.gene_all(Xp, yp); print(f"perturbed explainers done {time.time() - t0:.0f}s", flush=True)
    # counterfactuals: motif deletion / swap
    graphs = data["graphs"]["graphs"]
    probe_drugs = sorted(set(d_np.tolist()))
    edited, index = build_counterfactuals(graphs, probe_drugs, np.random.default_rng(12345))
    bank_cf = Bank(edited, device)
    with torch.no_grad():
        enc_cf = {k: v.detach() for k, v in model.encode_drugs(bank_cf).items()}
    delta_del = np.full((ex.P, ex.M), np.nan, np.float32); delta_swap = np.full((ex.P, ex.M), np.nan, np.float32)
    chem_valid = np.zeros((ex.P, ex.M), bool)
    pair_rows, cf_rows, kind_rows = [], [], []     # flat evaluation list
    for pi, d in enumerate(d_np):
        for m in range(graphs[d]["n_motifs"]):
            rec = index[(d, m)]
            pair_rows.append(pi); cf_rows.append(rec["del"]); kind_rows.append(("del", m))
            for k in rec["swap"]:
                pair_rows.append(pi); cf_rows.append(k); kind_rows.append(("swap", m))
            chem_valid[pi, m] = rec["chem_valid"]
    pair_rows = np.array(pair_rows); cf_rows = np.array(cf_rows)
    ycf = np.zeros(len(pair_rows), np.float32)
    with torch.no_grad():
        for i in range(0, len(pair_rows), 2048):
            sl = slice(i, i + 2048)
            ycf[sl] = model.forward_pairs(enc_cf, torch.as_tensor(cf_rows[sl], device=device), X[torch.as_tensor(pair_rows[sl], device=device)]).cpu().numpy()
    acc = {}
    for (kd, m), pi, yv in zip(kind_rows, pair_rows, ycf):
        if kd == "del":
            delta_del[pi, m] = yv - y0[pi]
        else:
            acc.setdefault((pi, m), []).append(yv - y0[pi])
    for (pi, m), v in acc.items():
        delta_swap[pi, m] = float(np.mean(v))
    print(f"motif counterfactuals done ({len(edited)} edited graphs) {time.time() - t0:.0f}s", flush=True)
    # counterfactual: gene expression knockdown
    delta_kd = ex.mask_gene(X, torch.as_tensor(y0, device=device), value=-2.0, expr_only=True)
    print(f"gene knockdown done {time.time() - t0:.0f}s", flush=True)

    motif_size = np.full((ex.P, ex.M), np.nan, np.float32)
    for pi, d in enumerate(d_np):
        cnt = np.bincount(graphs[d]["a2m"], minlength=graphs[d]["n_motifs"])
        motif_size[pi, :len(cnt)] = cnt
    arrays = {"probe_idx": probe, "d_idx": d_np, "c_idx": c_np, "y_true": data["y"][probe], "y0": y0, "y_pert": yp,
              "delta_del": delta_del, "delta_swap": delta_swap, "chem_valid": chem_valid, "motif_size": motif_size,
              "delta_kd": delta_kd.astype(np.float16), "valid": ex.valid.cpu().numpy()}
    for k, v in motif_clean.items(): arrays[f"motif_{k}"] = v
    for k, v in motif_pert.items(): arrays[f"motifP_{k}"] = v
    for k, v in gene_clean.items(): arrays[f"gene_{k}"] = v.astype(np.float16)
    for k, v in gene_pert.items(): arrays[f"geneP_{k}"] = v.astype(np.float16)
    np.savez_compressed(os.path.join(out_dir, "scores.npz"), **arrays)
    summ = {"run": a.run, "model": kind, "vocab": vocab, "split": split, "split_seed": ss, "seed": seed, "tag": tag, "n_probe": int(ex.P),
            "M": int(ex.M), "G": int(ex.G), "n_edited_graphs": len(edited), "time_s": time.time() - t0,
            "mean_abs_delta_del": float(np.nanmean(np.abs(delta_del))), "mean_abs_delta_swap": float(np.nanmean(np.abs(delta_swap))),
            "mean_abs_delta_kd": float(np.abs(delta_kd).mean()), "frac_chem_valid": float(chem_valid[ex.valid.cpu().numpy() > 0].mean()),
            "pert_pred_shift_rmse": float(np.sqrt(np.mean((yp - y0) ** 2)))}
    jdump(summ, os.path.join(out_dir, "summary.json"))
    print(f"DONE {a.run} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
