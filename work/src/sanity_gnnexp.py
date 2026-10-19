"""Planted-importance sanity check of the motif explainers (GNNExplainer-style masks in particular).

Synthetic labels with a known sparse mechanism are attached to the real inputs: y = BETA * [drug contains the planted
motif K] + cell effect + noise, where the cell effect is the standardised mean expression of 20 random panel genes.
The default MotifCell-GNN is trained on these labels (random split, seed 0), and every motif explainer is run on the
probe pairs whose drug contains K. Recovery = fraction of pairs whose top-ranked motif is the planted one; chance =
mean 1/n_motifs. The deletion counterfactual of the planted motif confirms that the model uses it.
Usage: python sanity_gnnexp.py [--epochs 15]  -> results/summary/gnnexp_sanity.csv
"""
import argparse, os, pickle, sys, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from model import forward_bucketed
from train import load_data, make_model
from explain import Explainer

BETA = 3.0
N_PLANTS = 3


def main():
    p = argparse.ArgumentParser(); p.add_argument("--epochs", type=int, default=15); a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data("brics", "random", 0, device)
    G = data["graphs"]; graphs = G["graphs"]; bank = data["bank"]
    rng = np.random.default_rng(0)
    # planted vocabulary ids: frequent motifs present in 15-40 drugs
    counts = {}
    for g in graphs:
        for v in set(g["motif_vid"].tolist()):
            if v > 0: counts[v] = counts.get(v, 0) + 1
    cands = [v for v, c in counts.items() if 15 <= c <= 40]
    plants = list(rng.choice(cands, N_PLANTS, replace=False))
    X = data["X"]; n_genes = data["n_genes"]
    cell_genes = rng.choice(n_genes, 20, replace=False)
    cell_eff = X[:, cell_genes].mean(1); cell_eff = (cell_eff - cell_eff.mean()) / cell_eff.std()
    rows = []
    for K in plants:
        has = np.array([K in set(g["motif_vid"].tolist()) for g in graphs])
        y_syn = BETA * has[data["pair_drug"]].astype(np.float32) + cell_eff[torch.as_tensor(data["pair_cell"], device=device)].cpu().numpy() \
            + 0.3 * rng.normal(size=len(data["pair_drug"])).astype(np.float32)
        y_t = torch.as_tensor(y_syn, device=device)
        seed_all(0)
        model = make_model("ccmg", data, 128, device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        pd_t = torch.as_tensor(data["pair_drug"], device=device); pc_t = torch.as_tensor(data["pair_cell"], device=device)
        tr = torch.as_tensor(data["tr"], device=device); gen = torch.Generator(device="cpu"); gen.manual_seed(0)
        t0 = time.time()
        for ep in range(a.epochs):
            model.train(); perm = tr[torch.randperm(len(tr), generator=gen).to(device)]
            for i in range(0, len(perm), 1024):
                b = perm[i:i + 1024]
                enc = model.encode_drugs(bank)
                loss = torch.mean((forward_bucketed(model, enc, bank, pd_t[b], X[pc_t[b]]) - y_t[b]) ** 2)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        model.eval()
        for prm in model.parameters(): prm.requires_grad_(False)
        with torch.no_grad():
            enc = {k: v.detach() for k, v in model.encode_drugs(bank).items()}
        # probe pairs whose drug contains K and has >= 2 motifs
        probe = data["probe"]; dn = data["pair_drug"][probe]; cn = data["pair_cell"][probe]
        keep = np.array([has[d] and graphs[d]["n_motifs"] >= 2 for d in dn])
        d_idx = torch.as_tensor(dn[keep], device=device); c_idx = torch.as_tensor(cn[keep], device=device)
        ex = Explainer(model, data, enc, d_idx, bank)
        y0, scores = ex.motif_all(X[c_idx], np.random.default_rng(1))
        # validation fit on synthetic labels (test pairs)
        te = data["te"]; pte = forward_bucketed(model, enc, bank, torch.as_tensor(data["pair_drug"][te], device=device), X[torch.as_tensor(data["pair_cell"][te], device=device)]).detach().cpu().numpy()
        r_te = float(np.corrcoef(pte, y_syn[te])[0, 1])
        planted = [[m for m in range(graphs[d]["n_motifs"]) if graphs[d]["motif_vid"][m] == K] for d in dn[keep]]
        nm = np.array([graphs[d]["n_motifs"] for d in dn[keep]])
        # deletion effect of the planted motif
        from chem import delete_motif
        from model import Bank
        edited, idx = [], []
        for i, d in enumerate(dn[keep]):
            m = planted[i][0]; edited.append(delete_motif(graphs[d], m)); idx.append(i)
        bank_cf = Bank(edited, device)
        with torch.no_grad():
            enc_cf = {k: v.detach() for k, v in model.encode_drugs(bank_cf).items()}
            ydel = model.forward_pairs(enc_cf, torch.arange(len(edited), device=device), X[c_idx]).cpu().numpy()
        ddel = ydel - y0
        for name, A in scores.items():
            top = np.nanargmax(np.where(np.isnan(A), -np.inf, np.abs(A)), 1)
            hit = np.array([top[i] in planted[i] for i in range(len(top))])
            rows.append({"planted_vid": int(K), "motif_key": G["vocab_keys"][K], "n_drugs_with_motif": int(has.sum()), "n_pairs": int(keep.sum()),
                         "explainer": name, "top1_recovery": float(hit.mean()), "chance": float(np.mean(1.0 / nm)), "r_test_synthetic": r_te,
                         "mean_delta_del_planted": float(ddel.mean()), "train_s": time.time() - t0})
        print(f"K={K} ({G['vocab_keys'][K]}): drugs {has.sum()}, pairs {keep.sum()}, r_test {r_te:.3f}, delta_del planted {ddel.mean():.2f}; " +
              ", ".join(f"{n} {r['top1_recovery']:.2f}" for n, r in zip(scores.keys(), rows[-len(scores):])), flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(SUMMARY, "gnnexp_sanity.csv"), index=False)
    print(df.groupby("explainer")[["top1_recovery", "chance"]].mean().round(3))


if __name__ == "__main__":
    main()
