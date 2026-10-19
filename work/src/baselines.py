"""DrEval-style baselines: naive mean-effect predictor and a tuned LightGBM tree ensemble.
Usage: python baselines.py [--split random] [--split_seed 0] [--seeds 0 1 2 3 4] [--tune 6] [--threads 4]
Results: results/baselines/<kind>_<split>_ss<s>_s<seed>.json (+ preds npz); lgbm_best_<split>.json (tuned config)
"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

BASE = os.path.join(RESULTS, "baselines")
os.makedirs(BASE, exist_ok=True)
N_CNV_TOP = 300
FP_BITS = 1024


def naive(pair_drug, pair_cell, y, tr, te, n_drugs, n_cells):
    """mu + drug effect + cell effect fitted on train; an effect is 0 for a drug / cell unseen in training."""
    mu = y[tr].mean()
    sd = np.bincount(pair_drug[tr], weights=y[tr] - mu, minlength=n_drugs); nd = np.bincount(pair_drug[tr], minlength=n_drugs)
    sc = np.bincount(pair_cell[tr], weights=y[tr] - mu, minlength=n_cells); nc = np.bincount(pair_cell[tr], minlength=n_cells)
    ad = np.where(nd > 0, sd / np.maximum(nd, 1), 0.0)
    # cell effect from residuals after the drug effect (one backfitting pass)
    res = y[tr] - mu - ad[pair_drug[tr]]
    sc = np.bincount(pair_cell[tr], weights=res, minlength=n_cells)
    ac = np.where(nc > 0, sc / np.maximum(nc, 1), 0.0)
    return mu + ad[pair_drug[te]] + ac[pair_cell[te]]


def features(d, meta, train_cells):
    from rdkit import Chem
    from chem import morgan_fp
    X, _ = build_cell_features(d, train_cells)
    G = d["expr"].shape[1]
    genes = meta["genes"]; tset = set(meta["target_genes"])
    tidx = [i for i, g in enumerate(genes) if g in tset]
    cnv_var = np.var(X[train_cells, 2 * G:], axis=0)
    cidx = np.argsort(-cnv_var)[:N_CNV_TOP]
    Xc = np.concatenate([X[:, :G], X[:, G:2 * G][:, tidx], X[:, 2 * G:][:, cidx]], 1).astype(np.float32)
    FP = np.stack([morgan_fp(Chem.MolFromSmiles(dr["smiles"]), FP_BITS) for dr in meta["drugs"]]).astype(np.float32)
    return Xc, FP


def design(Xc, FP, pair_drug, pair_cell, idx):
    return np.concatenate([FP[pair_drug[idx]], Xc[pair_cell[idx]]], 1)


SPACE = [
    dict(num_leaves=63, learning_rate=0.05, feature_fraction=0.5, bagging_fraction=0.8, min_child_samples=50, lambda_l2=1.0),
    dict(num_leaves=127, learning_rate=0.05, feature_fraction=0.3, bagging_fraction=0.8, min_child_samples=100, lambda_l2=5.0),
    dict(num_leaves=255, learning_rate=0.03, feature_fraction=0.3, bagging_fraction=0.7, min_child_samples=100, lambda_l2=10.0),
    dict(num_leaves=31, learning_rate=0.1, feature_fraction=0.7, bagging_fraction=0.9, min_child_samples=20, lambda_l2=0.0),
    dict(num_leaves=127, learning_rate=0.1, feature_fraction=0.5, bagging_fraction=0.8, min_child_samples=200, lambda_l2=1.0),
    dict(num_leaves=63, learning_rate=0.03, feature_fraction=0.2, bagging_fraction=0.6, min_child_samples=50, lambda_l2=20.0),
]


def fit_lgbm(cfg, Xtr, ytr, Xva, yva, seed, threads, rounds=3000):
    import lightgbm as lgb
    params = dict(objective="regression", metric="rmse", verbosity=-1, num_threads=threads, seed=seed,
                  bagging_freq=1, max_bin=63, **cfg)
    dtr = lgb.Dataset(Xtr, ytr, free_raw_data=True); dva = lgb.Dataset(Xva, yva, reference=dtr, free_raw_data=True)
    m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="random", choices=SPLITS)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--tune", type=int, default=6)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--skip_lgbm", action="store_true")
    a = p.parse_args()
    d = np.load(os.path.join(DATA, "dataset.npz")); meta = jload(os.path.join(DATA, "meta.json"))
    sp = np.load(os.path.join(DATA, "splits.npz"))
    tr, va, te = (sp[f"{a.split}_ss{a.split_seed}_{q}"] for q in ("train", "val", "test"))
    probe = sp["probe_pairs"]; pr = np.isin(te, probe)
    pd_, pc, y = d["pair_drug"].astype(np.int64), d["pair_cell"].astype(np.int64), d["y"].astype(np.float32)
    nD, nC = len(meta["drugs"]), len(meta["cells"])
    tag = f"{a.split}_ss{a.split_seed}"
    if all(os.path.exists(os.path.join(BASE, f"lgbm_{tag}_s{s}.json")) for s in a.seeds) and             os.path.exists(os.path.join(BASE, f"naive_{tag}_s0.json")):
        print("already done", tag); return

    def save(kind, seed, yhat, extra):
        met = {"test": regression_metrics(y[te], yhat, pd_[te], pc[te]),
               "probe": regression_metrics(y[te][pr], yhat[pr], pd_[te][pr], pc[te][pr]),
               "name": f"{kind}_{tag}_s{seed}", "model": kind, "vocab": "none", "split": a.split, "seed": seed,
               "split_seed": a.split_seed}
        met.update(extra)
        jdump(met, os.path.join(BASE, f"{kind}_{tag}_s{seed}.json"))
        np.savez_compressed(os.path.join(BASE, f"{kind}_{tag}_s{seed}_preds.npz"), idx=te, y=y[te], yhat=yhat, is_probe=pr)
        print(f"{kind} {tag} s{seed}: rmse {met['test']['rmse']:.4f} r {met['test']['pearson']:.4f} "
              f"per-drug r {met['test']['per_drug_pearson']:.4f} probe rmse {met['probe']['rmse']:.4f}", flush=True)

    yn = naive(pd_, pc, y, tr, te, nD, nC)
    save("naive", 0, yn, {})
    if a.skip_lgbm:
        return
    t0 = time.time()
    Xc, FP = features(d, meta, np.unique(pc[tr]))
    Xtr, Xva, Xte = design(Xc, FP, pd_, pc, tr), design(Xc, FP, pd_, pc, va), design(Xc, FP, pd_, pc, te)
    print(f"design {Xtr.shape} built in {time.time() - t0:.0f}s", flush=True)
    best_path = os.path.join(BASE, f"lgbm_best_{a.split}.json")
    best = None
    if os.path.exists(best_path):
        try:
            best = jload(best_path)
            assert "cfg" in best
        except Exception:
            best = None; os.remove(best_path)
    if best is None:
        trials = []
        for i, cfg in enumerate(SPACE[:a.tune]):
            t1 = time.time()
            m = fit_lgbm(cfg, Xtr, y[tr], Xva, y[va], 0, a.threads)
            trials.append({"cfg": cfg, "val_rmse": float(m.best_score["valid_0"]["rmse"]), "iters": int(m.best_iteration),
                           "t": time.time() - t1})
            print(f"tune {i}: val rmse {trials[-1]['val_rmse']:.4f} iters {m.best_iteration} {time.time() - t1:.0f}s", flush=True)
        best = dict(min(trials, key=lambda t: t["val_rmse"])); best["trials"] = trials
        jdump(best, best_path)
    for seed in a.seeds:
        t1 = time.time()
        m = fit_lgbm(best["cfg"], Xtr, y[tr], Xva, y[va], seed, a.threads)
        yh = m.predict(Xte, num_iteration=m.best_iteration).astype(np.float32)
        save("lgbm", seed, yh, {"cfg": best["cfg"], "iters": int(m.best_iteration), "val_rmse": float(m.best_score["valid_0"]["rmse"]),
                                 "train_time_s": time.time() - t1, "n_features": int(Xtr.shape[1])})
    print("BASELINES_DONE", tag, flush=True)


if __name__ == "__main__":
    main()
