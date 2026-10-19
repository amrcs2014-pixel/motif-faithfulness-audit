"""Train one neural model on one split and save checkpoint, test predictions and metrics.
Usage: python train.py --model ccmg|latefusion --vocab brics|recap --split random|unseen_drug|unseen_cell
                       --seed 0 [--split_seed 0] [--epochs 60] [--patience 8] [--batch 1024]
                       [--heads 1] [--attn_entropy 0] [--hidden 128] [--dropout 0.1] [--lr 1e-3] [--wd 1e-5]
Non-default architecture / optimisation settings are encoded in a run tag (e.g. h4, ent0.05, hid64-do0.3).
Result: results/runs/<name>/{metrics.json, preds_test.npz, model.pt, scaler.npz}
"""
import argparse, os, pickle, signal, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from model import Bank, build_model, forward_bucketed
from chem import ATOM_DIM, BOND_DIM

DEFAULTS = dict(heads=1, attn_entropy=0.0, hidden=128, dropout=0.1, lr=1e-3, wd=1e-5)


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ccmg", choices=MODELS)
    p.add_argument("--vocab", default="brics", choices=VOCABS)
    p.add_argument("--split", default="random", choices=SPLITS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--wd", type=float, default=DEFAULTS["wd"])
    p.add_argument("--hidden", type=int, default=DEFAULTS["hidden"])
    p.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    p.add_argument("--heads", type=int, default=DEFAULTS["heads"])
    p.add_argument("--attn_entropy", type=float, default=DEFAULTS["attn_entropy"],
                   help="weight of an attention-entropy bonus in the loss (flattens the motif-cell attention)")
    return p.parse_args(argv)


def tag_of(a):
    """run tag from the non-default settings (empty for the main configuration)."""
    parts = []
    if a.heads != DEFAULTS["heads"]: parts.append(f"h{a.heads}")
    if a.attn_entropy != DEFAULTS["attn_entropy"]: parts.append(f"ent{a.attn_entropy:g}")
    if a.hidden != DEFAULTS["hidden"]: parts.append(f"hid{a.hidden}")
    if a.dropout != DEFAULTS["dropout"]: parts.append(f"do{a.dropout:g}")
    if a.lr != DEFAULTS["lr"]: parts.append(f"lr{a.lr:g}")
    if a.wd != DEFAULTS["wd"]: parts.append(f"wd{a.wd:g}")
    return "-".join(parts)


def load_data(vocab, split, split_seed, device):
    d = np.load(os.path.join(DATA, "dataset.npz"))
    meta = jload(os.path.join(DATA, "meta.json"))
    sp = np.load(os.path.join(DATA, "splits.npz"))
    tr, va, te = (sp[f"{split}_ss{split_seed}_{p}"] for p in ("train", "val", "test"))
    probe = sp["probe_pairs"]
    pair_drug, pair_cell, y = d["pair_drug"].astype(np.int64), d["pair_cell"].astype(np.int64), d["y"].astype(np.float32)
    train_cells = np.unique(pair_cell[tr])
    X, scaler = build_cell_features(d, train_cells)
    with open(os.path.join(DATA, f"graphs_{vocab}.pkl"), "rb") as f:
        G = pickle.load(f)
    bank = Bank(G["graphs"], device)
    return dict(meta=meta, graphs=G, bank=bank, X=torch.as_tensor(X, device=device), pair_drug=pair_drug,
                pair_cell=pair_cell, y=y, tr=tr, va=va, te=te, probe=probe,
                scaler=scaler, n_genes=d["expr"].shape[1])


def make_model(kind, data, hidden, device, dropout=DEFAULTS["dropout"], heads=DEFAULTS["heads"]):
    m = build_model(kind, ATOM_DIM, BOND_DIM, data["graphs"]["n_vocab"], data["n_genes"], hidden=hidden, dropout=dropout, n_heads=heads)
    return m.to(device)


@torch.no_grad()
def predict(model, data, idx, batch=4096):
    model.eval()
    enc = model.encode_drugs(data["bank"])
    out = []
    pd_ = torch.as_tensor(data["pair_drug"][idx], device=data["X"].device)
    pc = torch.as_tensor(data["pair_cell"][idx], device=data["X"].device)
    for i in range(0, len(idx), batch):
        out.append(forward_bucketed(model, enc, data["bank"], pd_[i:i + batch], data["X"][pc[i:i + batch]]).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


@torch.no_grad()
def val_entropy(model, data, idx):
    """mean normalised attention entropy on a fixed subset of validation pairs (0 for models without attention)."""
    model.eval()
    enc = model.encode_drugs(data["bank"])
    pd_ = torch.as_tensor(data["pair_drug"][idx], device=data["X"].device)
    pc = torch.as_tensor(data["pair_cell"][idx], device=data["X"].device)
    _, ent = forward_bucketed(model, enc, data["bank"], pd_, data["X"][pc], return_entropy=True)
    return float(ent)


def main(argv=None):
    for name in ("SIGINT", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), signal.SIG_IGN)
    a = parse(argv)
    tag = tag_of(a)
    name = run_name(a.model, a.vocab, a.split, a.seed, a.split_seed, tag)
    out_dir = os.path.join(RUNS, name); os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(a.seed)
    t0 = time.time()
    data = load_data(a.vocab, a.split, a.split_seed, device)
    model = make_model(a.model, data, a.hidden, device, a.dropout, a.heads)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{name}: train {len(data['tr'])} val {len(data['va'])} test {len(data['te'])}; params {n_par}; "
          f"drugs {data['bank'].D} motifs {data['bank'].Nm} M {data['bank'].M}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    y_t = torch.as_tensor(data["y"], device=device)
    pd_t = torch.as_tensor(data["pair_drug"], device=device); pc_t = torch.as_tensor(data["pair_cell"], device=device)
    tr = torch.as_tensor(data["tr"], device=device)
    ent_idx = data["va"][:4096]
    g = torch.Generator(device="cpu"); g.manual_seed(a.seed)
    best, best_state, bad, hist = float("inf"), None, 0, []
    use_ent = a.model == "ccmg" and a.attn_entropy > 0
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = tr[torch.randperm(len(tr), generator=g).to(device)]
        tot, nb = 0.0, 0
        for i in range(0, len(perm), a.batch):
            b = perm[i:i + a.batch]
            enc = model.encode_drugs(data["bank"])
            if use_ent:
                pred, ent = forward_bucketed(model, enc, data["bank"], pd_t[b], data["X"][pc_t[b]], return_entropy=True)
                loss = torch.mean((pred - y_t[b]) ** 2) - a.attn_entropy * ent
            else:
                pred = forward_bucketed(model, enc, data["bank"], pd_t[b], data["X"][pc_t[b]])
                loss = torch.mean((pred - y_t[b]) ** 2)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(b); nb += len(b)
        pv = predict(model, data, data["va"])
        v_rmse = float(np.sqrt(np.mean((pv - data["y"][data["va"]]) ** 2)))
        v_ent = val_entropy(model, data, ent_idx) if a.model == "ccmg" else float("nan")
        sched.step(v_rmse)
        hist.append({"epoch": ep, "train_loss": tot / nb, "val_rmse": v_rmse, "val_attn_entropy": v_ent,
                     "lr": opt.param_groups[0]["lr"], "t": time.time() - t0})
        print(f"ep {ep:3d} train_loss {tot / nb:.4f} val_rmse {v_rmse:.4f} attn_ent {v_ent:.3f} lr {opt.param_groups[0]['lr']:.1e} "
              f"{time.time() - t0:.0f}s", flush=True)
        if v_rmse < best - 1e-4:
            best, bad = v_rmse, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)
    te = data["te"]
    pt = predict(model, data, te)
    yt = data["y"][te]
    met = {"test": regression_metrics(yt, pt, data["pair_drug"][te], data["pair_cell"][te])}
    pr = np.isin(te, data["probe"])
    met["probe"] = regression_metrics(yt[pr], pt[pr], data["pair_drug"][te][pr], data["pair_cell"][te][pr])
    pv = predict(model, data, data["va"])
    met["val"] = regression_metrics(data["y"][data["va"]], pv)
    met.update({"name": name, "model": a.model, "vocab": a.vocab if a.model == "ccmg" else "none", "split": a.split,
                "seed": a.seed, "split_seed": a.split_seed, "tag": tag, "epochs_run": len(hist), "best_val_rmse": best,
                "params": n_par, "train_time_s": time.time() - t0, "history": hist, "hidden": a.hidden, "dropout": a.dropout,
                "heads": a.heads, "attn_entropy": a.attn_entropy, "lr": a.lr, "wd": a.wd,
                "final_val_attn_entropy": val_entropy(model, data, ent_idx) if a.model == "ccmg" else float("nan")})
    jdump(met, os.path.join(out_dir, "metrics.json"))
    np.savez_compressed(os.path.join(out_dir, "preds_test.npz"), idx=te, y=yt, yhat=pt, is_probe=pr)
    np.savez(os.path.join(out_dir, "scaler.npz"), **data["scaler"])
    torch.save({"state": best_state, "args": vars(a)}, os.path.join(out_dir, "model.pt"))
    print(f"DONE {name}: test rmse {met['test']['rmse']:.4f} pearson {met['test']['pearson']:.4f} "
          f"spearman {met['test']['spearman']:.4f} per-drug r {met['test']['per_drug_pearson']:.4f}; "
          f"probe rmse {met['probe']['rmse']:.4f} r {met['probe']['pearson']:.4f}; {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
