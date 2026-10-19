"""Shared paths, constants and helpers for the motif-faithfulness audit pipeline."""
import json, os, random
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "raw")
DEPMAP = os.path.join(RAW, "depmap")
RESULTS = os.path.join(ROOT, "results")
RUNS = os.path.join(RESULTS, "runs")
EXPL = os.path.join(RESULTS, "explain")
SUMMARY = os.path.join(RESULTS, "summary")
FIGS = os.path.join(ROOT, "figs")
PAPER = os.path.join(ROOT, "paper")
LOGS = os.path.join(ROOT, "logs")
for d in (DATA, RESULTS, RUNS, EXPL, SUMMARY, FIGS, PAPER, LOGS):
    os.makedirs(d, exist_ok=True)

SPLITS = ["random", "unseen_drug", "unseen_cell"]
SEEDS = [0, 1, 2, 3, 4]            # training seeds (initialisation + batch order) on the fixed split (split seed 0)
SPLIT_SEEDS = [0, 1, 2, 3, 4]      # data-resampling seeds (split seed = training seed) for the sensitivity set
VOCABS = ["brics", "recap"]
MODELS = ["ccmg", "latefusion"]    # neural models trained on the GPU
MOTIF_EXPLAINERS = ["attn", "rollout", "ig", "mask", "gnnexp"]
GENE_EXPLAINERS = ["ig", "sal", "mask", "gnnexp"]

N_TOPVAR_GENES = 1000      # highly variable expression genes added to the target panel
PROBE_DRUGS = 30           # probe drugs (explanation evaluation set)
PROBE_CELLS = 100          # probe cells
MAX_PROBE_PAIRS = 2500
VAL_FRAC = 0.1
TEST_FRAC = 0.1
TOPK_MOTIF = 3
TOPK_GENE = 20
N_SWAP_DONORS = 3
PERTURB_SIGMA = 0.25       # omics noise (in standardised units) for the input-perturbation stability test
IG_STEPS = 32
GNNEXP_STEPS = 150

PYTHON = r"C:\Users\amr\AppData\Local\Programs\Python\Python311\python.exe"


def seed_all(seed):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def jdump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=_np)


def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _np(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.bool_,)): return bool(o)
    raise TypeError(str(type(o)))


def run_name(model, vocab, split, seed, split_seed=0, tag=""):
    v = vocab if model == "ccmg" else "none"
    return f"{model}_{v}_{split}_ss{split_seed}_s{seed}" + (f"_{tag}" if tag else "")


def ci95(x):
    """mean and 95% normal-approx CI half-width of a 1-D array (nan-ignoring)."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0: return float("nan"), float("nan")
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def boot_ci(x, n=2000, seed=0, stat=np.mean):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    s = stat(x[idx], axis=1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def regression_metrics(y, yhat, drug=None, cell=None, min_group=10):
    """RMSE, Pearson, Spearman overall; mean per-drug and per-cell Pearson (groups with >= min_group pairs)."""
    from scipy.stats import pearsonr, spearmanr
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    out = {"n": int(len(y)), "rmse": float(np.sqrt(np.mean((y - yhat) ** 2)))}
    out["pearson"] = float(pearsonr(y, yhat)[0]) if len(y) > 2 and np.std(yhat) > 0 else float("nan")
    out["spearman"] = float(spearmanr(y, yhat)[0]) if len(y) > 2 and np.std(yhat) > 0 else float("nan")
    for name, g in (("drug", drug), ("cell", cell)):
        if g is None: continue
        g = np.asarray(g); rs = []
        for u in np.unique(g):
            m = g == u
            if m.sum() >= min_group and np.std(yhat[m]) > 0 and np.std(y[m]) > 0:
                rs.append(pearsonr(y[m], yhat[m])[0])
        out[f"per_{name}_pearson"] = float(np.mean(rs)) if rs else float("nan")
        out[f"n_{name}_groups"] = len(rs)
    return out


def build_cell_features(d, train_cells):
    """[expression z | mutation 0/1 | log2 copy-number z] per cell, standardised on the training cells.
    Copy number: DepMap relative CN -> log2(x+1); genes missing from the CN file (all-NaN) -> 0 after scaling."""
    expr, mut, cnv = d["expr"], d["mut"], np.log2(d["cnv"] + 1.0)
    med = np.nanmedian(cnv, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    cnv = np.where(np.isnan(cnv), med[None, :], cnv)
    mu_e, sd_e = expr[train_cells].mean(0), expr[train_cells].std(0) + 1e-6
    mu_c, sd_c = cnv[train_cells].mean(0), cnv[train_cells].std(0) + 1e-6
    X = np.concatenate([(expr - mu_e) / sd_e, mut, (cnv - mu_c) / sd_c], 1).astype(np.float32)
    X = np.clip(X, -10, 10)
    return X, dict(mu_e=mu_e, sd_e=sd_e, mu_c=mu_c, sd_c=sd_c)
