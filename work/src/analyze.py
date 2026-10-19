"""Aggregate accuracy, explanation stability, counterfactual faithfulness and biology agreement over all finished
runs. Writes results/summary/*.json and *.csv. Usage: python analyze.py
"""
import glob, itertools, os, pickle, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr, fisher_exact, hypergeom, wilcoxon
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

MOTIF_EX = ["attn", "rollout", "ig", "mask", "gnnexp", "gnnexp2"]
MOTIF_CTRL = ["size", "random"]
GENE_EX = ["ig", "sal", "mask", "gnnexp"]
SIGNED_MOTIF = ["ig", "mask"]          # explainers whose sign predicts the direction of a deletion effect


# ---------------------------------------------------------------- helpers
def load_runs():
    rows = []
    for f in glob.glob(os.path.join(RUNS, "*", "metrics.json")):
        m = jload(f)
        for part in ("test", "probe", "val"):
            for k, v in m.get(part, {}).items():
                m[f"{part}_{k}"] = v
        m.pop("history", None); rows.append(m)
    for f in glob.glob(os.path.join(RESULTS, "baselines", "*_s*.json")):
        if "_preds" in f or "lgbm_best" in f: continue
        m = jload(f)
        for part in ("test", "probe"):
            for k, v in m.get(part, {}).items():
                m[f"{part}_{k}"] = v
        rows.append(m)
    df = pd.DataFrame(rows)
    return df


def spearman_rows(A, B, min_n=3):
    """row-wise Spearman between two [n, k] score matrices (NaN = padding), ignoring rows with < min_n valid
    entries or a constant vector (vectorised: average ranks per row, then Pearson on the ranks)."""
    from scipy.stats import rankdata
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    ok = ~np.isnan(A) & ~np.isnan(B)
    Am = np.where(ok, A, np.nan); Bm = np.where(ok, B, np.nan)
    ra = rankdata(Am, axis=1, nan_policy="omit"); rb = rankdata(Bm, axis=1, nan_policy="omit")
    ra = np.where(ok, ra, 0.0); rb = np.where(ok, rb, 0.0)
    n = ok.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = ra.sum(1) / n; mb = rb.sum(1) / n
        da = np.where(ok, ra - ma[:, None], 0.0); db = np.where(ok, rb - mb[:, None], 0.0)
        cov = (da * db).sum(1); va = (da ** 2).sum(1); vb = (db ** 2).sum(1)
        out = cov / np.sqrt(va * vb)
    out[(n < min_n) | (va <= 0) | (vb <= 0)] = np.nan
    return out


def topk_sets(S, k):
    """indices of the k largest |scores| per row (NaN ignored); k is capped at n_valid - 1 (>= 1)."""
    sets = []
    for row in S:
        ok = np.where(~np.isnan(row))[0]
        kk = max(1, min(k, len(ok) - 1)) if len(ok) > 1 else 1
        sets.append(set(ok[np.argsort(-np.abs(row[ok]))[:kk]]))
    return sets


def jaccard_rows(A, B, k):
    sa, sb = topk_sets(A, k), topk_sets(B, k)
    return np.array([len(a & b) / len(a | b) if (a | b) else np.nan for a, b in zip(sa, sb)])


def top1_hit(A, B):
    """fraction of rows where the argmax |A| equals the argmax |B| (NaN if either matrix is all-NaN)."""
    if np.all(np.isnan(A)) or np.all(np.isnan(B)):
        return float("nan")
    a = np.nanargmax(np.where(np.isnan(A), -np.inf, np.abs(A)), 1); b = np.nanargmax(np.where(np.isnan(B), -np.inf, np.abs(B)), 1)
    return float(np.mean(a == b))


def summarize(x, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0: return {"mean": np.nan, "lo": np.nan, "hi": np.nan, "n": 0}
    lo, hi = boot_ci(x, seed=seed) if len(x) > 1 else (np.nan, np.nan)
    return {"mean": float(x.mean()), "lo": lo, "hi": hi, "n": int(len(x)), "median": float(np.median(x))}


def load_scores(run):
    f = os.path.join(EXPL, run, "scores.npz")
    return dict(np.load(f)) if os.path.exists(f) else None


def motif_mat(S, name, pert=False, salt=0):
    key = ("motifP_" if pert else "motif_") + name
    if name == "size":
        return S["motif_size"]
    if name == "random":
        # fresh random scores for every comparison (the stored draw is seeded by the training seed, which would make
        # two runs with the same seed but different regimes trivially identical)
        rng = np.random.default_rng((hash((str(S["y0"][:3].round(4).tolist()), key, salt)) & 0xFFFFFFFF))
        R = rng.random(S["motif_attn"].shape).astype(np.float32)
        R[np.isnan(S["motif_attn"])] = np.nan
        return R
    if name.startswith("delta"):
        return S[name].astype(np.float32)
    if key not in S:                      # explainer variant absent from an older run: all-NaN (ignored downstream)
        return np.full(S["motif_attn"].shape, np.nan, np.float32)
    return S[key].astype(np.float32)


def gene_mat(S, name, pert=False):
    return S[("geneP_" if pert else "gene_") + name].astype(np.float32)


# ---------------------------------------------------------------- analyses
def accuracy_table(df):
    rows = []
    main = df[(df.split_seed == 0)]
    for (model, vocab, split), g in main.groupby(["model", "vocab", "split"]):
        r = {"model": model, "vocab": vocab, "split": split, "n_seeds": len(g)}
        for k in ["test_rmse", "test_pearson", "test_spearman", "test_per_drug_pearson", "test_per_cell_pearson",
                  "probe_rmse", "probe_pearson", "probe_per_drug_pearson"]:
            if k in g:
                v = g[k].astype(float).to_numpy(); r[k] = float(np.nanmean(v)); r[k + "_sd"] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0
        rows.append(r)
    acc = pd.DataFrame(rows)
    # resampling (split seed = seed) set
    rows = []
    res = df[(df.split_seed == df.seed) | (df.model == "naive")]
    for (model, vocab, split), g in res.groupby(["model", "vocab", "split"]):
        g = g.drop_duplicates("split_seed")
        r = {"model": model, "vocab": vocab, "split": split, "n_split_seeds": len(g)}
        for k in ["test_rmse", "test_pearson", "test_per_drug_pearson"]:
            if k in g:
                v = g[k].astype(float).to_numpy(); r[k] = float(np.nanmean(v)); r[k + "_sd"] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0
        rows.append(r)
    return acc, pd.DataFrame(rows)


def top_atom_sets(S, graphs, ex):
    """atom set of the top-|score| motif of every probe pair."""
    A = S["delta_del"] if ex == "delta_del" else motif_mat(S, ex)
    out = []
    for i, d in enumerate(S["d_idx"]):
        if np.all(np.isnan(A[i])):
            out.append(set()); continue
        m = int(np.nanargmax(np.where(np.isnan(A[i]), -np.inf, np.abs(A[i]))))
        out.append(set(graphs[d]["motif_atoms"][m]))
    return out


def stability(expl_runs, graphs_by_vocab=None):
    """seed, split, perturbation and vocabulary stability of motif- and gene-level explanations."""
    S = {r: load_scores(r) for r in expl_runs}
    S = {r: s for r, s in S.items() if s is not None}
    info = {r: jload(os.path.join(EXPL, r, "summary.json")) for r in S}
    rows = []

    def add(axis, vocab, split, ex, level, rho, jac, extra=None):
        d = {"axis": axis, "vocab": vocab, "split": split, "explainer": ex, "level": level,
             "rho_mean": float(np.nanmean(rho)), "rho_median": float(np.nanmedian(rho)), "jaccard_mean": float(np.nanmean(jac)),
             "n_pairs": int(np.sum(~np.isnan(rho)))}
        if extra: d.update(extra)
        rows.append(d)

    # seed stability: same vocab/split/split seed 0, different training seeds
    for vocab in VOCABS:
        for split in SPLITS:
            runs = [r for r in S if info[r]["vocab"] == vocab and info[r]["split"] == split and info[r]["split_seed"] == 0]
            for r1, r2 in itertools.combinations(sorted(runs), 2):
                for ex in MOTIF_EX + ["random", "delta_del", "delta_swap"]:
                    A, B = motif_mat(S[r1], ex), motif_mat(S[r2], ex)
                    add("seed", vocab, split, ex, "motif", spearman_rows(A, B), jaccard_rows(A, B, TOPK_MOTIF), {"pair": f"{r1}|{r2}", "top1": top1_hit(A, B)})
                for ex in GENE_EX:
                    A, B = gene_mat(S[r1], ex), gene_mat(S[r2], ex)
                    add("seed", vocab, split, ex, "gene", spearman_rows(A, B), jaccard_rows(A, B, TOPK_GENE), {"pair": f"{r1}|{r2}", "top1": top1_hit(A, B)})
            # prediction stability across seeds (reference)
            for r1, r2 in itertools.combinations(sorted(runs), 2):
                rows.append({"axis": "seed", "vocab": vocab, "split": split, "explainer": "prediction", "level": "y",
                             "rho_mean": float(spearmanr(S[r1]["y0"], S[r2]["y0"])[0]), "pair": f"{r1}|{r2}", "n_pairs": len(S[r1]["y0"])})
    # split stability: same vocab/seed, different split regime (same probe pairs)
    for vocab in VOCABS:
        for seed in SEEDS:
            runs = {info[r]["split"]: r for r in S if info[r]["vocab"] == vocab and info[r]["seed"] == seed and info[r]["split_seed"] == 0}
            for s1, s2 in itertools.combinations(SPLITS, 2):
                if s1 in runs and s2 in runs:
                    for ex in MOTIF_EX + ["random", "delta_del"]:
                        A, B = motif_mat(S[runs[s1]], ex), motif_mat(S[runs[s2]], ex)
                        add("split", vocab, f"{s1}|{s2}", ex, "motif", spearman_rows(A, B), jaccard_rows(A, B, TOPK_MOTIF), {"seed": seed, "top1": top1_hit(A, B)})
                    for ex in GENE_EX:
                        A, B = gene_mat(S[runs[s1]], ex), gene_mat(S[runs[s2]], ex)
                        add("split", vocab, f"{s1}|{s2}", ex, "gene", spearman_rows(A, B), jaccard_rows(A, B, TOPK_GENE), {"seed": seed, "top1": top1_hit(A, B)})
                    rows.append({"axis": "split", "vocab": vocab, "split": f"{s1}|{s2}", "explainer": "prediction", "level": "y",
                                 "rho_mean": float(spearmanr(S[runs[s1]]["y0"], S[runs[s2]]["y0"])[0]), "seed": seed})
    # perturbation stability: clean vs noisy inputs, same run
    for r in S:
        for ex in MOTIF_EX + ["random"]:
            A, B = motif_mat(S[r], ex), motif_mat(S[r], ex, pert=True)
            add("perturb", info[r]["vocab"], info[r]["split"], ex, "motif", spearman_rows(A, B), jaccard_rows(A, B, TOPK_MOTIF), {"run": r, "top1": top1_hit(A, B)})
        for ex in GENE_EX:
            A, B = gene_mat(S[r], ex), gene_mat(S[r], ex, pert=True)
            add("perturb", info[r]["vocab"], info[r]["split"], ex, "gene", spearman_rows(A, B), jaccard_rows(A, B, TOPK_GENE), {"run": r, "top1": top1_hit(A, B)})
        rows.append({"axis": "perturb", "vocab": info[r]["vocab"], "split": info[r]["split"], "explainer": "prediction", "level": "y",
                     "rho_mean": float(spearmanr(S[r]["y0"], S[r]["y_pert"])[0]), "run": r,
                     "pred_shift_rmse": float(np.sqrt(np.mean((S[r]["y0"] - S[r]["y_pert"]) ** 2)))})
    # vocabulary stability (gene level only: motif vocabularies are not comparable): BRICS vs RECAP, same split/seed
    for split in SPLITS:
        for seed in SEEDS:
            r1 = run_name("ccmg", "brics", split, seed); r2 = run_name("ccmg", "recap", split, seed)
            if r1 in S and r2 in S:
                for ex in GENE_EX:
                    A, B = gene_mat(S[r1], ex), gene_mat(S[r2], ex)
                    add("vocab", "brics|recap", split, ex, "gene", spearman_rows(A, B), jaccard_rows(A, B, TOPK_GENE), {"seed": seed, "top1": top1_hit(A, B)})
                if graphs_by_vocab is not None:
                    # motif level across vocabularies: do the top motifs of the two backbones cover the same atoms?
                    for ex in MOTIF_EX + ["delta_del"]:
                        sa = top_atom_sets(S[r1], graphs_by_vocab["brics"]["graphs"], ex); sb = top_atom_sets(S[r2], graphs_by_vocab["recap"]["graphs"], ex)
                        jac = np.array([len(a & b) / len(a | b) if (a | b) else np.nan for a, b in zip(sa, sb)])
                        ovl = np.array([len(a & b) / min(len(a), len(b)) if a and b else np.nan for a, b in zip(sa, sb)])
                        rows.append({"axis": "vocab", "vocab": "brics|recap", "split": split, "explainer": ex, "level": "motif_atoms", "seed": seed,
                                     "rho_mean": np.nan, "jaccard_mean": float(np.nanmean(jac)), "overlap_mean": float(np.nanmean(ovl)),
                                     "frac_disjoint": float(np.nanmean(jac == 0)), "n_pairs": int(np.sum(~np.isnan(jac)))})
    # data-resampling stability (split seed = seed, BRICS): compare each resampled run with the fixed-split seed-0 run
    for split in SPLITS:
        base = run_name("ccmg", "brics", split, 0, 0)
        for ss in SPLIT_SEEDS[1:]:
            r = run_name("ccmg", "brics", split, ss, ss)
            if base in S and r in S:
                for ex in MOTIF_EX + ["random"]:
                    A, B = motif_mat(S[base], ex), motif_mat(S[r], ex)
                    add("resample", "brics", split, ex, "motif", spearman_rows(A, B), jaccard_rows(A, B, TOPK_MOTIF), {"pair": f"{base}|{r}", "top1": top1_hit(A, B)})
                for ex in GENE_EX:
                    A, B = gene_mat(S[base], ex), gene_mat(S[r], ex)
                    add("resample", "brics", split, ex, "gene", spearman_rows(A, B), jaccard_rows(A, B, TOPK_GENE), {"pair": f"{base}|{r}", "top1": top1_hit(A, B)})
    # explainer agreement on the same model (motif level)
    for r in S:
        for e1, e2 in itertools.combinations(MOTIF_EX, 2):
            A, B = motif_mat(S[r], e1), motif_mat(S[r], e2)
            rows.append({"axis": "explainer_agreement", "vocab": info[r]["vocab"], "split": info[r]["split"], "explainer": f"{e1}|{e2}", "level": "motif",
                         "rho_mean": float(np.nanmean(spearman_rows(np.abs(A), np.abs(B)))), "jaccard_mean": float(np.nanmean(jaccard_rows(A, B, TOPK_MOTIF))),
                         "run": r, "top1": top1_hit(A, B)})
    return pd.DataFrame(rows), S, info


def faithfulness(S, info, meta, splits_meta):
    """agreement of each explainer's motif ranking with the model's own counterfactual motif effects."""
    rows, size_rows = [], []
    for r, s in S.items():
        valid = ~np.isnan(s["delta_del"])
        cv = s["chem_valid"].astype(bool)
        for cf in ("delta_del", "delta_swap"):
            D = s[cf]
            for ex in MOTIF_EX + MOTIF_CTRL:
                A = motif_mat(s, ex)
                rho_abs = spearman_rows(np.abs(A), np.abs(D))
                rec = {"run": r, "vocab": info[r]["vocab"], "split": info[r]["split"], "seed": info[r]["seed"], "split_seed": info[r]["split_seed"],
                       "counterfactual": cf, "explainer": ex, "rho_abs_mean": float(np.nanmean(rho_abs)), "rho_abs_median": float(np.nanmedian(rho_abs)),
                       "top1_hit": top1_hit(A, D), "n_pairs": int(np.sum(~np.isnan(rho_abs)))}
                if cf == "delta_del":
                    # valid-edit vs graph-edit: restrict the deletion effects to motifs whose remainder sanitised (or did not)
                    Dv = np.where(cv, D, np.nan); Dg = np.where(~cv & valid, D, np.nan)
                    rec["rho_abs_valid"] = float(np.nanmean(spearman_rows(np.abs(A), np.abs(Dv))))
                    rec["rho_abs_graph"] = float(np.nanmean(spearman_rows(np.abs(A), np.abs(Dg))))
                    # top-1 agreement on pairs whose top-|delta| motif is a valid edit
                    topd = np.nanargmax(np.where(np.isnan(D), -np.inf, np.abs(D)), 1)
                    okv = cv[np.arange(len(topd)), topd]
                    if okv.any() and not np.all(np.isnan(A)):
                        rec["top1_hit_validtop"] = top1_hit(A[okv], D[okv])
                    rec["frac_top_delta_valid"] = float(okv.mean())
                if ex in SIGNED_MOTIF:
                    # prediction drop on removal should be ~ +score for ig (f(x) - f(x without m) ~ IG_m) and mask (defined as y0 - y_masked)
                    rho_s = spearman_rows(A, -D)
                    rec["rho_signed_mean"] = float(np.nanmean(rho_s))
                    rec["sign_agree"] = float(np.nanmean((np.sign(A) == np.sign(-D))[valid]))
                # fidelity: |delta| of the explainer's top-1 motif relative to the mean |delta| over the drug's motifs
                top = np.nanargmax(np.where(np.isnan(A), -np.inf, np.abs(A)), 1)
                dtop = np.abs(D[np.arange(len(top)), top]); dmean = np.nanmean(np.abs(D), 1); dmax = np.nanmax(np.abs(D), 1)
                rec["fidelity_top1_ratio"] = float(np.nanmean(dtop / np.where(dmean > 0, dmean, np.nan)))
                rec["fidelity_top1_frac_of_max"] = float(np.nanmean(dtop / np.where(dmax > 0, dmax, np.nan)))
                rows.append(rec)
        # chance level of top-1 agreement = mean 1/n_motifs
        nv = valid.sum(1)
        rows.append({"run": r, "vocab": info[r]["vocab"], "split": info[r]["split"], "seed": info[r]["seed"], "split_seed": info[r]["split_seed"],
                     "counterfactual": "chance", "explainer": "chance", "top1_hit": float(np.mean(1.0 / nv)), "rho_abs_mean": 0.0, "n_pairs": int(len(nv))})
        # how much of the counterfactual effect is explained by motif size alone, and deletion vs swap agreement
        size_rows.append({"run": r, "vocab": info[r]["vocab"], "split": info[r]["split"], "seed": info[r]["seed"],
                          "rho_size_del": float(np.nanmean(spearman_rows(s["motif_size"], np.abs(s["delta_del"])))),
                          "rho_del_swap": float(np.nanmean(spearman_rows(np.abs(s["delta_del"]), np.abs(s["delta_swap"])))),
                          "mean_abs_del": float(np.nanmean(np.abs(s["delta_del"]))), "mean_abs_swap": float(np.nanmean(np.abs(s["delta_swap"]))),
                          "frac_del_increase_ic50": float(np.nanmean(s["delta_del"][valid] > 0)),
                          "frac_chem_valid": float(np.mean(s["chem_valid"][valid])),
                          "y_sd_probe": float(np.std(s["y0"]))})
    return pd.DataFrame(rows), pd.DataFrame(size_rows)


def cell_dependence(S, info):
    """How much does the motif explanation of a drug change with the cell? Per probe drug: fraction of its probe
    cells sharing the modal top-1 motif, mean Spearman between the score vectors of two different cells of the
    same drug, and the attention concentration (mean max alpha, normalised entropy)."""
    rows = []
    for r, s in S.items():
        dn = s["d_idx"]
        for ex in MOTIF_EX + ["delta_del"]:
            A = s["delta_del"] if ex == "delta_del" else motif_mat(s, ex)
            if np.all(np.isnan(A)):
                continue
            modal, rho_cells = [], []
            rng = np.random.default_rng(0)
            for x in sorted(set(dn.tolist())):
                sel = np.where(dn == x)[0]
                top = np.nanargmax(np.where(np.isnan(A[sel]), -np.inf, np.abs(A[sel])), 1)
                modal.append(np.bincount(top).max() / len(top))
                # 200 random cell pairs of this drug
                i = rng.integers(0, len(sel), 200); j = rng.integers(0, len(sel), 200); ok = i != j
                rho_cells.append(np.nanmean(spearman_rows(np.abs(A[sel[i[ok]]]), np.abs(A[sel[j[ok]]]))))
            rec = {"run": r, "vocab": info[r]["vocab"], "split": info[r]["split"], "seed": info[r]["seed"], "split_seed": info[r]["split_seed"],
                   "explainer": ex, "modal_top1_frac": float(np.mean(modal)), "rho_between_cells": float(np.nanmean(rho_cells))}
            if ex == "attn":
                al = np.where(np.isnan(A), 0.0, A); n = (~np.isnan(A)).sum(1)
                ent = -(al * np.log(al + 1e-12)).sum(1) / np.log(np.maximum(n, 2))
                rec["attn_max_mean"] = float(np.nanmax(A, 1).mean()); rec["attn_norm_entropy"] = float(ent.mean())
                rec["attn_max_gt_09"] = float(np.mean(np.nanmax(A, 1) > 0.9))
            rows.append(rec)
    return pd.DataFrame(rows)


def target_class_motifs(meta, graphs, min_drugs=3):
    """motif vocabulary ids enriched in each GDSC target pathway (Fisher exact, one-sided, over the whole drug panel)."""
    vid_sets = [set(g["motif_vid"].tolist()) - {0} for g in graphs]
    pathways = [d["pathway"] for d in meta["drugs"]]
    enriched = {}
    for pw in sorted(set(pathways)):
        if pw in ("Other", "Unclassified", "Other, kinases"):
            continue
        ind = [i for i, p in enumerate(pathways) if p == pw]
        if len(ind) < min_drugs:
            continue
        outd = [i for i in range(len(pathways)) if i not in ind]
        vids = set.union(*[vid_sets[i] for i in ind])
        hits = {}
        for v in vids:
            a = sum(v in vid_sets[i] for i in ind); b = len(ind) - a
            c = sum(v in vid_sets[i] for i in outd); d = len(outd) - c
            odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
            if p < 0.05 and a >= 2:
                hits[v] = {"p": float(p), "in_class": a, "out_class": c}
        enriched[pw] = hits
    return enriched


def biology(S, info, meta, splits_meta, graphs_by_vocab):
    """gene-level target recovery and dependency agreement; motif-level class-enriched motif recovery and
    cell-conditioning coherence of the counterfactual motif effects."""
    d = np.load(os.path.join(DATA, "dataset.npz"))
    crispr = d["crispr"]; genes = meta["genes"]; gidx = {g: i for i, g in enumerate(genes)}
    G = len(genes)
    drug_targets = {i: [gidx[t] for t in dr["targets"] if t in gidx] for i, dr in enumerate(meta["drugs"])}
    rows, dep_rows, motif_rows, coh_rows = [], [], [], []
    enriched = {v: target_class_motifs(meta, graphs_by_vocab[v]["graphs"]) for v in VOCABS}
    for r, s in S.items():
        vocab, split = info[r]["vocab"], info[r]["split"]
        dn, cn = s["d_idx"], s["c_idx"]
        has_t = np.array([len(drug_targets[x]) > 0 for x in dn])
        # chance of >= 1 target in top-k: 1 - hypergeom.pmf(0, G, n_targets, k)
        chance = np.array([1 - hypergeom.pmf(0, G, len(drug_targets[x]), TOPK_GENE) if drug_targets[x] else np.nan for x in dn])
        for ex in GENE_EX + ["kd"]:
            A = s["delta_kd"].astype(np.float32) if ex == "kd" else gene_mat(s, ex)
            order = np.argsort(-np.abs(A), 1)[:, :TOPK_GENE]
            hit = np.array([len(set(order[i]) & set(drug_targets[dn[i]])) > 0 if has_t[i] else np.nan for i in range(len(dn))], float)
            # rank of the best target gene (fraction of genes ranked above it)
            ranks = np.argsort(np.argsort(-np.abs(A), 1), 1)
            best_rank = np.array([ranks[i, drug_targets[dn[i]]].min() / G if has_t[i] else np.nan for i in range(len(dn))])
            rows.append({"run": r, "vocab": vocab, "split": split, "seed": info[r]["seed"], "split_seed": info[r]["split_seed"], "explainer": ex,
                         "target_in_top20": float(np.nanmean(hit)), "chance_top20": float(np.nanmean(chance)),
                         "best_target_rank_frac": float(np.nanmean(best_rank)), "n_pairs_with_target": int(np.sum(has_t))})
            # dependency agreement: per probe drug with a target screened by CRISPR, Spearman over probe cells between
            # |attribution of the target gene| and the dependency strength (-CRISPR gene effect)
            for x in sorted(set(dn.tolist())):
                for t in drug_targets[x]:
                    sel = np.where(dn == x)[0]
                    dep = -crispr[cn[sel], t]
                    ok = ~np.isnan(dep)
                    if ok.sum() >= 20:
                        a = np.abs(A[sel[ok], t]); rho = spearmanr(a, dep[ok])[0] if np.std(a) > 0 else np.nan
                        rho_signed = spearmanr(A[sel[ok], t], dep[ok])[0] if np.std(a) > 0 else np.nan
                        dep_rows.append({"run": r, "vocab": vocab, "split": split, "seed": info[r]["seed"], "split_seed": info[r]["split_seed"],
                                         "explainer": ex, "drug": meta["drugs"][x]["name"], "target": genes[t], "rho_abs": rho, "rho_signed": rho_signed, "n_cells": int(ok.sum())})
        # motif level: top-1 explained motif in the pathway-enriched set (chance = enriched fraction of the drug's motifs)
        gr = graphs_by_vocab[vocab]["graphs"]
        for ex in MOTIF_EX + MOTIF_CTRL + ["delta_del", "delta_swap"]:
            A = s[ex] if ex.startswith("delta") else motif_mat(s, ex)
            hits, chance_m = [], []
            for i, x in enumerate(dn):
                pw = meta["drugs"][x]["pathway"]
                enr = set(enriched[vocab].get(pw, {}).keys())
                vids = gr[x]["motif_vid"]
                flag = np.array([v in enr for v in vids])
                if flag.any() and not flag.all():
                    top = np.nanargmax(np.where(np.isnan(A[i]), -np.inf, np.abs(A[i])))
                    hits.append(float(flag[top])); chance_m.append(float(flag.mean()))
            motif_rows.append({"run": r, "vocab": vocab, "split": split, "seed": info[r]["seed"], "split_seed": info[r]["split_seed"], "explainer": ex,
                               "top1_in_enriched": float(np.mean(hits)) if hits else np.nan, "chance": float(np.mean(chance_m)) if chance_m else np.nan,
                               "n_pairs": len(hits)})
        # cell-conditioning coherence: for each probe drug with a CRISPR-screened target, Spearman over probe cells between the
        # deletion effect of the drug's top counterfactual motif and the dependency strength of the target
        for x in sorted(set(dn.tolist())):
            sel = np.where(dn == x)[0]
            D = s["delta_del"][sel]
            mstar = int(np.nanargmax(np.nanmean(np.abs(D), 0)))
            for t in drug_targets[x]:
                dep = -crispr[cn[sel], t]; ok = ~np.isnan(dep)
                if ok.sum() >= 20 and np.std(D[ok, mstar]) > 0:
                    coh_rows.append({"run": r, "vocab": vocab, "split": split, "seed": info[r]["seed"], "split_seed": info[r]["split_seed"],
                                     "drug": meta["drugs"][x]["name"], "target": genes[t], "motif": mstar,
                                     "rho_del_dep": spearmanr(D[ok, mstar], dep[ok])[0],
                                     "rho_absdel_dep": spearmanr(np.abs(D[ok, mstar]), dep[ok])[0],
                                     "rho_pred_dep": spearmanr(s["y0"][sel][ok], dep[ok])[0],
                                     "rho_true_dep": spearmanr(s["y_true"][sel][ok], dep[ok])[0], "n_cells": int(ok.sum())})
    n_enr = {v: {pw: len(h) for pw, h in enriched[v].items()} for v in VOCABS}
    return pd.DataFrame(rows), pd.DataFrame(dep_rows), pd.DataFrame(motif_rows), pd.DataFrame(coh_rows), n_enr


def drug_level_seed_agreement(S, info):
    """Drug-level reproducibility of the explanation across training seeds: for every run the modal top-1 motif of
    each probe drug (over its probe cells); then, per (vocab, split, split seed, tag) group of seeds, the fraction of
    drugs whose modal motif is identical in every seed, the mean pairwise agreement over seed pairs and the mean
    plurality share (fraction of seeds agreeing with the most common modal motif)."""
    groups = {}
    for r in S:
        key = (info[r]["vocab"], info[r]["split"], info[r]["split_seed"], info[r].get("tag", ""))
        groups.setdefault(key, []).append(r)
    rows = []
    for key, runs in groups.items():
        if len(runs) < 2:
            continue
        runs = sorted(runs)
        dn = S[runs[0]]["d_idx"]; drugs = sorted(set(dn.tolist()))
        for ex in MOTIF_EX + ["random", "delta_del"]:
            modal = np.zeros((len(runs), len(drugs)), int)
            for i, r in enumerate(runs):
                A = motif_mat(S[r], ex, salt=i)
                if np.all(np.isnan(A)):
                    modal = None; break
                top = np.nanargmax(np.where(np.isnan(A), -np.inf, np.abs(A)), 1)
                for j, x in enumerate(drugs):
                    modal[i, j] = np.bincount(top[dn == x]).argmax()
            if modal is None:
                continue
            all_agree = np.mean([len(set(modal[:, j])) == 1 for j in range(len(drugs))])
            pair = np.mean([np.mean(modal[a] == modal[b]) for a, b in itertools.combinations(range(len(runs)), 2)])
            plural = np.mean([np.bincount(modal[:, j]).max() / len(runs) for j in range(len(drugs))])
            rows.append({"vocab": key[0], "split": key[1], "split_seed": key[2], "tag": key[3], "explainer": ex, "n_runs": len(runs),
                         "n_drugs": len(drugs), "frac_all_agree": float(all_agree), "pairwise_agree": float(pair), "plurality_share": float(plural)})
    return pd.DataFrame(rows)


def ensemble_accuracy(df):
    """Accuracy of the mean prediction over the five training seeds (fixed split) for each neural model / regime."""
    rows = []
    d = np.load(os.path.join(DATA, "dataset.npz")); pdr, pcl = d["pair_drug"], d["pair_cell"]
    base = df[(df.split_seed == 0) & (df["tag"] == "") & (df.model.isin(["ccmg", "latefusion"]))]
    for (model, vocab, split), g in base.groupby(["model", "vocab", "split"]):
        preds, idx = [], None
        for name in g.name:
            f = os.path.join(RUNS, name, "preds_test.npz")
            if os.path.exists(f):
                z = np.load(f); preds.append(z["yhat"]); idx = z["idx"]; y = z["y"]; is_probe = z["is_probe"]
        if len(preds) < 2:
            continue
        yh = np.mean(preds, 0)
        m = regression_metrics(y, yh, pdr[idx], pcl[idx])
        rows.append({"model": model + "-ens", "vocab": vocab, "split": split, "n_members": len(preds), "test_rmse": m["rmse"], "test_pearson": m["pearson"],
                     "test_per_drug_pearson": m["per_drug_pearson"], "probe_rmse": float(np.sqrt(np.mean((y[is_probe] - yh[is_probe]) ** 2)))})
    return pd.DataFrame(rows)


def variant_summary(S, info, df, tag, label):
    """one row per configuration (random regime): accuracy, attention shape, cell dependence, seed and drug-level
    reproducibility, counterfactual faithfulness (deletion)."""
    runs = sorted(r for r in S if info[r]["split"] == "random" and info[r]["split_seed"] == 0 and info[r].get("tag", "") == tag and info[r]["vocab"] == "brics")
    if not runs:
        return None
    sub = {r: S[r] for r in runs}; inf = {r: info[r] for r in runs}
    row = {"tag": tag, "label": label, "n_runs": len(runs)}
    g = df[(df.model == "ccmg") & (df.vocab == "brics") & (df.split == "random") & (df.split_seed == 0) & (df["tag"] == tag)]
    if len(g):
        row.update({"test_rmse": float(g.test_rmse.mean()), "r_drug": float(g.test_per_drug_pearson.mean()), "r_drug_sd": float(g.test_per_drug_pearson.std(ddof=1)) if len(g) > 1 else 0.0,
                    "val_rmse": float(g.best_val_rmse.mean()), "attn_entropy_final": float(g.final_val_attn_entropy.mean()) if "final_val_attn_entropy" in g else np.nan,
                    "heads": int(g.heads.fillna(1).iloc[0]) if "heads" in g else 1, "hidden": int(g.hidden.iloc[0]),
                    "dropout": float(g.dropout.fillna(0.1).iloc[0]) if "dropout" in g else 0.1, "epochs": float(g.epochs_run.mean())})
    cd = cell_dependence(sub, inf)
    for ex in ("attn", "rollout", "ig", "mask", "delta_del"):
        c = cd[cd.explainer == ex]
        row[f"cd_modal_{ex}"] = float(c.modal_top1_frac.mean()) if len(c) else np.nan
    c = cd[cd.explainer == "attn"]
    row["attn_max"] = float(c.attn_max_mean.mean()) if len(c) else np.nan; row["attn_gt09"] = float(c.attn_max_gt_09.mean()) if len(c) else np.nan
    row["attn_norm_entropy"] = float(c.attn_norm_entropy.mean()) if len(c) else np.nan
    # seed stability (pair-level) and drug-level agreement
    pairs = list(itertools.combinations(runs, 2))
    for ex in ("attn", "rollout", "ig", "mask", "delta_del"):
        rhos = [np.nanmean(spearman_rows(np.abs(motif_mat(sub[a], ex)), np.abs(motif_mat(sub[b], ex)))) for a, b in pairs]
        row[f"seed_rho_{ex}"] = float(np.nanmean(rhos)) if rhos else np.nan
    row["seed_rho_prediction"] = float(np.mean([spearmanr(sub[a]["y0"], sub[b]["y0"])[0] for a, b in pairs])) if pairs else np.nan
    dsa = drug_level_seed_agreement(sub, inf)
    for ex in ("attn", "rollout", "ig", "mask", "delta_del"):
        dd = dsa[dsa.explainer == ex]
        row[f"drug_agree_{ex}"] = float(dd.pairwise_agree.mean()) if len(dd) else np.nan
        row[f"drug_all_{ex}"] = float(dd.frac_all_agree.mean()) if len(dd) else np.nan
    # faithfulness against deletion
    for ex in ("attn", "rollout", "ig", "mask", "gnnexp", "gnnexp2"):
        rh, t1 = [], []
        for r in runs:
            A = motif_mat(sub[r], ex); D = sub[r]["delta_del"]
            rh.append(np.nanmean(spearman_rows(np.abs(A), np.abs(D)))); t1.append(top1_hit(A, D))
        row[f"faith_rho_{ex}"] = float(np.nanmean(rh)); row[f"faith_top1_{ex}"] = float(np.nanmean(t1))
    return row


def main():
    meta = jload(os.path.join(DATA, "meta.json")); splits_meta = jload(os.path.join(DATA, "splits_meta.json"))
    graphs_by_vocab = {}
    for v in VOCABS:
        with open(os.path.join(DATA, f"graphs_{v}.pkl"), "rb") as f:
            graphs_by_vocab[v] = pickle.load(f)
    df_all = load_runs()
    if "tag" not in df_all:
        df_all["tag"] = ""
    df_all["tag"] = df_all["tag"].fillna("")
    df_all.to_csv(os.path.join(SUMMARY, "runs.csv"), index=False)
    df = df_all[df_all.tag == ""]
    acc, acc_res = accuracy_table(df)
    ens = ensemble_accuracy(df_all)
    acc.to_csv(os.path.join(SUMMARY, "accuracy.csv"), index=False); acc_res.to_csv(os.path.join(SUMMARY, "accuracy_resampled.csv"), index=False)
    ens.to_csv(os.path.join(SUMMARY, "accuracy_ensemble.csv"), index=False)
    # tuning grid and attention variants (random regime, tagged runs)
    tuned = df_all[(df_all.tag != "") & (df_all.model == "ccmg")]
    tuned.to_csv(os.path.join(SUMMARY, "runs_tagged.csv"), index=False)
    all_expl = sorted(os.path.basename(os.path.dirname(f)) for f in glob.glob(os.path.join(EXPL, "*", "summary.json")))
    tags = {r: jload(os.path.join(EXPL, r, "summary.json")).get("tag", "") for r in all_expl}
    expl_runs = [r for r in all_expl if tags[r] == ""]
    print(f"{len(df_all)} run records ({len(df)} main); {len(expl_runs)} main explained runs, {len(all_expl) - len(expl_runs)} variant explained runs", flush=True)
    stab, S, info = stability(expl_runs, graphs_by_vocab)
    stab.to_csv(os.path.join(SUMMARY, "stability.csv"), index=False)
    faith, cfstats = faithfulness(S, info, meta, splits_meta)
    faith.to_csv(os.path.join(SUMMARY, "faithfulness.csv"), index=False); cfstats.to_csv(os.path.join(SUMMARY, "counterfactual_stats.csv"), index=False)
    cell_dependence(S, info).to_csv(os.path.join(SUMMARY, "cell_dependence.csv"), index=False)
    drug_level_seed_agreement(S, info).to_csv(os.path.join(SUMMARY, "drug_seed_agreement.csv"), index=False)
    # variants: default vs attention controls vs tuned configuration (random regime, BRICS)
    vrows = []
    row = variant_summary(S, info, df_all, "", "default (1 head)")
    if row: vrows.append(row)
    vtags = sorted(set(t for t in tags.values() if t))
    if vtags:
        Sv = {r: load_scores(r) for r in all_expl if tags[r]}
        Sv = {r: s for r, s in Sv.items() if s is not None}
        infov = {r: jload(os.path.join(EXPL, r, "summary.json")) for r in Sv}
        labels = {"h4": "4 attention heads", "ent0.05": "entropy bonus (1 head)"}
        for t in vtags:
            row = variant_summary(Sv, infov, df_all, t, labels.get(t, f"tuned: {t}"))
            if row: vrows.append(row)
    pd.DataFrame(vrows).to_csv(os.path.join(SUMMARY, "variants.csv"), index=False)
    bio, dep, mbio, coh, n_enr = biology(S, info, meta, splits_meta, graphs_by_vocab)
    bio.to_csv(os.path.join(SUMMARY, "bio_gene.csv"), index=False); dep.to_csv(os.path.join(SUMMARY, "bio_dependency.csv"), index=False)
    mbio.to_csv(os.path.join(SUMMARY, "bio_motif.csv"), index=False); coh.to_csv(os.path.join(SUMMARY, "bio_coherence.csv"), index=False)
    jdump({"n_runs": int(len(df)), "explained_runs": expl_runs, "n_enriched_motifs_per_pathway": n_enr,
           "dataset": {"n_pairs": meta["n_pairs"], "n_drugs": len(meta["drugs"]), "n_cells": len(meta["cells"]), "n_genes": len(meta["genes"]),
                       "n_target_genes": len(meta["target_genes"]), "n_drugs_with_targets": sum(1 for d in meta["drugs"] if d["targets"]),
                       "n_probe_pairs": splits_meta["n_probe_pairs"], "n_probe_drugs": len(splits_meta["probe_drugs"]),
                       "n_probe_cells": len(splits_meta["probe_cells"])}}, os.path.join(SUMMARY, "overview.json"))
    print("summary written to", SUMMARY)


if __name__ == "__main__":
    main()
