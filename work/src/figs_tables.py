"""Numbers (paper/numbers.tex), LaTeX tables (paper/tab_*.tex) and figures (figs/*.pdf -> paper/) from
results/summary/*. Usage: python figs_tables.py   (run analyze.py first; build.py does both)
"""
import glob, os, pickle, shutil, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

# validated categorical palette (dataviz reference instance, light mode), fixed slot order
PAL = {"attn": "#2a78d6", "rollout": "#eb6834", "ig": "#1baf7a", "mask": "#eda100", "gnnexp": "#e87ba4", "gnnexp2": "#e34948",
       "sal": "#6250d6", "kd": "#008300", "delta_del": "#008300", "delta_swap": "#6250d6",
       "random": "#9a9a9a", "size": "#5f5e5a", "chance": "#9a9a9a", "prediction": "#0b0b0b"}
MPAL = {"naive": "#9a9a9a", "lgbm": "#eda100", "latefusion": "#eb6834", "ccmg|brics": "#2a78d6", "ccmg|recap": "#1baf7a"}
LAB = {"attn": "attention", "rollout": "rollout", "ig": "integrated gradients", "mask": "masking", "gnnexp": "GNNExplainer", "gnnexp2": "GNNExplainer (input mask)",
       "sal": "gradient$\\times$input", "kd": "knockdown $\\Delta_{\\mathrm{kd}}$", "random": "random", "size": "motif size",
       "chance": "chance", "delta_del": "$|\\Delta_{\\mathrm{del}}|$", "delta_swap": "$|\\Delta_{\\mathrm{swap}}|$",
       "prediction": "prediction"}
MLAB = {"naive": "naive mean effect", "lgbm": "LightGBM (tuned)", "latefusion": "late fusion", "ccmg|brics": "MotifCell-GNN (BRICS)",
        "ccmg|recap": "MotifCell-GNN (RECAP)"}
SLAB = {"random": "random", "unseen_drug": "unseen drug", "unseen_cell": "unseen cell"}
ALAB = {"seed": "seed", "split": "regime", "perturb": "input noise", "vocab": "vocabulary", "resample": "resampled split"}
MOTIF_EX = ["attn", "rollout", "ig", "mask", "gnnexp"]
GENE_EX = ["ig", "sal", "mask", "gnnexp"]
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9, "axes.titleweight": "bold",
                     "legend.frameon": False, "pdf.fonttype": 42, "figure.dpi": 150})

MAC = []


DIGIT = {"0": "Zero", "1": "One", "2": "B", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def mac(name, val):
    name = "".join(DIGIT.get(ch, ch) for ch in name)   # LaTeX macro names cannot contain digits (gnnexp2 -> GnnexpB)
    MAC.append(f"\\newcommand{{\\{name}}}{{{val}}}")


def f2(x, nd=2):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def pct(x, nd=0):
    return "--" if x is None or np.isnan(x) else f"{100 * x:.{nd}f}\\%"


def num(x):
    return f"{int(x):,}".replace(",", "{,}")


def read(name):
    f = os.path.join(SUMMARY, name)
    return pd.read_csv(f) if os.path.exists(f) and os.path.getsize(f) > 2 else pd.DataFrame()


def style(ax, ylab=None, xlab=None):
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    if ylab: ax.set_ylabel(ylab)
    if xlab: ax.set_xlabel(xlab)


def bars(ax, cats, groups, vals, errs=None, colors=None, width=0.8, labels=None):
    """grouped thin bars: cats on x, one bar per group."""
    n = len(groups); w = width / n
    x = np.arange(len(cats))
    for i, g in enumerate(groups):
        v = vals[g]; e = None if errs is None else errs[g]
        ax.bar(x + (i - (n - 1) / 2) * w, v, w * 0.9, color=colors[g], yerr=e, error_kw=dict(lw=0.8, capsize=1.5, ecolor=TEXT2),
               label=labels[g] if labels else g, linewidth=0)
    ax.set_xticks(x); ax.set_xticklabels([c.replace(" ", "\n") for c in cats])


def fig_legend(fig, axes, ncol=7, extra=None):
    """one figure-level legend below the panels (entities keep their colour in every panel)."""
    seen, H, L = set(), [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l); H.append(h); L.append(l)
    fig.legend(H, L, loc="lower center", ncol=ncol, fontsize=7, bbox_to_anchor=(0.5, -0.01), handlelength=1.2, columnspacing=1.0)
    fig.tight_layout(rect=[0, 0.1, 1, 1])


# ------------------------------------------------------------------ dataset / setup numbers
def dataset_numbers(meta, splits_meta, overview, runs):
    d = overview["dataset"]
    resp = pd.read_csv(os.path.join(RAW, "GDSC2_fitted_dose_response_27Oct23.csv"), usecols=["DRUG_ID"])
    smi = pd.read_csv(os.path.join(DATA, "drug_smiles.csv"))
    mac("nDrugsGDSC", num(resp.DRUG_ID.nunique())); mac("nDrugsPubChem", num(smi.smiles.notna().sum()))
    mac("nDrugsSMILES", num(d["n_drugs"])); mac("nPairs", num(d["n_pairs"])); mac("nDrugs", num(d["n_drugs"])); mac("nCells", num(d["n_cells"]))
    dd = np.load(os.path.join(DATA, "dataset.npz"))
    mac("nCellsCrispr", num((~np.isnan(dd["crispr"]).all(1)).sum()))
    mac("nGenes", num(d["n_genes"])); mac("nTargetGenes", num(d["n_target_genes"])); mac("nDrugsTargets", num(d["n_drugs_with_targets"]))
    for v in VOCABS:
        with open(os.path.join(DATA, f"graphs_{v}.pkl"), "rb") as f:
            G = pickle.load(f)
        nm = np.array([g["n_motifs"] for g in G["graphs"]]); V = v.capitalize()
        mac(f"nVocab{V}", num(G["n_vocab"] - 1)); mac(f"motifs{V}", f2(nm.mean(), 1)); mac(f"motifs{V}Med", num(np.median(nm))); mac(f"motifs{V}Max", num(nm.max()))
        mac(f"nDrugsMultiMotif{V}", num((nm >= 2).sum()))
    mac("nProbeDrugs", num(d["n_probe_drugs"])); mac("nProbeCells", num(d["n_probe_cells"])); mac("nProbePairs", num(d["n_probe_pairs"]))
    mac("nProbePathways", num(len(set(splits_meta["probe_drug_pathways"]))))
    mac("hiddenDim", "128"); mac("nCnvTop", "300"); mac("topkMotif", str(TOPK_MOTIF)); mac("topkGene", str(TOPK_GENE))
    mac("perturbSigma", str(PERTURB_SIGMA)); mac("nSwapDonors", str(N_SWAP_DONORS)); mac("igSteps", str(IG_STEPS)); mac("gnnexpSteps", str(GNNEXP_STEPS))
    neural = runs[runs.model.isin(["ccmg", "latefusion"])] if len(runs) else runs
    mac("nTrainRuns", num(len(neural))); mac("nExplainRuns", num(len(overview["explained_runs"])))
    cc = neural[neural.model == "ccmg"] if len(neural) else neural
    mac("nParams", num(cc.params.iloc[0]) if len(cc) else "--")
    mac("runMinutes", f2(cc.train_time_s.median() / 60, 0) if len(cc) else "--")
    mac("epochsMedian", num(cc.epochs_run.median()) if len(cc) else "--")
    sp = pd.DataFrame({"regime": SPLITS})
    spl = np.load(os.path.join(DATA, "splits.npz"))
    for t in SPLITS:
        T = "".join(w.capitalize() for w in t.split("_"))
        for p in ("train", "val", "test"):
            mac(f"n{T}{p.capitalize()}", num(len(spl[f"{t}_ss0_{p}"])))


# ------------------------------------------------------------------ accuracy
def accuracy(acc, acc_res):
    if not len(acc): return
    acc = acc.copy(); acc["key"] = np.where(acc.model == "ccmg", "ccmg|" + acc.vocab.astype(str), acc.model)
    order = ["naive", "lgbm", "latefusion", "ccmg|brics", "ccmg|recap"]
    rows = []
    MK = {"naive": "Naive", "lgbm": "Lgbm", "latefusion": "Late", "ccmg|brics": "CcmgBrics", "ccmg|recap": "CcmgRecap"}
    for k in order:
        r = {"model": MLAB[k]}
        for s in SPLITS:
            g = acc[(acc.key == k) & (acc.split == s)]
            S = "".join(w.capitalize() for w in s.split("_"))
            if len(g):
                g = g.iloc[0]
                r[s] = (g.test_rmse, g.get("test_rmse_sd", 0), g.test_pearson, g.test_per_drug_pearson, g.get("test_per_drug_pearson_sd", 0), g.probe_rmse)
            for m, col in (("Rmse", "test_rmse"), ("R", "test_pearson"), ("Rd", "test_per_drug_pearson"), ("Rc", "test_per_cell_pearson"), ("PRmse", "probe_rmse"), ("RdSd", "test_per_drug_pearson_sd")):
                mac(f"acc{MK[k]}{S}{m}", f2(float(g[col]) if len(g) and col in g else np.nan, 3))
        rows.append(r)
    L = ["\\begin{table}[t]\\centering\\small\\caption{\\textbf{Accuracy under the DrEval-style protocol (fixed split, five training seeds).} RMSE and Pearson correlation over all test pairs and mean within-drug Pearson correlation ($r_{\\mathrm{drug}}$, drugs with $\\ge 10$ test pairs), mean $\\pm$ s.d.\\ over seeds (the naive predictor is deterministic). Probe RMSE is the RMSE on the \\nProbePairs{} probe pairs.}\\label{tab:acc}",
         "\\begin{adjustbox}{max width=\\textwidth}\\begin{tabular}{l" + "ccc" * 3 + "}\\toprule", " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{SLAB[s]}}}" for s in SPLITS) + "\\\\",
         " & ".join(["Model"] + ["RMSE", "$r$", "$r_{\\mathrm{drug}}$"] * 3) + "\\\\\\midrule"]
    for r in rows:
        cells = []
        for s in SPLITS:
            if s in r:
                rm, rsd, rp, rd, rdsd, pr = r[s]
                cells += [f2(rm, 3) + (f"\\,$\\pm$\\,{rsd:.3f}" if rsd > 0 else ""), f2(rp, 3), f2(rd, 3) + (f"\\,$\\pm$\\,{rdsd:.3f}" if rdsd > 0 and not np.isnan(rd) else "")]
            else:
                cells += ["--", "--", "--"]
        L.append(r["model"] + " & " + " & ".join(cells) + "\\\\")
    L += ["\\bottomrule\\end{tabular}\\end{adjustbox}\\end{table}"]
    open(os.path.join(PAPER, "tab_accuracy.tex"), "w", encoding="utf-8").write("\n".join(L))
    # resampled-split table (sensitivity)
    if len(acc_res):
        acc_res = acc_res.copy(); acc_res["key"] = np.where(acc_res.model == "ccmg", "ccmg|" + acc_res.vocab.astype(str), acc_res.model)
        L = ["\\begin{table}[t]\\centering\\small\\caption{\\textbf{Accuracy under re-drawn splits} (split seed $=$ training seed; mean $\\pm$ s.d.\\ over the available split seeds).}\\label{tab:accres}",
             "\\begin{tabular}{l" + "cc" * 3 + "}\\toprule", " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{SLAB[s]}}}" for s in SPLITS) + "\\\\",
             " & ".join(["Model"] + ["RMSE", "$r_{\\mathrm{drug}}$"] * 3) + "\\\\\\midrule"]
        for k in ["naive", "lgbm", "latefusion", "ccmg|brics"]:
            if acc_res[acc_res.key == k].n_split_seeds.max() < 2:
                continue  # models without re-drawn splits (late fusion) are not shown here
            cells = []
            for s in SPLITS:
                g = acc_res[(acc_res.key == k) & (acc_res.split == s)]
                if len(g):
                    g = g.iloc[0]
                    cells += [f2(g.test_rmse, 3) + (f"\\,$\\pm$\\,{g.test_rmse_sd:.3f}" if g.test_rmse_sd > 0 else ""),
                              f2(g.test_per_drug_pearson, 3) + (f"\\,$\\pm$\\,{g.test_per_drug_pearson_sd:.3f}" if g.test_per_drug_pearson_sd > 0 and not np.isnan(g.test_per_drug_pearson) else "")]
                else:
                    cells += ["--", "--"]
            L.append(MLAB[k] + " & " + " & ".join(cells) + "\\\\")
        L += ["\\bottomrule\\end{tabular}\\end{table}"]
        open(os.path.join(PAPER, "tab_accuracy_resampled.tex"), "w", encoding="utf-8").write("\n".join(L))
    # figure
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax, metric, ylab in zip(axes, ["test_per_drug_pearson", "test_rmse"], ["within-drug Pearson $r$", "RMSE ($\\ln$ IC$_{50}$)"]):
        vals = {k: [acc[(acc.key == k) & (acc.split == s)][metric].mean() if len(acc[(acc.key == k) & (acc.split == s)]) else np.nan for s in SPLITS] for k in order}
        errs = {k: [acc[(acc.key == k) & (acc.split == s)][metric + "_sd"].mean() if metric + "_sd" in acc and len(acc[(acc.key == k) & (acc.split == s)]) else 0 for s in SPLITS] for k in order}
        bars(ax, [SLAB[s] for s in SPLITS], order, vals, errs, MPAL, labels=MLAB); style(ax, ylab)
    fig_legend(fig, axes, ncol=5); fig.savefig(os.path.join(FIGS, "fig_accuracy.pdf"), bbox_inches="tight"); plt.close(fig)
    # headline numbers: gap to naive and to LightGBM on the within-drug correlation, per regime
    for s in SPLITS:
        T = "".join(w.capitalize() for w in s.split("_"))
        def get(k, m):
            g = acc[(acc.key == k) & (acc.split == s)]
            return float(g[m].iloc[0]) if len(g) else np.nan
        mac(f"gapNaive{T}", f2(get("ccmg|brics", "test_per_drug_pearson") - get("naive", "test_per_drug_pearson"), 3))
        mac(f"gapLgbm{T}", f2(get("ccmg|brics", "test_per_drug_pearson") - get("lgbm", "test_per_drug_pearson"), 3))
        mac(f"gapLate{T}", f2(get("ccmg|brics", "test_per_drug_pearson") - get("latefusion", "test_per_drug_pearson"), 3))
        mac(f"gapRmseLgbm{T}", f2(get("lgbm", "test_rmse") - get("ccmg|brics", "test_rmse"), 3))


# ------------------------------------------------------------------ stability
def stability(stab):
    if not len(stab): return pd.DataFrame()
    agg = stab.groupby(["axis", "vocab", "level", "explainer"]).agg(rho=("rho_mean", "mean"), rho_sd=("rho_mean", "std"), jac=("jaccard_mean", "mean"),
                                                                     jac_sd=("jaccard_mean", "std"), top1=("top1", "mean"), n=("rho_mean", "size")).reset_index()
    agg.to_csv(os.path.join(SUMMARY, "stability_agg.csv"), index=False)

    def cell(axis, vocab, level, ex, col):
        g = agg[(agg.axis == axis) & (agg.vocab == vocab) & (agg.level == level) & (agg.explainer == ex)]
        return float(g[col].iloc[0]) if len(g) else np.nan

    axes_m = ["seed", "split", "perturb"]
    # table: motif level (both vocabularies) and gene level (BRICS backbone; + vocabulary axis)
    L = ["\\begin{table}[t]\\centering\\small\\caption{\\textbf{Stability of explanations on the probe pairs.} Mean over pairs (and over the ten seed pairs, the three regime pairs or the runs) of the Spearman correlation $\\rho$ between the two score vectors and the Jaccard overlap $J$ of the top-$k$ sets ($k=\\topkMotif$ motifs, $k=\\topkGene$ genes). Seed: same split, different training seed. Regime: same seed, different exposure regime (random / unseen drug / unseen cell), same pairs. Input noise: Gaussian noise ($\\sigma=\\perturbSigma$) on expression and copy number. Vocabulary: BRICS versus RECAP backbone (gene level). The prediction row is the same comparison applied to the predicted $\\ln$ IC$_{50}$ of the probe pairs.}\\label{tab:stab}",
         "\\begin{adjustbox}{max width=\\textwidth}\\begin{tabular}{ll" + "cc" * 3 + "}\\toprule", "Level & Explainer & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{ALAB[a]}}}" for a in axes_m) + "\\\\",
         " & & " + " & ".join(["$\\rho$", "$J$"] * 3) + "\\\\\\midrule"]
    for vocab in VOCABS:
        first = True
        for ex in MOTIF_EX + ["gnnexp2", "random", "delta_del", "prediction"]:
            cells = []
            for a in axes_m:
                if ex == "prediction":
                    cells += [f2(cell(a, vocab, "y", ex, "rho")), "--"]
                else:
                    cells += [f2(cell(a, vocab, "motif", ex, "rho")), f2(cell(a, vocab, "motif", ex, "jac"))]
            L.append((f"motif, {vocab.upper()}" if first else "") + f" & {LAB[ex]} & " + " & ".join(cells) + "\\\\"); first = False
        L.append("\\midrule")
    first = True
    for ex in GENE_EX + ["prediction"]:
        cells = []
        for a in axes_m:
            if ex == "prediction":
                cells += [f2(cell(a, "brics", "y", ex, "rho")), "--"]
            else:
                cells += [f2(cell(a, "brics", "gene", ex, "rho")), f2(cell(a, "brics", "gene", ex, "jac"))]
        L.append(("gene, BRICS" if first else "") + f" & {LAB[ex]} & " + " & ".join(cells) + "\\\\"); first = False
    L += ["\\bottomrule\\end{tabular}\\end{adjustbox}\\end{table}"]
    open(os.path.join(PAPER, "tab_stability.tex"), "w", encoding="utf-8").write("\n".join(L))
    # macros (BRICS motif level, gene level)
    for a in axes_m:
        A = a.capitalize()
        for ex in MOTIF_EX + ["gnnexp2", "random", "delta_del", "prediction"]:
            lvl = "y" if ex == "prediction" else "motif"
            ex_m = "Ddel" if ex == "delta_del" else ex.capitalize()
            mac(f"stab{A}M{ex_m}Top", f2(cell(a, "brics", lvl, ex, "top1")))
            if ex == "delta_del":
                mac(f"stab{A}M{ex_m}Rho", f2(cell(a, "brics", lvl, ex, "rho"))); mac(f"stab{A}M{ex_m}Jac", f2(cell(a, "brics", lvl, ex, "jac")))
                mac(f"stab{A}M{ex_m}RhoRecap", f2(cell(a, "recap", lvl, ex, "rho"))); continue
            mac(f"stab{A}M{ex.capitalize()}Rho", f2(cell(a, "brics", lvl, ex, "rho"))); mac(f"stab{A}M{ex.capitalize()}Jac", f2(cell(a, "brics", lvl, ex, "jac")))
            mac(f"stab{A}M{ex.capitalize()}RhoRecap", f2(cell(a, "recap", lvl, ex, "rho")))
        for ex in GENE_EX:
            mac(f"stab{A}G{ex.capitalize()}Rho", f2(cell(a, "brics", "gene", ex, "rho"))); mac(f"stab{A}G{ex.capitalize()}Jac", f2(cell(a, "brics", "gene", ex, "jac")))
    for ex in GENE_EX:
        mac(f"stabVocabG{ex.capitalize()}Rho", f2(cell("vocab", "brics|recap", "gene", ex, "rho")))
    # motif level across vocabularies: atom overlap of the top motifs
    va = stab[(stab.axis == "vocab") & (stab.level == "motif_atoms")] if "level" in stab else pd.DataFrame()
    for ex in MOTIF_EX + ["gnnexp2", "delta_del"]:
        g_ = va[va.explainer == ex] if len(va) else va
        E = "Ddel" if ex == "delta_del" else ex.capitalize()
        mac(f"vocabAtomJac{E}", f2(g_.jaccard_mean.mean()) if len(g_) else "--")
        mac(f"vocabAtomOverlap{E}", f2(g_.overlap_mean.mean()) if len(g_) and "overlap_mean" in g_ else "--")
        mac(f"vocabAtomDisjoint{E}", pct(g_.frac_disjoint.mean()) if len(g_) and "frac_disjoint" in g_ else "--")
    # per-regime seed stability (BRICS) for the prediction, the attention and the deletion effect
    for split in SPLITS:
        S_ = "".join(w.capitalize() for w in split.split("_"))
        g_ = stab[(stab.axis == "seed") & (stab.vocab == "brics") & (stab.split == split)]
        for ex, E in (("prediction", "Prediction"), ("attn", "Attn"), ("delta_del", "Ddel"), ("mask", "Mask"), ("ig", "Ig"), ("rollout", "Rollout")):
            gg = g_[g_.explainer == ex]
            mac(f"stabSeed{S_}M{E}Rho", f2(gg.rho_mean.mean()) if len(gg) else "--")
            if ex != "prediction":
                mac(f"stabSeed{S_}M{E}Top", f2(gg.top1.mean()) if len(gg) else "--")
    mot = agg[(agg.level == "motif") & (agg.explainer.isin(MOTIF_EX)) & (agg.axis == "seed")]
    mac("stabSeedMotifRhoMin", f2(mot.rho.min()) if len(mot) else "--"); mac("stabSeedMotifRhoMax", f2(mot.rho.max()) if len(mot) else "--")
    best = mot.sort_values("rho", ascending=False).iloc[0] if len(mot) else None
    mac("stabSeedMotifBest", LAB[best.explainer] if best is not None else "--")
    worst = mot.sort_values("rho").iloc[0] if len(mot) else None
    mac("stabSeedMotifWorst", LAB[worst.explainer] if worst is not None else "--")
    gen = agg[(agg.level == "gene") & (agg.explainer.isin(GENE_EX)) & (agg.axis == "seed") & (agg.vocab == "brics")]
    mac("stabSeedGeneRhoMin", f2(gen.rho.min()) if len(gen) else "--"); mac("stabSeedGeneRhoMax", f2(gen.rho.max()) if len(gen) else "--")
    mac("stabSeedGeneJacMin", f2(gen.jac.min()) if len(gen) else "--"); mac("stabSeedGeneJacMax", f2(gen.jac.max()) if len(gen) else "--")
    # figure: rho by axis, explainer, both levels
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    for ax, (lvl, vocab, exs, ttl) in zip(axes, [("motif", "brics", MOTIF_EX + ["random"], "motif level, BRICS"), ("motif", "recap", MOTIF_EX + ["random"], "motif level, RECAP"), ("gene", "brics", GENE_EX, "gene level, BRICS")]):
        vals = {ex: [cell(a, vocab, lvl, ex, "rho") for a in axes_m] for ex in exs}
        errs = {ex: [cell(a, vocab, lvl, ex, "rho_sd") if not np.isnan(cell(a, vocab, lvl, ex, "rho_sd")) else 0 for a in axes_m] for ex in exs}
        bars(ax, [ALAB[a] for a in axes_m], exs, vals, errs, PAL, labels=LAB)
        pred = [cell(a, vocab, "y", "prediction", "rho") for a in axes_m]
        ax.plot(np.arange(3), pred, "_", color=TEXT, ms=14, mew=1.5, label="prediction")
        style(ax, "Spearman $\\rho$ between runs" if lvl == "motif" and vocab == "brics" else None); ax.set_title(ttl); ax.set_ylim(-0.1, 1.05); ax.axhline(0, color=TEXT2, lw=0.6)
    fig_legend(fig, axes, ncol=6); fig.savefig(os.path.join(FIGS, "fig_stability.pdf"), bbox_inches="tight"); plt.close(fig)
    return agg


# ------------------------------------------------------------------ faithfulness
def faithfulness(faith, cfstats):
    if not len(faith): return
    agg = faith.groupby(["vocab", "split", "counterfactual", "explainer"]).agg(rho=("rho_abs_mean", "mean"), rho_sd=("rho_abs_mean", "std"), top1=("top1_hit", "mean"),
                                                                                top1_sd=("top1_hit", "std"), fid=("fidelity_top1_frac_of_max", "mean"), rho_s=("rho_signed_mean", "mean"),
                                                                                sign=("sign_agree", "mean")).reset_index()
    agg.to_csv(os.path.join(SUMMARY, "faithfulness_agg.csv"), index=False)
    for c in ("rho_abs_valid", "rho_abs_graph", "top1_hit_validtop", "frac_top_delta_valid"):
        if c not in faith:
            faith[c] = np.nan
    pooled = faith.groupby(["vocab", "counterfactual", "explainer"]).agg(rho=("rho_abs_mean", "mean"), rho_sd=("rho_abs_mean", "std"), top1=("top1_hit", "mean"),
                                                                           fid=("fidelity_top1_frac_of_max", "mean"), rho_s=("rho_signed_mean", "mean"), sign=("sign_agree", "mean"),
                                                                           rho_v=("rho_abs_valid", "mean"), rho_g=("rho_abs_graph", "mean"), top1_v=("top1_hit_validtop", "mean"),
                                                                           frac_v=("frac_top_delta_valid", "mean")).reset_index()

    def cell(vocab, cf, ex, col, split=None):
        g = pooled[(pooled.vocab == vocab) & (pooled.counterfactual == cf) & (pooled.explainer == ex)] if split is None else \
            agg[(agg.vocab == vocab) & (agg.split == split) & (agg.counterfactual == cf) & (agg.explainer == ex)]
        return float(g[col].iloc[0]) if len(g) and col in g else np.nan

    exs = MOTIF_EX + ["gnnexp2", "size", "random"]
    L = ["\\begin{table}[t]\\centering\\small\\caption{\\textbf{Counterfactual faithfulness of motif explanations.} Agreement between each explainer's motif ranking and the model's own response to editing the molecule: $\\rho$ is the mean per-pair Spearman correlation between $|$score$|$ and $|\\Delta|$; top-1 is the fraction of pairs whose highest-scored motif is the motif with the largest $|\\Delta|$ (chance $=$ mean $1/n_{\\mathrm{motifs}}$); fidelity is the $|\\Delta_{\\mathrm{del}}|$ of the top-scored motif as a fraction of the largest $|\\Delta_{\\mathrm{del}}|$ of that drug. Pooled over the three regimes and five seeds; per-regime values in Fig.~\\ref{fig:faith}.}\\label{tab:faith}",
         "\\begin{tabular}{l" + "ccc" + "cc" + "ccc" + "cc" + "}\\toprule",
         " & \\multicolumn{5}{c}{BRICS} & \\multicolumn{5}{c}{RECAP}\\\\", " & \\multicolumn{3}{c}{deletion} & \\multicolumn{2}{c}{swap} & \\multicolumn{3}{c}{deletion} & \\multicolumn{2}{c}{swap}\\\\",
         "Explainer & $\\rho$ & top-1 & fidelity & $\\rho$ & top-1 & $\\rho$ & top-1 & fidelity & $\\rho$ & top-1\\\\\\midrule"]
    for ex in exs:
        cells = []
        for vocab in VOCABS:
            cells += [f2(cell(vocab, "delta_del", ex, "rho")), f2(cell(vocab, "delta_del", ex, "top1")), f2(cell(vocab, "delta_del", ex, "fid")),
                      f2(cell(vocab, "delta_swap", ex, "rho")), f2(cell(vocab, "delta_swap", ex, "top1"))]
        L.append(f"{LAB[ex]} & " + " & ".join(cells) + "\\\\")
    L.append("chance (top-1) & " + " & ".join(["--", f2(cell("brics", "chance", "chance", "top1")), "--", "--", f2(cell("brics", "chance", "chance", "top1")),
                                               "--", f2(cell("recap", "chance", "chance", "top1")), "--", "--", f2(cell("recap", "chance", "chance", "top1"))]) + "\\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}"]
    open(os.path.join(PAPER, "tab_faith.tex"), "w", encoding="utf-8").write("\n".join(L))
    for vocab in VOCABS:
        V = vocab.capitalize()
        for ex in exs:
            for cf, C in (("delta_del", "Del"), ("delta_swap", "Swap")):
                mac(f"faith{V}{C}{ex.capitalize()}Rho", f2(cell(vocab, cf, ex, "rho"))); mac(f"faith{V}{C}{ex.capitalize()}Top", f2(cell(vocab, cf, ex, "top1")))
                mac(f"faith{V}{C}{ex.capitalize()}Fid", f2(cell(vocab, cf, ex, "fid")))
            for s in SPLITS:
                S = "".join(w.capitalize() for w in s.split("_"))
                mac(f"faith{V}Del{ex.capitalize()}Rho{S}", f2(cell(vocab, "delta_del", ex, "rho", s)))
        mac(f"faith{V}ChanceTop", f2(cell(vocab, "chance", "chance", "top1")))
        for ex in exs:
            mac(f"faith{V}Del{ex.capitalize()}RhoValid", f2(cell(vocab, "delta_del", ex, "rho_v"))); mac(f"faith{V}Del{ex.capitalize()}RhoGraph", f2(cell(vocab, "delta_del", ex, "rho_g")))
            mac(f"faith{V}Del{ex.capitalize()}TopValid", f2(cell(vocab, "delta_del", ex, "top1_v")))
        mac(f"fracTopDeltaValid{V}", pct(cell(vocab, "delta_del", "attn", "frac_v")))
        for ex in ["ig", "mask"]:
            mac(f"faith{V}Del{ex.capitalize()}RhoSigned", f2(cell(vocab, "delta_del", ex, "rho_s"))); mac(f"faith{V}Del{ex.capitalize()}Sign", f2(cell(vocab, "delta_del", ex, "sign")))
        best = pooled[(pooled.vocab == vocab) & (pooled.counterfactual == "delta_del") & (pooled.explainer.isin(MOTIF_EX))].sort_values("rho", ascending=False)
        if len(best):
            mac(f"faith{V}Best", LAB[best.iloc[0].explainer]); mac(f"faith{V}Worst", LAB[best.iloc[-1].explainer])
    # valid-edit vs graph-edit table
    L = ["\\begin{table}[t]\\centering\\small\\caption{\\textbf{Valid-edit versus graph-edit deletions.} Agreement ($\\rho$, mean per-pair Spearman) between each explainer and the deletion effect when the comparison is restricted to motifs whose remainder sanitises as a molecule in RDKit (valid edit) or does not (graph edit: atoms removed, original atom features kept); and top-1 agreement on the pairs whose largest-effect motif is a valid edit. Pooled over regimes and seeds; -- where fewer than three graph-edit motifs per drug leave no per-pair correlation to compute (RECAP drugs have a median of three motifs).}\\label{tab:valid}",
         "\\begin{tabular}{lcccccc}\\toprule", " & \\multicolumn{3}{c}{BRICS} & \\multicolumn{3}{c}{RECAP}\\\\", "Explainer & $\\rho$ valid & $\\rho$ graph & top-1 (valid) & $\\rho$ valid & $\\rho$ graph & top-1 (valid)\\\\\\midrule"]
    for ex in exs:
        cells = []
        for vocab in VOCABS:
            cells += [f2(cell(vocab, "delta_del", ex, "rho_v")), f2(cell(vocab, "delta_del", ex, "rho_g")), f2(cell(vocab, "delta_del", ex, "top1_v"))]
        L.append(f"{LAB[ex]} & " + " & ".join(cells) + "\\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}"]
    open(os.path.join(PAPER, "tab_valid.tex"), "w", encoding="utf-8").write("\n".join(L))
    cd = read("cell_dependence.csv")
    if len(cd):
        cagg = cd.groupby(["vocab", "explainer"]).agg(modal=("modal_top1_frac", "mean"), rho=("rho_between_cells", "mean"),
                                                        amax=("attn_max_mean", "mean"), aent=("attn_norm_entropy", "mean"), a09=("attn_max_gt_09", "mean")).reset_index()
        cagg.to_csv(os.path.join(SUMMARY, "cell_dependence_agg.csv"), index=False)
        for vocab in VOCABS:
            V = vocab.capitalize()
            for ex in MOTIF_EX + ["gnnexp2", "delta_del"]:
                g_ = cagg[(cagg.vocab == vocab) & (cagg.explainer == ex)]
                E = "Ddel" if ex == "delta_del" else ex.capitalize()
                mac(f"cd{V}{E}Modal", pct(float(g_.modal.iloc[0])) if len(g_) else "--"); mac(f"cd{V}{E}Rho", f2(float(g_.rho.iloc[0])) if len(g_) else "--")
            g_ = cagg[(cagg.vocab == vocab) & (cagg.explainer == "attn")]
            mac(f"attnMax{V}", f2(float(g_.amax.iloc[0])) if len(g_) else "--"); mac(f"attnEntropy{V}", f2(float(g_.aent.iloc[0])) if len(g_) else "--")
            mac(f"attnPeaked{V}", pct(float(g_.a09.iloc[0])) if len(g_) else "--")
        L = ["\\begin{table}[t]\\centering\\small\\caption{\\textbf{How cell-conditioned are the motif explanations?} For each probe drug and explainer: the fraction of the drug's probe cells that share the modal top-1 motif (1 $=$ the same motif in every cell) and the mean Spearman correlation between the motif score vectors of two different probe cells of the same drug; mean over drugs, pooled over regimes and seeds. The last row is the model's own counterfactual deletion effect.}\\label{tab:celldep}",
             "\\begin{tabular}{lcccc}\\toprule", " & \\multicolumn{2}{c}{BRICS} & \\multicolumn{2}{c}{RECAP}\\\\", "Explainer & same top-1 & $\\rho$ between cells & same top-1 & $\\rho$ between cells\\\\\\midrule"]
        for ex in MOTIF_EX + ["gnnexp2", "delta_del"]:
            cells = []
            for vocab in VOCABS:
                g_ = cagg[(cagg.vocab == vocab) & (cagg.explainer == ex)]
                cells += [f2(float(g_.modal.iloc[0])) if len(g_) else "--", f2(float(g_.rho.iloc[0])) if len(g_) else "--"]
            L.append(f"{LAB[ex]} & " + " & ".join(cells) + "\\\\")
        L += ["\\bottomrule\\end{tabular}\\end{table}"]
        open(os.path.join(PAPER, "tab_celldep.tex"), "w", encoding="utf-8").write("\n".join(L))
    if len(cfstats):
        mac("fracChemValid", pct(cfstats.frac_chem_valid.mean())); mac("meanAbsDel", f2(cfstats.mean_abs_del.mean())); mac("meanAbsSwap", f2(cfstats.mean_abs_swap.mean()))
        mac("rhoSizeDel", f2(cfstats.rho_size_del.mean())); mac("rhoDelSwap", f2(cfstats.rho_del_swap.mean())); mac("fracDelIncrease", pct(cfstats.frac_del_increase_ic50.mean()))
        mac("ySdProbe", f2(cfstats.y_sd_probe.mean()))
        for vocab in VOCABS:
            g = cfstats[cfstats.vocab == vocab]
            mac(f"meanAbsDel{vocab.capitalize()}", f2(g.mean_abs_del.mean()) if len(g) else "--"); mac(f"rhoSizeDel{vocab.capitalize()}", f2(g.rho_size_del.mean()) if len(g) else "--")
    # figure: per-regime rho (deletion) for both vocabularies + top-1 vs chance
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, vocab in zip(axes[:2], VOCABS):
        vals = {ex: [cell(vocab, "delta_del", ex, "rho", s) for s in SPLITS] for ex in exs}
        errs = {ex: [cell(vocab, "delta_del", ex, "rho_sd", s) if not np.isnan(cell(vocab, "delta_del", ex, "rho_sd", s)) else 0 for s in SPLITS] for ex in exs}
        bars(ax, [SLAB[s] for s in SPLITS], exs, vals, errs, PAL, labels=LAB); style(ax, "$\\rho$(|score|, $|\\Delta_{\\mathrm{del}}|$)" if vocab == "brics" else None)
        ax.set_title(f"deletion, {vocab.upper()}"); ax.axhline(0, color=TEXT2, lw=0.6); ax.set_ylim(-0.3, 1.0)
    ax = axes[2]
    vals = {ex: [cell(v, "delta_del", ex, "top1") for v in VOCABS] for ex in exs}
    bars(ax, [v.upper() for v in VOCABS], exs, vals, None, PAL, labels=LAB)
    for i, v in enumerate(VOCABS):
        ax.plot([i - 0.45, i + 0.45], [cell(v, "chance", "chance", "top1")] * 2, color=TEXT, lw=1.2, ls="--", label="chance" if i == 0 else None)
    style(ax, "top-1 agreement"); ax.set_title("top-1 agreement, deletion"); ax.set_ylim(0, 1)
    fig_legend(fig, axes, ncol=8); fig.savefig(os.path.join(FIGS, "fig_faith.pdf"), bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ biology
def biology(bio, dep, mbio, coh, overview):
    if not len(bio): return
    bagg = bio.groupby(["vocab", "explainer"]).agg(hit=("target_in_top20", "mean"), hit_sd=("target_in_top20", "std"), chance=("chance_top20", "mean"),
                                                    rank=("best_target_rank_frac", "mean"), n=("n_pairs_with_target", "mean")).reset_index()
    dagg = dep.groupby(["vocab", "explainer"]).agg(rho=("rho_abs", "mean"), rho_med=("rho_abs", "median"), rho_s=("rho_signed", "mean"), n=("rho_abs", "size"),
                                                    frac_pos=("rho_abs", lambda x: float(np.mean(x > 0)))).reset_index() if len(dep) else pd.DataFrame()
    magg = mbio.groupby(["vocab", "explainer"]).agg(hit=("top1_in_enriched", "mean"), hit_sd=("top1_in_enriched", "std"), chance=("chance", "mean"), n=("n_pairs", "mean")).reset_index() if len(mbio) else pd.DataFrame()
    for df_, nm in ((bagg, "bio_gene_agg"), (dagg, "bio_dependency_agg"), (magg, "bio_motif_agg")):
        df_.to_csv(os.path.join(SUMMARY, nm + ".csv"), index=False)

    def g(df_, vocab, ex, col):
        if not len(df_): return np.nan
        r = df_[(df_.vocab == vocab) & (df_.explainer == ex)]
        return float(r[col].iloc[0]) if len(r) else np.nan

    gene_exs = GENE_EX + ["kd"]
    L = ["\\begin{table}[t]\\centering\\small\\caption{\\textbf{Agreement with drug--target biology} (BRICS backbone, pooled over regimes and seeds; RECAP in the text). Gene level: fraction of probe pairs (drugs with mapped GDSC targets) with at least one annotated target among the top-\\topkGene{} genes, against the hypergeometric chance; mean rank fraction of the best-ranked target (0 $=$ first); dependency agreement $\\rho_{\\mathrm{dep}}$ is the Spearman correlation across probe cells between $|$attribution of the target gene$|$ and the cell's CRISPR dependency on that target, averaged over (drug, target, run) triples with $\\ge 20$ screened cells, with the fraction of positive correlations. Motif level: fraction of probe pairs whose top-ranked motif is pathway-enriched, against the enriched fraction of the drug's motifs.}\\label{tab:bio}",
         "\\begin{tabular}{llcccc}\\toprule", "Level & Explainer & target in top-\\topkGene{} & chance & best-target rank & $\\rho_{\\mathrm{dep}}$ (frac.\\ $>0$)\\\\\\midrule"]
    first = True
    for ex in gene_exs:
        L.append(("gene" if first else "") + f" & {LAB[ex]} & {f2(g(bagg, 'brics', ex, 'hit'))} & {f2(g(bagg, 'brics', ex, 'chance'))} & {f2(g(bagg, 'brics', ex, 'rank'))} & "
                 f"{f2(g(dagg, 'brics', ex, 'rho'))} ({f2(g(dagg, 'brics', ex, 'frac_pos'))})\\\\"); first = False
    L += ["\\midrule", "Level & Explainer & top-1 motif enriched & chance & & \\\\\\midrule"]
    first = True
    for ex in MOTIF_EX + ["size", "random", "delta_del", "delta_swap"]:
        L.append(("motif, BRICS" if first else "") + f" & {LAB[ex]} & {f2(g(magg, 'brics', ex, 'hit'))} & {f2(g(magg, 'brics', ex, 'chance'))} & & \\\\"); first = False
    L += ["\\bottomrule\\end{tabular}\\end{table}"]
    open(os.path.join(PAPER, "tab_bio.tex"), "w", encoding="utf-8").write("\n".join(L))
    for vocab in VOCABS:
        V = vocab.capitalize()
        for ex in gene_exs:
            E = ex.capitalize()
            mac(f"bio{V}G{E}Hit", pct(g(bagg, vocab, ex, "hit"), 1)); mac(f"bio{V}G{E}Chance", pct(g(bagg, vocab, ex, "chance"), 1)); mac(f"bio{V}G{E}Rank", f2(g(bagg, vocab, ex, "rank")))
            mac(f"bio{V}G{E}Dep", f2(g(dagg, vocab, ex, "rho"))); mac(f"bio{V}G{E}DepPos", f2(g(dagg, vocab, ex, "frac_pos")))
        for ex in MOTIF_EX + ["size", "random", "delta_del", "delta_swap"]:
            E = ex.replace("delta_", "D").capitalize() if ex.startswith("delta") else ex.capitalize()
            mac(f"bio{V}M{E}Hit", pct(g(magg, vocab, ex, "hit"))); mac(f"bio{V}M{E}Chance", pct(g(magg, vocab, ex, "chance")))
        mn = magg[(magg.vocab == vocab)]
        mac(f"bio{V}MPairs", num(mn.n.mean()) if len(mn) else "--")
    bn = bagg[bagg.vocab == "brics"]
    mac("bioGenePairsWithTarget", num(bn.n.mean()) if len(bn) else "--")
    mac("bioDepTriples", num(dagg[(dagg.vocab == "brics") & (dagg.explainer == "ig")].n.iloc[0]) if len(dagg) and len(dagg[(dagg.vocab == "brics") & (dagg.explainer == "ig")]) else "--")
    ne = overview.get("n_enriched_motifs_per_pathway", {})
    for v in VOCABS:
        mac(f"nEnrichedPathways{v.capitalize()}", num(sum(1 for k, n in ne.get(v, {}).items() if n > 0))); mac(f"nEnrichedMotifs{v.capitalize()}", num(sum(ne.get(v, {}).values())))
    if len(coh):
        c = coh[coh.vocab == "brics"]
        mac("cohRhoDel", f2(c.rho_del_dep.mean())); mac("cohRhoAbsDel", f2(c.rho_absdel_dep.mean())); mac("cohRhoPred", f2(c.rho_pred_dep.mean())); mac("cohRhoTrue", f2(c.rho_true_dep.mean()))
        mac("cohN", num(len(c))); mac("cohFracPosPred", pct(np.mean(c.rho_pred_dep > 0))); mac("cohFracPosDel", pct(np.mean(c.rho_absdel_dep > 0)))
        mac("cohNDrugs", num(c.drug.nunique()))
        cr = coh[coh.vocab == "recap"]
        mac("cohRhoAbsDelRecap", f2(cr.rho_absdel_dep.mean()) if len(cr) else "--")
    # figure
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    ax = axes[0]
    vals = {ex: [g(bagg, v, ex, "hit") for v in VOCABS] for ex in gene_exs}
    bars(ax, [v.upper() for v in VOCABS], gene_exs, vals, None, PAL, labels=LAB)
    for i, v in enumerate(VOCABS):
        ax.plot([i - 0.45, i + 0.45], [g(bagg, v, "ig", "chance")] * 2, color=TEXT, lw=1.2, ls="--", label="chance" if i == 0 else None)
    style(ax, "pairs with a target in top-20"); ax.set_title("target recovery (genes)")
    ax.set_ylim(0, max(0.1, 2.5 * np.nanmax([np.nanmax(v) for v in vals.values()] + [0.04])))
    ax = axes[1]
    if len(dep):
        d = dep[dep.vocab == "brics"]
        data = [d[d.explainer == ex].rho_abs.dropna().to_numpy() for ex in gene_exs]
        vp = ax.violinplot([x if len(x) else [0] for x in data], positions=np.arange(len(gene_exs)), showmedians=True, widths=0.8)
        for b, ex in zip(vp["bodies"], gene_exs):
            b.set_facecolor(PAL[ex]); b.set_edgecolor("none"); b.set_alpha(0.8)
        for k in ("cmedians", "cmins", "cmaxes", "cbars"):
            vp[k].set_color(TEXT2); vp[k].set_linewidth(0.8)
        ax.set_xticks(np.arange(len(gene_exs))); ax.set_xticklabels([LAB[e].replace("integrated gradients", "IG").replace("knockdown $\\Delta_{\\mathrm{kd}}$", "$\\Delta_{\\mathrm{kd}}$").replace("gradient$\\times$input", "grad$\\times$in") for e in gene_exs], fontsize=7)
        ax.axhline(0, color=TEXT2, lw=0.6)
    style(ax, "$\\rho_{\\mathrm{dep}}$ per (drug, target, run)"); ax.set_title("attribution vs CRISPR dependency")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax = axes[2]
    mexs = MOTIF_EX + ["random", "delta_del"]
    vals = {ex: [g(magg, v, ex, "hit") for v in VOCABS] for ex in mexs}
    bars(ax, [v.upper() for v in VOCABS], mexs, vals, None, PAL, labels=LAB)
    for i, v in enumerate(VOCABS):
        ax.plot([i - 0.45, i + 0.45], [g(magg, v, "attn", "chance")] * 2, color=TEXT, lw=1.2, ls="--", label="chance" if i == 0 else None)
    style(ax, "top-1 motif pathway-enriched"); ax.set_title("enriched-motif recovery (motifs)"); ax.set_ylim(0, 1)
    fig_legend(fig, [axes[0], axes[2]], ncol=7); fig.savefig(os.path.join(FIGS, "fig_bio.pdf"), bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ example drug
def example(meta, run_pref="ccmg_brics_random_ss0_s0", drug_name="Lapatinib"):
    runs = sorted(glob.glob(os.path.join(EXPL, "ccmg_brics_random_ss0_s*", "scores.npz")))
    if not runs: return
    f = os.path.join(EXPL, run_pref, "scores.npz"); f = f if os.path.exists(f) else runs[0]
    S = dict(np.load(f))
    with open(os.path.join(DATA, "graphs_brics.pkl"), "rb") as fh:
        graphs = pickle.load(fh)["graphs"]
    names = [d["name"] for d in meta["drugs"]]
    di = names.index(drug_name) if drug_name in names and (S["d_idx"] == names.index(drug_name)).any() else int(S["d_idx"][0])
    sel = np.where(S["d_idx"] == di)[0]; g = graphs[di]; nm = g["n_motifs"]
    rows = [("attention", np.abs(S["motif_attn"][sel, :nm])), ("rollout", np.abs(S["motif_rollout"][sel, :nm])), ("integrated gradients", np.abs(S["motif_ig"][sel, :nm])),
            ("masking", np.abs(S["motif_mask"][sel, :nm])), ("GNNExplainer", S["motif_gnnexp"][sel, :nm]), ("$|\\Delta_{\\mathrm{del}}|$", np.abs(S["delta_del"][sel, :nm])),
            ("$|\\Delta_{\\mathrm{swap}}|$", np.abs(S["delta_swap"][sel, :nm]))]
    # per pair normalisation (share of total), then mean over the drug's probe cells
    H = np.stack([np.nanmean(r / np.nansum(r, 1, keepdims=True), 0) for _, r in rows])
    # fraction of cells in which each motif is top-1
    T = np.stack([np.bincount(np.nanargmax(np.where(np.isnan(r), -np.inf, r), 1), minlength=nm) / len(sel) for _, r in rows])
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"] + ["#9a9a9a"] * 20
    fig = plt.figure(figsize=(7.2, 3.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.55)
    ax0 = fig.add_subplot(gs[0]); ax0.axis("off")
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
        mol = Chem.MolFromSmiles(g["smiles"])
        hex2rgb = lambda h: tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))
        acol = {a: hex2rgb(cols[int(g["a2m"][a])]) for a in range(mol.GetNumAtoms())}
        dr = rdMolDraw2D.MolDraw2DCairo(900, 620)
        opts = dr.drawOptions(); opts.useBWAtomPalette(); opts.fillHighlights = True; opts.highlightRadius = 0.42; opts.bondLineWidth = 2
        dr.DrawMolecule(mol, highlightAtoms=list(range(mol.GetNumAtoms())), highlightAtomColors=acol, highlightBonds=[])
        dr.FinishDrawing()
        png = os.path.join(FIGS, "example_mol.png"); open(png, "wb").write(dr.GetDrawingText())
        ax0.imshow(plt.imread(png)); ax0.set_title(f"{names[di]} ({nm} BRICS motifs)", fontsize=9)
    except Exception as e:
        ax0.text(0.5, 0.5, f"structure drawing unavailable\n{e}", ha="center", va="center", fontsize=7)
    for ax, M, ttl in ((fig.add_subplot(gs[1]), H, "mean score share"), (fig.add_subplot(gs[2]), T, "top-1 frequency")):
        im = ax.imshow(M, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("blues", ["#ffffff", "#cde2fb", "#3987e5", "#0d366b"]), vmin=0, vmax=max(0.5, np.nanmax(M)), aspect="auto")
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows] if ttl.startswith("mean") else [""] * len(rows), fontsize=7)
        ax.set_xticks(range(nm)); ax.set_xticklabels([f"m{m + 1}" for m in range(nm)], fontsize=7)
        for t, m in zip(ax.get_xticklabels(), range(nm)): t.set_color(cols[m]); t.set_fontweight("bold")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color="white" if M[i, j] > 0.6 * max(0.5, np.nanmax(M)) else TEXT)
        ax.set_title(ttl, fontsize=8); ax.tick_params(length=0)
        for s in ("top", "right", "left", "bottom"): ax.spines[s].set_visible(False)
    fig.savefig(os.path.join(FIGS, "fig_example.pdf"), bbox_inches="tight"); plt.close(fig)
    mac("exampleDrug", names[di]); mac("exampleNMotifs", str(nm)); mac("exampleNCells", str(len(sel)))
    mac("exampleTargets", ", ".join(meta["drugs"][di]["targets"]) or "none")


def tag_label(t):
    """human-readable label of a run tag such as lr0.0003 or hid64-do0.3."""
    def sci(v):
        m, e = f"{float(v):.0e}".split("e")
        return f"{m}\\times10^{{{int(e)}}}"
    parts = []
    for p in str(t).split("-"):
        if p == "default": parts.append("default")
        elif p.startswith("hid"): parts.append(f"hidden size {p[3:]}")
        elif p.startswith("do"): parts.append(f"dropout {p[2:]}")
        elif p.startswith("lr"): parts.append(f"learning rate ${sci(p[2:])}$")
        elif p.startswith("wd"): parts.append(f"weight decay ${sci(p[2:])}$")
        else: parts.append(p)
    return ", ".join(parts)


def revision_tables(runs):
    """drug-level seed agreement, attention controls / tuned configuration, tuning grid, seed ensemble, entropy curve."""
    # 1. drug-level seed agreement (main runs)
    dsa = read("drug_seed_agreement.csv")
    if len(dsa):
        d0 = dsa[(dsa.split_seed == 0) & (dsa.tag.fillna("") == "")]
        for vocab in VOCABS:
            V = vocab.capitalize()
            for ex in MOTIF_EX + ["gnnexp2", "random", "delta_del"]:
                E = "Ddel" if ex == "delta_del" else ex.capitalize()
                g = d0[(d0.vocab == vocab) & (d0.explainer == ex)]
                gr = g[g.split == "random"]
                mac(f"dsa{V}{E}All", pct(gr.frac_all_agree.mean()) if len(gr) else "--"); mac(f"dsa{V}{E}Pair", pct(gr.pairwise_agree.mean()) if len(gr) else "--")
                mac(f"dsa{V}{E}AllPooled", pct(g.frac_all_agree.mean()) if len(g) else "--"); mac(f"dsa{V}{E}PairPooled", pct(g.pairwise_agree.mean()) if len(g) else "--")
        L = ["\\begin{table}[t]\\centering\\small\\caption{\\textbf{Drug-level reproducibility of the top motif across training seeds.} For every run the modal top-1 motif of each probe drug over its probe cells is recorded; \\emph{all five} is the fraction of the \\nProbeDrugs{} probe drugs whose modal motif is identical in all five seeds, \\emph{pairwise} the mean agreement over the ten seed pairs. Fixed split; random regime and pooled over the three regimes.}\\label{tab:drugagree}",
             "\\begin{adjustbox}{max width=\\textwidth}\\begin{tabular}{l" + "cccc" * 2 + "}\\toprule",
             " & \\multicolumn{4}{c}{BRICS} & \\multicolumn{4}{c}{RECAP}\\\\", " & \\multicolumn{2}{c}{random} & \\multicolumn{2}{c}{all regimes} & \\multicolumn{2}{c}{random} & \\multicolumn{2}{c}{all regimes}\\\\",
             "Explainer & all five & pairwise & all five & pairwise & all five & pairwise & all five & pairwise\\\\\\midrule"]
        for ex in MOTIF_EX + ["gnnexp2", "random", "delta_del"]:
            cells = []
            for vocab in VOCABS:
                g = d0[(d0.vocab == vocab) & (d0.explainer == ex)]; gr = g[g.split == "random"]
                cells += [pct(gr.frac_all_agree.mean()) if len(gr) else "--", pct(gr.pairwise_agree.mean()) if len(gr) else "--",
                          pct(g.frac_all_agree.mean()) if len(g) else "--", pct(g.pairwise_agree.mean()) if len(g) else "--"]
            L.append(f"{LAB[ex]} & " + " & ".join(cells) + "\\\\")
        L += ["\\bottomrule\\end{tabular}\\end{adjustbox}\\end{table}"]
        open(os.path.join(PAPER, "tab_drugagree.tex"), "w", encoding="utf-8").write("\n".join(L))
    # 2. attention controls and tuned configuration
    var = read("variants.csv")
    if len(var):
        def vget(tag, col):
            g = var[var.tag.fillna("") == tag]
            return float(g[col].iloc[0]) if len(g) and col in g and not pd.isna(g[col].iloc[0]) else np.nan
        tuned_tags = [t for t in var.tag.fillna("") if t not in ("", "h4", "ent0.05")]
        tuned = tuned_tags[0] if tuned_tags else None
        mac("tunedTag", tag_label(tuned) if tuned else "the default")
        for tag, T in (("", "Def"), ("h4", "Hfour"), ("ent0.05", "Ent"), (tuned, "Tuned")):
            if tag is None:
                for col, C in (("r_drug", "Rd"), ("attn_max", "AttnMax"), ("cd_modal_attn", "CdAttn"), ("seed_rho_attn", "SeedAttn"), ("seed_rho_delta_del", "SeedDdel"),
                               ("drug_agree_attn", "DrugAttn"), ("drug_agree_delta_del", "DrugDdel"), ("faith_rho_attn", "FaithAttn"), ("faith_rho_rollout", "FaithRollout"),
                               ("faith_rho_ig", "FaithIg"), ("attn_gt09", "AttnPeaked"), ("attn_norm_entropy", "AttnEnt")):
                    mac(f"var{T}{C}", "--")
                continue
            for col, C in (("r_drug", "Rd"), ("attn_max", "AttnMax"), ("cd_modal_attn", "CdAttn"), ("seed_rho_attn", "SeedAttn"), ("seed_rho_delta_del", "SeedDdel"),
                           ("faith_rho_attn", "FaithAttn"), ("faith_rho_rollout", "FaithRollout"), ("faith_rho_ig", "FaithIg"), ("attn_norm_entropy", "AttnEnt")):
                mac(f"var{T}{C}", f2(vget(tag, col)))
            for col, C in (("drug_agree_attn", "DrugAttn"), ("drug_agree_delta_del", "DrugDdel"), ("attn_gt09", "AttnPeaked"), ("cd_modal_attn", "CdAttnPct")):
                mac(f"var{T}{C}", pct(vget(tag, col)))
        rows = [("", "default (1 head)"), ("h4", "4 heads"), ("ent0.05", "entropy bonus")]
        if tuned:
            rows.append((tuned, "tuned (" + tag_label(tuned) + ")"))
        L = ["\\begin{table}[t]\\centering\\small\\caption{\\textbf{Attention controls and tuned backbone (random regime, BRICS, five seeds each).} $r_{\\mathrm{drug}}$: within-drug test correlation. Max $\\alpha$: mean largest attention weight (the fraction of pairs above 0.9 and the normalised entropy are given in the text). Same top-1: fraction of a drug's cells sharing the top attention motif. $\\rho_{\\mathrm{seed}}$: pair-level Spearman correlation across seeds of the attention scores and of the deletion effects. Drug agreement: pairwise agreement of the modal top attention motif across seeds. Faithfulness: mean per-pair $\\rho$ between $|$score$|$ and $|\\Delta_{\\mathrm{del}}|$ for attention and for rollout.}\\label{tab:variants}",
             "\\begin{adjustbox}{max width=\\textwidth}\\begin{tabular}{lcccccccc}\\toprule",
             "Configuration & $r_{\\mathrm{drug}}$ & max $\\alpha$ & same top-1 & $\\rho_{\\mathrm{seed}}$ attn & $\\rho_{\\mathrm{seed}}$ $\\Delta_{\\mathrm{del}}$ & drug agreement & faith.\\ attn & faith.\\ rollout\\\\\\midrule"]
        for tag, lab in rows:
            L.append(f"{lab} & {f2(vget(tag, 'r_drug'), 3)} & {f2(vget(tag, 'attn_max'))} & {pct(vget(tag, 'cd_modal_attn'))} & "
                     f"{f2(vget(tag, 'seed_rho_attn'))} & {f2(vget(tag, 'seed_rho_delta_del'))} & {pct(vget(tag, 'drug_agree_attn'))} & "
                     f"{f2(vget(tag, 'faith_rho_attn'))} & {f2(vget(tag, 'faith_rho_rollout'))}\\\\")
        L += ["\\bottomrule\\end{tabular}\\end{adjustbox}\\end{table}"]
        open(os.path.join(PAPER, "tab_variants.tex"), "w", encoding="utf-8").write("\n".join(L))
    # 3. tuning grid
    tg = read("runs_tagged.csv")
    if len(tg):
        grid = tg[(tg.split == "random") & (tg.seed == 0) & (~tg.tag.isin(["h4", "ent0.05"]))].copy()
        base = runs[(runs.model == "ccmg") & (runs.vocab == "brics") & (runs.split == "random") & (runs.seed == 0) & (runs.split_seed == 0) & (runs.tag.fillna("") == "")]
        if len(base):
            b = base.iloc[0]
            grid = pd.concat([pd.DataFrame([{"tag": "default", "hidden": 128, "dropout": 0.1, "lr": 1e-3, "wd": 1e-5, "best_val_rmse": b.best_val_rmse,
                                              "test_rmse": b.test_rmse, "test_per_drug_pearson": b.test_per_drug_pearson, "epochs_run": b.epochs_run}]), grid], ignore_index=True)
        grid = grid.sort_values("best_val_rmse")
        mac("nTuningConfigs", num(len(grid))); mac("tuningBestTag", str(grid.iloc[0].tag).replace("-", ", ")); mac("tuningBestVal", f2(grid.iloc[0].best_val_rmse, 3))
        mac("nTuningConfigsGrid", num(len(grid) - 1))
        mac("tuningDefaultVal", f2(float(base.iloc[0].best_val_rmse), 3) if len(base) else "--"); mac("tuningWorstVal", f2(grid.iloc[-1].best_val_rmse, 3))
        mac("tuningRdMin", f2(grid.test_per_drug_pearson.min(), 3)); mac("tuningRdMax", f2(grid.test_per_drug_pearson.max(), 3))
        L = ["\\begin{table}[t]\\centering\\small\\caption{\\textbf{Backbone tuning grid} (random regime, seed 0, fixed split): best validation RMSE, test RMSE and within-drug test correlation for each configuration; the configuration with the lowest validation RMSE was re-trained with five seeds (Table~\\ref{tab:variants}).}\\label{tab:tuning}",
             "\\begin{tabular}{lcccccc}\\toprule", "Configuration & hidden & dropout & lr & weight decay & val RMSE & test RMSE / $r_{\\mathrm{drug}}$\\\\\\midrule"]
        for _, r in grid.iterrows():
            L.append(f"{tag_label(r.tag)} & {int(r.hidden)} & {r.dropout:g} & {r.lr:g} & {r.wd:g} & {r.best_val_rmse:.3f} & {r.test_rmse:.3f} / {r.test_per_drug_pearson:.3f}\\\\")
        L += ["\\bottomrule\\end{tabular}\\end{table}"]
        open(os.path.join(PAPER, "tab_tuning.tex"), "w", encoding="utf-8").write("\n".join(L))
    # 4. seed ensemble
    ens = read("accuracy_ensemble.csv")
    if len(ens):
        for _, r in ens.iterrows():
            S_ = "".join(w.capitalize() for w in r.split.split("_")); M = {"ccmg-ens": "CcmgEns" + r.vocab.capitalize(), "latefusion-ens": "LateEns"}.get(r.model, r.model)
            mac(f"acc{M}{S_}Rd", f2(r.test_per_drug_pearson, 3)); mac(f"acc{M}{S_}Rmse", f2(r.test_rmse, 3))
    # 5. attention entropy against training epoch (from the run histories)
    import glob as _glob, json as _json
    curves = {}
    for tag, lab in (("", "default (1 head)"), ("h4", "4 heads"), ("ent0.05", "entropy bonus")):
        for seed in SEEDS:
            f = os.path.join(RUNS, run_name("ccmg", "brics", "random", seed, 0, tag), "metrics.json")
            if os.path.exists(f):
                h = _json.load(open(f))["history"]
                if h and "val_attn_entropy" in h[0]:
                    curves.setdefault(lab, []).append([e["val_attn_entropy"] for e in h])
    if curves:
        fig, ax = plt.subplots(figsize=(3.6, 2.5))
        cols = {"default (1 head)": PAL["attn"], "4 heads": PAL["rollout"], "entropy bonus": PAL["ig"]}
        for lab, cs in curves.items():
            for i, c in enumerate(cs):
                ax.plot(range(1, len(c) + 1), c, color=cols[lab], lw=1.2, alpha=0.8, label=lab if i == 0 else None)
        style(ax, "normalised attention entropy", "epoch"); ax.set_ylim(0, 1); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_entropy.pdf")); plt.close(fig)
        mac("entropyFigAvailable", "yes")


def control_tables():
    """positive-control biomarkers, curated pharmacophores and the planted-importance sanity check."""
    EXL = {"ig": "IG", "sal": "grad$\\times$input", "mask": "masking", "gnnexp": "GNNExplainer", "kd": "$\\Delta_{\\mathrm{kd}}$"}
    bio = read("bio_positive_controls.csv")
    if len(bio):
        b = bio[(bio.vocab == "brics") & (bio.split == "random")]
        agg = b.groupby(["drug", "gene", "block", "direction"]).mean(numeric_only=True).reset_index()
        agg.to_csv(os.path.join(SUMMARY, "bio_positive_controls_agg.csv"), index=False)
        mac("nBiomarkerPairs", num(len(agg)))
        mac("bioPcTrueSignOk", num(int(round(agg.true_sign_ok.sum())))); mac("bioPcPredSignOk", num(int(round(agg.pred_sign_ok.sum()))))
        strong = agg[(agg.p_true < 0.05) & (agg.true_sign_ok > 0.5)]
        mac("nBiomarkerPairsStrong", num(len(strong)))
        for ex in ("ig", "mask", "kd", "sal", "gnnexp"):
            E = ex.capitalize()
            mac(f"bioPcRank{E}", f2(agg[f"rank_{ex}"].median())); mac(f"bioPcTop{E}", pct(agg[f"top20_{ex}"].mean()))
            mac(f"bioPcRank{E}Strong", f2(strong[f"rank_{ex}"].median()) if len(strong) else "--"); mac(f"bioPcTop{E}Strong", pct(strong[f"top20_{ex}"].mean()) if len(strong) else "--")
            mac(f"bioPcMag{E}", f2(agg[f"mag_assoc_{ex}"].mean()))
            if f"sign_ok_{ex}" in agg:
                mac(f"bioPcSign{E}", pct(agg[f"sign_ok_{ex}"].mean()))
        mac("bioPcChanceTop", pct(TOPK_GENE / 1141.0, 1))
        def pair(drug, gene, block):
            g = agg[(agg.drug == drug) & (agg.gene == gene) & (agg.block == block)]
            return g.iloc[0] if len(g) else None
        for key, (drug, gene, block) in {"NutlinTp": ("Nutlin-3a (-)", "TP53", "mut"), "WeeTp": ("Wee1 Inhibitor", "TP53", "mut"),
                                         "AfuPik": ("Afuresertib", "PIK3CA", "mut"), "SelBraf": ("Selumetinib", "BRAF", "mut"),
                                         "SelKras": ("Selumetinib", "KRAS", "mut"), "NavBcl": ("Navitoclax", "BCL2", "expr")}.items():
            r = pair(drug, gene, block)
            mac(f"bioPc{key}Rank", f2(r.rank_ig, 3) if r is not None else "--"); mac(f"bioPc{key}Top", pct(r.top20_ig) if r is not None else "--")
            mac(f"bioPc{key}Sign", pct(r.sign_ok_ig) if r is not None else "--"); mac(f"bioPc{key}True", f"{r.assoc_true:+.2f}" if r is not None else "--")
            mac(f"bioPc{key}Pred", f"{r.assoc_pred:+.2f}" if r is not None else "--"); mac(f"bioPc{key}RankMask", f2(r.rank_mask) if r is not None else "--")
        ex_ = agg[agg.block != "mut"]
        mac("bioPcExprRankMin", f2(ex_.rank_ig.min()) if len(ex_) else "--"); mac("bioPcExprRankMax", f2(ex_.rank_ig.max()) if len(ex_) else "--")
        mac("bioPcExprKdRank", f2(ex_.rank_kd.median()) if len(ex_) else "--"); mac("bioPcExprTopIg", pct(ex_.top20_ig.mean()) if len(ex_) else "--")
        mac("nBiomarkerPairsExpr", num(len(ex_))); mac("nBiomarkerPairsMut", num(len(agg) - len(ex_)))
        L = ["\\begin{table}[t]\\centering\\small\\caption{\\textbf{Positive-control biomarkers} (BRICS backbone, random regime, mean over five seeds). For each drug--biomarker pair with a known direction, the association of the measured and of the predicted $\\ln$ IC$_{50}$ with the biomarker across the drug's probe cells (Spearman $\\rho$ for expression and copy number; mean difference mutant $-$ wild type for mutations; a sign in the expected direction is marked $\\checkmark$), and the median rank fraction (0 $=$ first of \\nGenes{} genes) and top-\\topkGene{} rate of the biomarker gene in the gene-level explanations and in the knockdown counterfactual.}\\label{tab:poscontrol}",
             "\\begin{adjustbox}{max width=\\textwidth}\\begin{tabular}{lllcccccccc}\\toprule",
             "Drug & Biomarker & expected & $n_+$ & measured & predicted & rank IG & top-20 IG & rank mask & rank $\\Delta_{\\mathrm{kd}}$ & top-20 $\\Delta_{\\mathrm{kd}}$\\\\\\midrule"]
        for _, r in agg.iterrows():
            chk = lambda ok: " $\\checkmark$" if ok > 0.5 else ""
            L.append(f"{r.drug} & {r.gene} ({r.block}) & {r.direction} & {int(round(r.n_pos))} & {r.assoc_true:+.2f}{chk(r.true_sign_ok)} & {r.assoc_pred:+.2f}{chk(r.pred_sign_ok)} & "
                     f"{r.rank_ig:.2f} & {100 * r.top20_ig:.0f}\\% & {r.rank_mask:.2f} & {r.rank_kd:.2f} & {100 * r.top20_kd:.0f}\\%\\\\")
        L += ["\\bottomrule\\end{tabular}\\end{adjustbox}\\end{table}"]
        open(os.path.join(PAPER, "tab_poscontrol.tex"), "w", encoding="utf-8").write("\n".join(L))
    ph = read("bio_pharmacophore.csv"); pm = read("pharmacophore_match.csv")
    if len(ph):
        pagg = ph.groupby(["vocab", "confidence", "explainer"]).agg(hit=("top1_in_pharmacophore", "mean"), hit_sd=("top1_in_pharmacophore", "std"),
                                                                    chance=("chance", "mean"), n=("n_pairs", "mean"), cover=("mean_cover_of_top1", "mean")).reset_index()
        pagg.to_csv(os.path.join(SUMMARY, "bio_pharmacophore_agg.csv"), index=False)
        if len(pm):
            pb = pm[pm.vocab == "brics"].drop_duplicates("drug")   # a drug name can occur under two GDSC drug ids
            mac("nPharmHigh", num((pb.confidence == "high").sum())); mac("nPharmMedium", num((pb.confidence == "medium").sum())); mac("nPharmLow", num((pb.confidence == "low").sum()))
            mac("nPharmMatched", num(pb.matched.sum())); mac("nPharmDrugs", num(len(pb)))
        for vocab in VOCABS:
            V = vocab.capitalize()
            for conf, C in (("high", "High"), ("all", "All")):
                for ex in MOTIF_EX + ["gnnexp2", "size", "random", "delta_del", "delta_swap"]:
                    g = pagg[(pagg.vocab == vocab) & (pagg.confidence == conf) & (pagg.explainer == ex)]
                    E = ex.replace("delta_", "D").capitalize() if ex.startswith("delta") else ex.capitalize()
                    mac(f"pharm{V}{C}{E}", pct(float(g.hit.iloc[0])) if len(g) else "--")
                g = pagg[(pagg.vocab == vocab) & (pagg.confidence == conf) & (pagg.explainer == "attn")]
                mac(f"pharm{V}{C}Chance", pct(float(g.chance.iloc[0])) if len(g) else "--"); mac(f"pharm{V}{C}N", num(g.n.iloc[0]) if len(g) else "--")
        L = ["\\begin{table}[t]\\centering\\small\\caption{\\textbf{Curated-pharmacophore recovery.} Fraction of probe pairs (pooled over regimes and seeds) whose top-ranked motif contains atoms of the author-curated pharmacophore of the drug, for the high-confidence pharmacophores and for all 30; chance $=$ mean fraction of a drug's motifs that overlap the pharmacophore. Pairs whose drug has no or only pharmacophore motifs are excluded.}\\label{tab:pharm}",
             "\\begin{adjustbox}{max width=\\textwidth}\\begin{tabular}{lcccc}\\toprule", " & \\multicolumn{2}{c}{BRICS} & \\multicolumn{2}{c}{RECAP}\\\\",
             "Explainer & high-confidence & all & high-confidence & all\\\\\\midrule"]
        for ex in MOTIF_EX + ["gnnexp2", "size", "random", "delta_del", "delta_swap"]:
            cells = []
            for vocab in VOCABS:
                for conf in ("high", "all"):
                    g = pagg[(pagg.vocab == vocab) & (pagg.confidence == conf) & (pagg.explainer == ex)]
                    cells.append(pct(float(g.hit.iloc[0])) if len(g) else "--")
            L.append(f"{LAB[ex]} & " + " & ".join(cells) + "\\\\")
        cells = []
        for vocab in VOCABS:
            for conf in ("high", "all"):
                g = pagg[(pagg.vocab == vocab) & (pagg.confidence == conf) & (pagg.explainer == "attn")]
                cells.append(pct(float(g.chance.iloc[0])) if len(g) else "--")
        L.append("chance & " + " & ".join(cells) + "\\\\")
        L += ["\\bottomrule\\end{tabular}\\end{adjustbox}\\end{table}"]
        open(os.path.join(PAPER, "tab_pharm.tex"), "w", encoding="utf-8").write("\n".join(L))
    sa = read("gnnexp_sanity.csv")
    if len(sa):
        sagg = sa.groupby("explainer").agg(rec=("top1_recovery", "mean"), rec_min=("top1_recovery", "min"), chance=("chance", "mean"), n=("n_pairs", "mean")).reset_index()
        for ex in ("attn", "rollout", "ig", "mask", "gnnexp", "gnnexp2", "random"):
            g = sagg[sagg.explainer == ex]
            mac(f"sanity{ex.capitalize()}", pct(float(g.rec.iloc[0])) if len(g) else "--"); mac(f"sanity{ex.capitalize()}Min", pct(float(g.rec_min.iloc[0])) if len(g) else "--")
        mac("sanityChance", pct(sagg.chance.mean())); mac("sanityNPlants", num(sa.planted_vid.nunique())); mac("sanityRTest", f2(sa.r_test_synthetic.mean()))
        mac("sanityDelta", f2(abs(sa.mean_delta_del_planted.mean()))); mac("sanityBeta", f"{3.0:g}"); mac("sanityNPairs", num(sa.n_pairs.mean()))
        L = ["\\begin{table}[t]\\centering\\small\\caption{\\textbf{Planted-importance sanity check.} The default backbone was trained on synthetic labels $y=\\sanityBeta\\,[\\text{drug contains motif }K]+\\text{cell effect}+\\text{noise}$ for \\sanityNPlants{} planted motifs $K$ (test correlation \\sanityRTest; mean deletion effect of the planted motif $-$\\sanityDelta); recovery is the fraction of probe pairs whose top-ranked motif is the planted one (mean and minimum over the planted motifs; chance \\sanityChance).}\\label{tab:sanity}",
             "\\begin{tabular}{lcc}\\toprule", "Explainer & recovery & minimum over plants\\\\\\midrule"]
        for ex in ("attn", "rollout", "ig", "mask", "gnnexp", "gnnexp2", "random"):
            g = sagg[sagg.explainer == ex]
            if len(g):
                L.append(f"{LAB[ex]} & {pct(float(g.rec.iloc[0]))} & {pct(float(g.rec_min.iloc[0]))}\\\\")
        L += ["\\bottomrule\\end{tabular}\\end{table}"]
        open(os.path.join(PAPER, "tab_sanity.tex"), "w", encoding="utf-8").write("\n".join(L))


def headlines():
    """templated headline sentences from the macros already computed (values re-read from MAC)."""
    v = {m.split("}{")[0].split("\\")[-1]: m.split("}{")[1][:-1] for m in MAC}
    acc = (f"the CCMG-class model improves the within-drug correlation over the naive mean-effect predictor by {v.get('gapNaiveRandom', '--')} under the random regime "
           f"but not on unseen drugs ({v.get('gapNaiveUnseenDrug', '--')}), and differs from the tuned tree ensemble by "
           f"{v.get('gapLgbmRandom', '--')} (random), {v.get('gapLgbmUnseenDrug', '--')} (unseen drug) and {v.get('gapLgbmUnseenCell', '--')} (unseen cell)")
    stab = (f"under the random regime, two training seeds whose predictions of the probe pairs agree at $\\rho={v.get('stabSeedRandomMPredictionRho', '--')}$ agree on the motif ranking only at "
            f"$\\rho={v.get('stabSeedRandomMAttnRho', '--')}$ for the model's own attention (top-1 agreement {v.get('stabSeedRandomMAttnTop', '--')}, chance {v.get('faithBricsChanceTop', '--')}) and at most "
            f"{v.get('stabSeedRandomMMaskRho', '--')} (masking), and the counterfactual deletion effects themselves agree only at {v.get('stabSeedRandomMDdelRho', '--')}; "
            f"gene rankings agree at $\\rho$ {v.get('stabSeedGeneRhoMin', '--')}--{v.get('stabSeedGeneRhoMax', '--')} with top-20 Jaccard {v.get('stabSeedGeneJacMin', '--')}--{v.get('stabSeedGeneJacMax', '--')}; "
            f"all explainers are robust to omics input noise ($\\rho\\ge{v.get('stabPerturbMGnnexpRho', '--')}$) and the attention assigns the same top motif to {v.get('cdBricsAttnModal', '--')} of a drug's cells")
    faith = (f"against motif deletion, attention reaches $\\rho={v.get('faithBricsDelAttnRho', '--')}$ and top-1 agreement {v.get('faithBricsDelAttnTop', '--')} "
             f"(chance {v.get('faithBricsChanceTop', '--')}), rollout {v.get('faithBricsDelRolloutRho', '--')}, integrated gradients {v.get('faithBricsDelIgRho', '--')} and a GNNExplainer-style "
             f"mask {v.get('faithBricsDelGnnexpRho', '--')} under BRICS, while model-internal masking reaches {v.get('faithBricsDelMaskRho', '--')} by construction")
    bio = (f"annotated drug targets appear among the top-20 attributed genes in {v.get('bioBricsGIgHit', '--')} of probe pairs for integrated gradients against a chance of "
           f"{v.get('bioBricsGIgChance', '--')}, target attributions do not track CRISPR dependency ($\\rho_{{\\mathrm{{dep}}}}={v.get('bioBricsGIgDep', '--')}$), and the top-ranked motif is "
           f"pathway-enriched in {v.get('bioBricsMAttnHit', '--')} of pairs for attention (chance {v.get('bioBricsMAttnChance', '--')})")
    mac("accHeadline", acc); mac("stabHeadline", stab); mac("faithHeadline", faith); mac("bioHeadline", bio)


def main():
    meta = jload(os.path.join(DATA, "meta.json")); splits_meta = jload(os.path.join(DATA, "splits_meta.json"))
    overview = jload(os.path.join(SUMMARY, "overview.json"))
    runs = read("runs.csv")
    dataset_numbers(meta, splits_meta, overview, runs)
    accuracy(read("accuracy.csv"), read("accuracy_resampled.csv"))
    stability(read("stability.csv"))
    faithfulness(read("faithfulness.csv"), read("counterfactual_stats.csv"))
    biology(read("bio_gene.csv"), read("bio_dependency.csv"), read("bio_motif.csv"), read("bio_coherence.csv"), overview)
    example(meta)
    revision_tables(runs)
    control_tables()
    headlines()
    # every result macro the text refers to must exist, even before its runs are finished: define the missing ones
    # as "--" (build.py flags any such macro that is actually referenced)
    import re as _re
    defined = set(_re.findall(r"\\newcommand\{\\(\w+)\}", "\n".join(MAC)))
    body = ""
    for f in os.listdir(PAPER):
        if f.endswith(".tex") and f not in ("numbers.tex", "scirep_style.tex", "main.tex"):
            body += open(os.path.join(PAPER, f), encoding="utf-8").read()
    prefixes = ("var", "dsa", "tuning", "tuned", "acc", "stab", "faith", "bio", "cd", "coh", "vocabAtom", "attn", "fracTop", "gap", "entropy", "nTuning",
                "pharm", "sanity", "nPharm", "nBiomarker")
    for name in sorted(set(_re.findall(r"\\([A-Za-z]+)", body))):
        # result macros are camelCase (a lowercase prefix followed by an uppercase letter); plain LaTeX commands such as
        # \varepsilon never are, so they are left alone
        if name not in defined and name.startswith(prefixes) and _re.match(r"^[a-z]+[A-Z]", name) and name not in ("accHeadline",):
            MAC.append(f"\\newcommand{{\\{name}}}{{--}}")
    open(os.path.join(PAPER, "numbers.tex"), "w", encoding="utf-8").write("\n".join(MAC) + "\n")
    for f in glob.glob(os.path.join(FIGS, "fig_*.pdf")):
        shutil.copy(f, PAPER)
    print(f"wrote {len(MAC)} macros, tables and {len(glob.glob(os.path.join(FIGS, 'fig_*.pdf')))} figures")


if __name__ == "__main__":
    main()
