"""Positive-control biology tests on the finished explanation runs:
(1) biomarker pairs with a known direction (pharmacophores.BIOMARKERS): does the measured response, the prediction and
    the explanation track the biomarker across the drug's probe cells, and does the explanation rank the biomarker gene?
(2) curated pharmacophores (pharmacophores.PHARMACOPHORE): is the top-ranked motif the one that carries the pharmacophore?
Usage: python bio_controls.py   -> results/summary/bio_positive_controls.csv, bio_pharmacophore.csv, pharmacophore_match.csv
"""
import glob, os, pickle, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from rdkit import Chem
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from pharmacophores import PHARMACOPHORE, BIOMARKERS
from analyze import motif_mat, load_scores

GENE_EX = ["ig", "sal", "mask", "gnnexp", "kd"]
MOTIF_EX = ["attn", "rollout", "ig", "mask", "gnnexp", "gnnexp2", "size", "random", "delta_del", "delta_swap"]


def main_runs(vocab="brics", split="random"):
    runs = []
    for f in glob.glob(os.path.join(EXPL, f"ccmg_{vocab}_{split}_ss0_s*", "summary.json")):
        s = jload(f)
        if s.get("tag", "") == "":
            runs.append(os.path.basename(os.path.dirname(f)))
    return sorted(runs)


def biomarker_tests(meta, d, S, info):
    genes = meta["genes"]; gi = {g: i for i, g in enumerate(genes)}
    names = [x["name"] for x in meta["drugs"]]
    G = len(genes)
    rows = []
    for r, s in S.items():
        dn, cn = s["d_idx"], s["c_idx"]
        for drug, gene, block, direction in BIOMARKERS:
            if drug not in names or gene not in gi:
                continue
            di, g = names.index(drug), gi[gene]
            sel = np.where(dn == di)[0]
            if len(sel) < 20:
                continue
            cells = cn[sel]
            feat = {"expr": d["expr"], "mut": d["mut"], "cnv": np.log2(d["cnv"] + 1.0)}[block][cells, g]
            y, yh = s["y_true"][sel], s["y0"][sel]
            if block == "mut":
                pos = feat > 0.5
                if pos.sum() < 3 or (~pos).sum() < 3:
                    continue
                assoc_true = float(y[pos].mean() - y[~pos].mean()); assoc_pred = float(yh[pos].mean() - yh[~pos].mean())
                p_true = float(mannwhitneyu(y[pos], y[~pos]).pvalue)
            else:
                pos = feat >= np.percentile(feat, 75)
                assoc_true = float(spearmanr(feat, y)[0]); assoc_pred = float(spearmanr(feat, yh)[0])
                p_true = float(spearmanr(feat, y)[1])
            # expected sign of the association of ln IC50 with the feature: sensitive -> negative, resistant -> positive
            exp_sign = -1 if direction == "sensitive" else 1
            rec = {"run": r, "vocab": info[r]["vocab"], "split": info[r]["split"], "seed": info[r]["seed"], "drug": drug, "gene": gene, "block": block,
                   "direction": direction, "n_cells": int(len(sel)), "n_pos": int(pos.sum()), "assoc_true": assoc_true, "p_true": p_true,
                   "assoc_pred": assoc_pred, "true_sign_ok": bool(np.sign(assoc_true) == exp_sign), "pred_sign_ok": bool(np.sign(assoc_pred) == exp_sign)}
            for ex in GENE_EX:
                A = s["delta_kd"].astype(np.float32) if ex == "kd" else s["gene_" + ex].astype(np.float32)
                A = A[sel]
                ranks = (np.argsort(np.argsort(-np.abs(A), 1), 1)[:, g]) / G      # 0 = top
                rec[f"rank_{ex}"] = float(np.median(ranks)); rec[f"rank_{ex}_pos"] = float(np.median(ranks[pos]))
                rec[f"top20_{ex}"] = float(np.mean(ranks * G < TOPK_GENE)); rec[f"top20_{ex}_pos"] = float(np.mean(ranks[pos] * G < TOPK_GENE))
                # magnitude of the biomarker attribution larger in biomarker-positive cells?
                a = np.abs(A[:, g])
                rec[f"mag_assoc_{ex}"] = float(spearmanr(feat, a)[0]) if np.std(a) > 0 and np.std(feat) > 0 else np.nan
                # direction: for signed explainers the attribution of the feature in positive cells should carry the expected sign
                if ex in ("ig", "sal", "mask"):
                    if block == "mut":
                        blk = G
                    else:
                        blk = 0 if block == "expr" else 2 * G
                    # gene-level scores are summed over the three omics entries; use the per-gene signed score
                    sg = A[pos, g]
                    rec[f"sign_ok_{ex}"] = float(np.mean(np.sign(sg) == exp_sign)) if len(sg) else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def pharmacophore_motifs(meta, graphs_by_vocab):
    """per vocabulary and probe drug: motif indices overlapping the curated SMARTS match; match statistics."""
    out, stats = {}, []
    for vocab, G in graphs_by_vocab.items():
        for di, dr in enumerate(meta["drugs"]):
            if dr["name"] not in PHARMACOPHORE:
                continue
            smarts, conf, why = PHARMACOPHORE[dr["name"]]
            mol = Chem.MolFromSmiles(dr["smiles"]); patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                stats.append({"vocab": vocab, "drug": dr["name"], "confidence": conf, "matched": False, "note": "bad SMARTS"}); continue
            matches = mol.GetSubstructMatches(patt)
            atoms = set(a for m in matches for a in m)
            g = G["graphs"][di]
            pm = sorted(set(int(g["a2m"][a]) for a in atoms))
            cover = {m: len(atoms & set(g["motif_atoms"][m])) / len(g["motif_atoms"][m]) for m in pm}
            out[(vocab, di)] = {"pharm_motifs": pm, "atoms": atoms, "cover": cover, "confidence": conf}
            stats.append({"vocab": vocab, "drug": dr["name"], "confidence": conf, "matched": bool(atoms), "n_atoms": len(atoms),
                          "n_motifs": int(g["n_motifs"]), "n_pharm_motifs": len(pm), "frac_pharm_motifs": len(pm) / g["n_motifs"],
                          "max_cover": max(cover.values()) if cover else 0.0, "smarts": smarts})
    return out, pd.DataFrame(stats)


def pharmacophore_tests(meta, graphs_by_vocab, S, info):
    pm, stats = pharmacophore_motifs(meta, graphs_by_vocab)
    rows = []
    for r, s in S.items():
        vocab = info[r]["vocab"]; dn = s["d_idx"]
        for conf_set, label in ((("high",), "high"), (("high", "medium"), "high+medium"), (("high", "medium", "low"), "all")):
            for ex in MOTIF_EX:
                A = s[ex] if ex.startswith("delta") else motif_mat(s, ex)
                hits, chance, cov = [], [], []
                for i, di in enumerate(dn):
                    key = (vocab, int(di))
                    if key not in pm or pm[key]["confidence"] not in conf_set:
                        continue
                    P = pm[key]["pharm_motifs"]; n = int(np.sum(~np.isnan(A[i])))
                    if not P or len(P) >= n:
                        continue
                    top = int(np.nanargmax(np.where(np.isnan(A[i]), -np.inf, np.abs(A[i]))))
                    hits.append(float(top in P)); chance.append(len(P) / n)
                    cov.append(pm[key]["cover"].get(top, 0.0))
                if hits:
                    rows.append({"run": r, "vocab": vocab, "split": info[r]["split"], "seed": info[r]["seed"], "confidence": label, "explainer": ex,
                                 "top1_in_pharmacophore": float(np.mean(hits)), "chance": float(np.mean(chance)), "n_pairs": len(hits),
                                 "mean_cover_of_top1": float(np.mean(cov))})
    return pd.DataFrame(rows), stats


def main():
    meta = jload(os.path.join(DATA, "meta.json")); d = np.load(os.path.join(DATA, "dataset.npz"))
    graphs_by_vocab = {}
    for v in VOCABS:
        with open(os.path.join(DATA, f"graphs_{v}.pkl"), "rb") as f:
            graphs_by_vocab[v] = pickle.load(f)
    runs = []
    for v in VOCABS:
        for sp in SPLITS:
            runs += main_runs(v, sp)
    S = {r: load_scores(r) for r in runs}; S = {r: s for r, s in S.items() if s is not None}
    info = {r: jload(os.path.join(EXPL, r, "summary.json")) for r in S}
    print(f"{len(S)} runs", flush=True)
    bio = biomarker_tests(meta, d, S, info)
    bio.to_csv(os.path.join(SUMMARY, "bio_positive_controls.csv"), index=False)
    ph, stats = pharmacophore_tests(meta, graphs_by_vocab, S, info)
    ph.to_csv(os.path.join(SUMMARY, "bio_pharmacophore.csv"), index=False)
    stats.to_csv(os.path.join(SUMMARY, "pharmacophore_match.csv"), index=False)
    print(stats[stats.vocab == "brics"][["drug", "confidence", "matched", "n_motifs", "n_pharm_motifs", "max_cover"]].to_string(index=False))
    b = bio[(bio.vocab == "brics") & (bio.split == "random")]
    if len(b):
        print(b.groupby(["drug", "gene", "block"])[["n_pos", "assoc_true", "assoc_pred", "rank_ig", "top20_ig", "rank_kd", "top20_kd", "sign_ok_ig"]].mean().round(3).to_string())
    p = ph[(ph.vocab == "brics") & (ph.split == "random")]
    if len(p):
        print(p.groupby(["confidence", "explainer"])[["top1_in_pharmacophore", "chance", "n_pairs"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
