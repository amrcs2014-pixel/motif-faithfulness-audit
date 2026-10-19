"""Probe set and data splits.

Probe set: PROBE_DRUGS drugs (>= 3 motifs under both vocabularies, >= 1 mapped target gene, well measured, spread
over GDSC target pathways) x PROBE_CELLS cells (CRISPR screened, well measured, spread over lineages). The measured
probe pairs (capped at MAX_PROBE_PAIRS) are never used for training or validation under any split, so the same
pairs can be explained under three exposure regimes: random (pair unseen, drug and cell seen), unseen-drug
(probe drugs fully held out) and unseen-cell (probe cells fully held out).

For split type t and split seed s the remaining pairs / drugs / cells are divided 80/10/10 into train/val/test.
Writes data/splits.npz (index arrays) and data/splits_meta.json.  Usage: python splits.py
"""
import os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *


def round_robin(groups, n, rng):
    """Pick n items spread over groups (dict group -> list), shuffling within groups."""
    keys = sorted(groups)
    rng.shuffle(keys)
    pools = {k: list(rng.permutation(groups[k])) for k in keys}
    out = []
    while len(out) < n and any(pools.values()):
        for k in keys:
            if pools[k] and len(out) < n:
                out.append(pools[k].pop())
    return out


def main():
    d = np.load(os.path.join(DATA, "dataset.npz"))
    meta = jload(os.path.join(DATA, "meta.json"))
    pd_, pc, y = d["pair_drug"], d["pair_cell"], d["y"]
    n_drugs, n_cells = len(meta["drugs"]), len(meta["cells"])
    nm = {}
    for v in VOCABS:
        with open(os.path.join(DATA, f"graphs_{v}.pkl"), "rb") as f:
            nm[v] = np.array([g["n_motifs"] for g in pickle.load(f)["graphs"]])
    rng = np.random.default_rng(0)
    drug_counts = np.bincount(pd_, minlength=n_drugs)
    cell_counts = np.bincount(pc, minlength=n_cells)
    has_crispr = ~np.isnan(d["crispr"]).all(axis=1)

    # probe drugs
    elig = [i for i in range(n_drugs) if nm["brics"][i] >= 3 and nm["recap"][i] >= 3 and meta["drugs"][i]["targets"]
            and drug_counts[i] >= np.percentile(drug_counts, 40)]
    groups = {}
    for i in elig:
        groups.setdefault(meta["drugs"][i]["pathway"], []).append(i)
    probe_drugs = sorted(round_robin(groups, PROBE_DRUGS, rng))
    # probe cells
    eligc = [i for i in range(n_cells) if has_crispr[i] and cell_counts[i] >= np.percentile(cell_counts, 40)]
    groupsc = {}
    for i in eligc:
        groupsc.setdefault(meta["cell_lineage"][i], []).append(i)
    probe_cells = sorted(round_robin(groupsc, PROBE_CELLS, rng))
    is_pd = np.isin(pd_, probe_drugs); is_pc = np.isin(pc, probe_cells)
    probe_pairs = np.where(is_pd & is_pc)[0]
    if len(probe_pairs) > MAX_PROBE_PAIRS:
        probe_pairs = np.sort(rng.choice(probe_pairs, MAX_PROBE_PAIRS, replace=False))
    is_probe = np.zeros(len(y), bool); is_probe[probe_pairs] = True
    print(f"eligible drugs {len(elig)} -> probe drugs {len(probe_drugs)}; eligible cells {len(eligc)} -> probe cells "
          f"{len(probe_cells)}; probe pairs {len(probe_pairs)}")

    out = {}
    for s in SPLIT_SEEDS:
        r = np.random.default_rng(1000 + s)
        # random: all non-probe pairs 80/10/10
        idx = r.permutation(np.where(~is_probe)[0])
        n = len(idx); nv, nt = int(VAL_FRAC * n), int(TEST_FRAC * n)
        out[f"random_ss{s}_val"] = np.sort(idx[:nv]); out[f"random_ss{s}_test"] = np.sort(np.concatenate([idx[nv:nv + nt], probe_pairs]))
        out[f"random_ss{s}_train"] = np.sort(idx[nv + nt:])
        # unseen drug: non-probe drugs 80/10/10; all pairs of probe drugs -> test
        others = r.permutation([i for i in range(n_drugs) if i not in probe_drugs])
        nv, nt = int(round(VAL_FRAC * n_drugs)), int(round(TEST_FRAC * n_drugs)) - len(probe_drugs)
        nt = max(nt, 0)
        vd, td = set(others[:nv]), set(others[nv:nv + nt]) | set(probe_drugs)
        trd = set(others[nv + nt:])
        out[f"unseen_drug_ss{s}_train"] = np.where(np.isin(pd_, list(trd)))[0]
        out[f"unseen_drug_ss{s}_val"] = np.where(np.isin(pd_, list(vd)))[0]
        out[f"unseen_drug_ss{s}_test"] = np.where(np.isin(pd_, list(td)))[0]
        # unseen cell
        othersc = r.permutation([i for i in range(n_cells) if i not in probe_cells])
        nv, nt = int(round(VAL_FRAC * n_cells)), max(int(round(TEST_FRAC * n_cells)) - len(probe_cells), 0)
        vc, tc = set(othersc[:nv]), set(othersc[nv:nv + nt]) | set(probe_cells)
        trc = set(othersc[nv + nt:])
        out[f"unseen_cell_ss{s}_train"] = np.where(np.isin(pc, list(trc)))[0]
        out[f"unseen_cell_ss{s}_val"] = np.where(np.isin(pc, list(vc)))[0]
        out[f"unseen_cell_ss{s}_test"] = np.where(np.isin(pc, list(tc)))[0]
        for t in SPLITS:
            tr, va, te = (out[f"{t}_ss{s}_{p}"] for p in ("train", "val", "test"))
            assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
            assert not is_probe[tr].any() and not is_probe[va].any() and is_probe[te].sum() == len(probe_pairs)
            if s == 0:
                print(f"{t}: train {len(tr)} val {len(va)} test {len(te)} (probe {is_probe[te].sum()})")
    np.savez_compressed(os.path.join(DATA, "splits.npz"), probe_pairs=probe_pairs, **out)
    jdump({"probe_drugs": probe_drugs, "probe_cells": probe_cells, "n_probe_pairs": int(len(probe_pairs)),
           "probe_drug_names": [meta["drugs"][i]["name"] for i in probe_drugs],
           "probe_drug_pathways": [meta["drugs"][i]["pathway"] for i in probe_drugs],
           "probe_drug_targets": [meta["drugs"][i]["targets"] for i in probe_drugs],
           "probe_drug_n_motifs": {v: [int(nm[v][i]) for i in probe_drugs] for v in VOCABS},
           "probe_cell_lineages": [meta["cell_lineage"][i] for i in probe_cells]},
          os.path.join(DATA, "splits_meta.json"))
    print("saved splits.npz / splits_meta.json")


if __name__ == "__main__":
    main()
