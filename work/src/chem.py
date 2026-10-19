"""Drug featurisation, motif decomposition (BRICS and RECAP-rule bond cleavage), motif vocabularies and
input-space counterfactual edits (delete / swap one motif).

A motif here is always a chemical substructure of the drug: a connected set of heavy atoms obtained by cutting
the molecule at BRICS bonds (Degen et al. 2008) or at RECAP cleavage bonds (Lewell et al. 1998).
"""
import os, pickle, sys
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, AllChem, rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

SYMS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se", "Pt", "other"]
HYB = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3,
       Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2]
ATOM_DIM = len(SYMS) + 6 + 4 + (len(HYB) + 1) + 1 + 5 + 1 + 3   # 40
BOND_DIM = 6

# RECAP cleavage rules (Lewell et al. 1998) as two-atom bond SMARTS; the bond between the matched atoms is cut.
RECAP_RULES = {
    "amide": "[C;$(C=O)]-!@[N;+0;!$(N=*)]",
    "ester": "[C;$(C=O)]-!@[O;+0;$(O-[#6])]",
    "amine": "[N;+0;!$(N=*);!$(N-C=O);!$(N-S(=O)=O);D2,D3]-!@[C;!$(C=O);!$(C=N)]",
    "urea": "[N;+0]-!@[C;$(C(=O)(N)N)]",
    "ether": "[O;+0;D2;!$(O-C=O);!$(O-S)]-!@[C;!$(C=O)]",
    "olefin": "[C;D2,D3]=!@[C;D2,D3]",
    "quaternary_N": "[N;+1;D4]-!@[C]",
    "aromN_aliphC": "[n]-!@[C;!$(C=O)]",
    "lactamN_aliphC": "[N;R;$(N-C=O)]-!@[C;!R;!$(C=O)]",
    "aromC_aromC": "[c]-!@[c]",
    "sulphonamide": "[S;$(S(=O)=O)]-!@[N;+0]",
}
_RECAP_PATTS = {k: Chem.MolFromSmarts(v) for k, v in RECAP_RULES.items()}


def onehot(v, choices, other=True):
    out = [0.0] * (len(choices) + (1 if other else 0))
    if v in choices: out[choices.index(v)] = 1.0
    elif other: out[-1] = 1.0
    return out


def atom_features(a):
    f = onehot(a.GetSymbol(), SYMS[:-1])                       # 14
    f += onehot(min(a.GetDegree(), 5), [0, 1, 2, 3, 4, 5], other=False)   # 6
    f += onehot(a.GetFormalCharge(), [-1, 0, 1])                # 4
    f += onehot(a.GetHybridization(), HYB)                      # 6
    f += [1.0 if a.GetIsAromatic() else 0.0]                    # 1
    f += onehot(min(a.GetTotalNumHs(), 4), [0, 1, 2, 3, 4], other=False)  # 5
    f += [1.0 if a.IsInRing() else 0.0]                         # 1
    f += onehot(str(a.GetChiralTag()), ["CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW"])  # 3
    return f


def bond_features(b):
    bt = b.GetBondType()
    return [1.0 if bt == Chem.BondType.SINGLE else 0.0, 1.0 if bt == Chem.BondType.DOUBLE else 0.0,
            1.0 if bt == Chem.BondType.TRIPLE else 0.0, 1.0 if bt == Chem.BondType.AROMATIC else 0.0,
            1.0 if b.GetIsConjugated() else 0.0, 1.0 if b.IsInRing() else 0.0]


def featurize(mol):
    x = np.array([atom_features(a) for a in mol.GetAtoms()], np.float32).reshape(-1, ATOM_DIM)
    src, dst, ea = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx(); f = bond_features(b)
        src += [i, j]; dst += [j, i]; ea += [f, f]
    ei = np.array([src, dst], np.int64).reshape(2, -1)
    ea = np.array(ea, np.float32).reshape(-1, BOND_DIM)
    return x, ei, ea


def cut_bonds(mol, vocab):
    """Set of (i, j) atom-index pairs whose bond is cut under the given motif vocabulary."""
    cuts = set()
    if vocab == "brics":
        for (i, j), _labels in BRICS.FindBRICSBonds(mol):
            cuts.add((min(i, j), max(i, j)))
    elif vocab == "recap":
        for name, patt in _RECAP_PATTS.items():
            for match in mol.GetSubstructMatches(patt):
                i, j = match[0], match[1]
                b = mol.GetBondBetweenAtoms(i, j)
                if b is not None and not b.IsInRing():
                    cuts.add((min(i, j), max(i, j)))
    else:
        raise ValueError(vocab)
    return cuts


def partition(mol, cuts, min_size=2):
    """Connected components after cutting; singletons are merged into the smallest neighbouring component.
    Returns atom2motif (n_atoms,), list of motif atom lists, motif-motif edges (set of pairs), attachments
    [(atom_in, atom_out)] per cut bond."""
    n = mol.GetNumAtoms()
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb

    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if (min(i, j), max(i, j)) not in cuts: union(i, j)
    # merge tiny components across cut bonds
    changed = True
    while changed:
        changed = False
        comp = {}
        for a in range(n): comp.setdefault(find(a), []).append(a)
        small = [r for r, atoms in comp.items() if len(atoms) < min_size]
        for r in small:
            atoms = comp[r]
            nbrs = []
            for (i, j) in cuts:
                if find(i) == r and find(j) != r: nbrs.append(find(j))
                elif find(j) == r and find(i) != r: nbrs.append(find(i))
            if nbrs:
                tgt = min(nbrs, key=lambda q: len(comp[q]))
                union(r, tgt); changed = True
                break
    roots = sorted(set(find(a) for a in range(n)), key=lambda r: min(a for a in range(n) if find(a) == r))
    rid = {r: k for k, r in enumerate(roots)}
    a2m = np.array([rid[find(a)] for a in range(n)], np.int64)
    motifs = [[a for a in range(n) if a2m[a] == k] for k in range(len(roots))]
    medges = set()
    attach = []
    for (i, j) in cuts:
        if a2m[i] != a2m[j]:
            medges.add((min(a2m[i], a2m[j]), max(a2m[i], a2m[j])))
            attach.append((i, j))
    return a2m, motifs, medges, attach


def motif_key(mol, atoms):
    return Chem.MolFragmentToSmiles(mol, atomsToUse=list(atoms), canonical=True)


def build_graph(mol, vocab):
    x, ei, ea = featurize(mol)
    cuts = cut_bonds(mol, vocab)
    a2m, motifs, medges, attach = partition(mol, cuts)
    keys = [motif_key(mol, m) for m in motifs]
    return {"x": x, "edge_index": ei, "edge_attr": ea, "a2m": a2m, "motif_atoms": motifs, "motif_keys": keys,
            "motif_edges": sorted(medges), "attach": attach, "smiles": Chem.MolToSmiles(mol), "n_atoms": len(x),
            "n_motifs": len(motifs)}


def morgan_fp(mol, nbits=1024, radius=2):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    return np.array(gen.GetFingerprintAsNumPy(mol), np.float32)


def build_all(vocab):
    meta = jload(os.path.join(DATA, "meta.json"))
    graphs, vocab_count = [], {}
    for d in meta["drugs"]:
        mol = Chem.MolFromSmiles(d["smiles"])
        g = build_graph(mol, vocab)
        graphs.append(g)
        for k in g["motif_keys"]: vocab_count[k] = vocab_count.get(k, 0) + 1
    # vocabulary: id 0 = UNK (motifs occurring once); frequent motifs get ids by frequency
    keys = sorted([k for k, c in vocab_count.items() if c >= 2], key=lambda k: (-vocab_count[k], k))
    vid = {k: i + 1 for i, k in enumerate(keys)}
    for g in graphs:
        g["motif_vid"] = np.array([vid.get(k, 0) for k in g["motif_keys"]], np.int64)
    out = {"vocab": vocab, "graphs": graphs, "vocab_keys": ["<UNK>"] + keys, "vocab_count": vocab_count,
           "n_vocab": len(keys) + 1}
    with open(os.path.join(DATA, f"graphs_{vocab}.pkl"), "wb") as f:
        pickle.dump(out, f)
    nm = np.array([g["n_motifs"] for g in graphs])
    print(f"[{vocab}] drugs {len(graphs)}  motifs/drug mean {nm.mean():.2f} median {np.median(nm):.0f} "
          f"min {nm.min()} max {nm.max()}  drugs with >=2 motifs {(nm >= 2).sum()}  distinct motif keys {len(vocab_count)}"
          f"  vocab (freq>=2) {len(keys)}")
    return out


# ---------------------------------------------------------------- counterfactual edits (input space)

def _subgraph(g, keep):
    """Restrict a graph dict to the atoms in `keep` (sorted list), remapping indices."""
    keep = sorted(keep); idx = {a: i for i, a in enumerate(keep)}
    mask = np.isin(g["edge_index"][0], keep) & np.isin(g["edge_index"][1], keep)
    ei = g["edge_index"][:, mask]
    ei = np.vectorize(idx.get)(ei) if ei.size else ei.reshape(2, 0)
    return {"x": g["x"][keep], "edge_index": ei.astype(np.int64), "edge_attr": g["edge_attr"][mask],
            "a2m": g["a2m"][keep]}


def _refeaturize(g, keep):
    """Try to rebuild atom features chemically (RDKit) for the molecule restricted to `keep`; None on failure."""
    try:
        mol = Chem.MolFromSmiles(g["smiles"])
        rw = Chem.RWMol(mol)
        for a in sorted(set(range(mol.GetNumAtoms())) - set(keep), reverse=True):
            rw.RemoveAtom(a)
        m2 = rw.GetMol()
        for a in m2.GetAtoms(): a.SetNoImplicit(False); a.SetNumExplicitHs(0)
        Chem.SanitizeMol(m2)
        x, ei, ea = featurize(m2)
        if len(x) != len(keep): return None
        return x
    except Exception:
        return None


def delete_motif(g, m):
    """Remove motif m (all its atoms and the motif node). Returns a graph dict with atoms re-featurised when
    RDKit can sanitise the remainder, otherwise with the original atom features (graph-level deletion)."""
    keep = [a for a in range(g["n_atoms"]) if g["a2m"][a] != m]
    if not keep: return None
    sg = _subgraph(g, keep)
    x2 = _refeaturize(g, keep)
    if x2 is not None: sg["x"] = x2
    sg["chem_valid"] = x2 is not None
    # motif bookkeeping: drop motif m and renumber
    old = sorted(set(range(g["n_motifs"])) - {m}); rm = {o: i for i, o in enumerate(old)}
    sg["a2m"] = np.array([rm[v] for v in sg["a2m"]], np.int64)
    sg["motif_vid"] = g["motif_vid"][old]
    sg["motif_edges"] = [(rm[a], rm[b]) for a, b in g["motif_edges"] if a != m and b != m]
    sg["n_atoms"] = len(keep); sg["n_motifs"] = len(old); sg["kept_motifs"] = old
    return sg


def swap_motif(g, m, donor, dm):
    """Replace motif m of g by motif dm of donor graph (graph-level edit: donor atoms and internal bonds are
    inserted and reconnected at the former cut bonds). The inserted motif keeps the donor's vocabulary id."""
    keep = [a for a in range(g["n_atoms"]) if g["a2m"][a] != m]
    if not keep: return None
    base = _subgraph(g, keep); idx = {a: i for i, a in enumerate(keep)}
    datoms = [a for a in range(donor["n_atoms"]) if donor["a2m"][a] == dm]
    didx = {a: i + len(keep) for i, a in enumerate(datoms)}
    dsub = _subgraph(donor, datoms)
    # donor attachment atoms (atoms of dm that had cut bonds), fallback: all donor atoms
    datt = [i for (i, j) in donor["attach"] if donor["a2m"][i] == dm] + [j for (i, j) in donor["attach"] if donor["a2m"][j] == dm]
    datt = sorted(set(datt)) or datoms
    x = np.concatenate([base["x"], dsub["x"]], 0)
    ei = [base["edge_index"], dsub["edge_index"] + len(keep)]
    ea = [base["edge_attr"], dsub["edge_attr"]]
    # reconnect: every cut bond of g that touched motif m
    single = np.array([[1, 0, 0, 0, 0, 0]], np.float32)
    k = 0
    for (i, j) in g["attach"]:
        if g["a2m"][i] == m and g["a2m"][j] != m: out_atom = j
        elif g["a2m"][j] == m and g["a2m"][i] != m: out_atom = i
        else: continue
        a_new = didx[datt[k % len(datt)]]; k += 1
        o = idx[out_atom]
        ei.append(np.array([[o, a_new], [a_new, o]], np.int64)); ea.append(np.repeat(single, 2, 0))
    old = sorted(set(range(g["n_motifs"])) - {m}); rm = {o: i for i, o in enumerate(old)}
    a2m = np.array([rm[v] for v in base["a2m"]] + [len(old)] * len(datoms), np.int64)
    medges = [(rm[a], rm[b]) for a, b in g["motif_edges"] if a != m and b != m]
    medges += [(rm[a if b == m else b], len(old)) for a, b in g["motif_edges"] if m in (a, b) and (a if b == m else b) != m]
    return {"x": x, "edge_index": np.concatenate(ei, 1), "edge_attr": np.concatenate(ea, 0), "a2m": a2m,
            "motif_vid": np.concatenate([g["motif_vid"][old], donor["motif_vid"][[dm]]]),
            "motif_edges": sorted(set(medges)), "n_atoms": len(x), "n_motifs": len(old) + 1,
            "kept_motifs": old, "swapped_in": int(len(old))}


if __name__ == "__main__":
    for v in VOCABS:
        build_all(v)
