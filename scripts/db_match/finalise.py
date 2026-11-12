#!/usr/bin/env python3
"""Build the final SurfPro table from match.py's output (run in the RDKit venv).

    ~/data/chemdb/venv/bin/python finalise.py
    ~/data/chemdb/venv/bin/python finalise.py --dbs chembl --out /tmp/x.csv

The GPU candidates (top 512 by bit Tanimoto) are re-ranked by the Tanimoto of
unfolded Morgan *count* fingerprints (r=2; sum of min / sum of max counts),
which, unlike bit fingerprints, separates chain-length homologues. Both
similarities are reported. Columns, per database <db> (chembl, pubchem):

  in_<db>, <db>_match     main molecule found (exact | stereo_differs |
                          same_inchikey | none); same_inchikey = a candidate
                          with a different canonical SMILES but the same
                          standard InChIKey (resonance form of a charged ring,
                          tautomer);
                          counterions removed and neutralised on both sides,
                          so other salts / the free acid or base count
  <db>_ids                up to 10 matching database ids
  PubChem compounds deposited only as part of SurfPro itself are excluded
  in match.py (see EXCLUDE there).
  <db>_nn_*               nearest neighbour (the molecule itself if found)
  <db>_nn_other_*         nearest *different* molecule (stereo-free SMILES differs)
                          *_id, *_smiles, *_tanimoto (count, used for ranking),
                          *_tanimoto_bits (2048-bit Morgan)
and best_nn_* / best_nn_other_* over all databases, in_any.
"""
import argparse, os

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/data/chemdb")
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
_cache, _keys = {}, {}


def inchikey(smi):
    if smi not in _keys:
        m = Chem.MolFromSmiles(smi)
        _keys[smi] = Chem.MolToInchiKey(m) if m is not None else None
    return _keys[smi]


def count_fp(smi):
    if smi not in _cache:
        m = Chem.MolFromSmiles(smi)
        _cache[smi] = _gen.GetSparseCountFingerprint(m) if m is not None else None
    return _cache[smi]


def pick(cands, qfp, q_flat):
    """Nearest and nearest-different candidate by (count, bit) Tanimoto."""
    c = cands.drop_duplicates("smiles").copy()     # one parent, many salts/ids
    c["tanimoto"] = [DataStructs.TanimotoSimilarity(qfp, f) if f is not None else -1.0
                     for f in map(count_fp, c["smiles"])]
    c = c.sort_values(["tanimoto", "tanimoto_bits"], ascending=False)
    nn = c.iloc[0] if len(c) else None
    other = c[c["smiles_nostereo"] != q_flat]
    return nn, (other.iloc[0] if len(other) else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dbs", nargs="+", default=["chembl", "pubchem"])
    ap.add_argument("--workdir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--surfpro", default=os.path.join(HERE, "../../data/SurfPro.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "../../data/SurfPro_db_matches.csv"))
    args = ap.parse_args()

    raw = pd.read_csv(args.surfpro)
    q = pd.read_parquet(os.path.join(args.workdir, "queries.parquet")).set_index("row")
    out = raw[["SMILES", "type"]].copy()
    out.insert(0, "row", raw.index)
    out["canonical_smiles"] = q["smiles"].reindex(raw.index)
    out["canonical_smiles_nostereo"] = q["smiles_nostereo"].reindex(raw.index)

    for db in args.dbs:
        ex = pd.read_csv(os.path.join(args.workdir, f"exact_{db}.csv"),
                         keep_default_na=False).set_index("row").reindex(raw.index)
        out[f"in_{db}"] = ex["match"].isin(["exact", "stereo_differs"])
        out[f"{db}_match"] = ex["match"].fillna("not_standardised")
        out[f"{db}_ids"] = ex["ids"].fillna("")
        cand = pd.read_parquet(os.path.join(args.workdir, f"candidates_{db}.parquet"))
        res = {}
        for row, c in cand.groupby("row", sort=False):
            res[row] = pick(c, count_fp(q.at[row, "smiles"]), q.at[row, "smiles_nostereo"])
            # same compound drawn differently (charge on another ring N,
            # tautomer): canonical SMILES differ but the standard InChIKey,
            # which normalises these, is equal
            if out.at[row, f"{db}_match"] == "none":
                key = inchikey(q.at[row, "smiles"])
                same = [i for i, smi in zip(c["id"], c["smiles"]) if inchikey(smi) == key]
                if key and same:
                    out.at[row, f"{db}_match"] = "same_inchikey"
                    out.at[row, f"in_{db}"] = True
                    out.at[row, f"{db}_ids"] = ";".join(sorted(set(same))[:10])
        for k, tag in ((0, "nn"), (1, "nn_other")):
            for col in ("id", "smiles", "tanimoto", "tanimoto_bits"):
                out[f"{db}_{tag}_{col}"] = [
                    (res[r][k][col] if r in res and res[r][k] is not None else None)
                    for r in raw.index]
            for col in ("tanimoto", "tanimoto_bits"):
                out[f"{db}_{tag}_{col}"] = out[f"{db}_{tag}_{col}"].astype(float).round(4)
        print(f"{db}: {int(out[f'in_{db}'].sum())} found")

    out["in_any"] = out[[f"in_{db}" for db in args.dbs]].any(axis=1)
    for tag in ("nn", "nn_other"):
        sim = out[[f"{db}_{tag}_tanimoto" for db in args.dbs]].fillna(-1).to_numpy()
        which = np.array(args.dbs)[sim.argmax(1)]
        out[f"best_{tag}_db"] = which
        for col in ("id", "smiles", "tanimoto", "tanimoto_bits"):
            out[f"best_{tag}_{col}"] = [out.at[i, f"{d}_{tag}_{col}"] for i, d in enumerate(which)]

    out.to_csv(args.out, index=False)
    print(f"written {args.out}\n")
    print(pd.DataFrame({db: out[f"{db}_match"].value_counts() for db in args.dbs}).fillna(0).astype(int))
    print(f"in any database: {int(out['in_any'].sum())} / {len(out)}")
    print("\nnearest different molecule (all databases), count Tanimoto:")
    print(out["best_nn_other_tanimoto"].describe(percentiles=[.1, .25, .5, .75, .9]).round(3).to_string())


if __name__ == "__main__":
    main()
