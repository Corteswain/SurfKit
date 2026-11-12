#!/usr/bin/env python3
"""Look up every SurfPro entry in the standardised reference databases.

    python match.py                        # chembl + pubchem -> ~/data/chemdb/results/
    python match.py --dbs chembl
    ~/data/chemdb/venv/bin/python finalise.py   # -> ../../data/SurfPro_db_matches.csv

Requires the shards from build_db.py for surfpro and every database. Runs in
the base conda env (needs torch with CUDA, numpy, pandas, pyarrow; no RDKit).

1. Exact lookup on the standardised parent SMILES (counterions removed,
   neutralised, RDKit-canonical), so another salt or the free acid/base of the
   same molecule counts as found:
     <db>_match = exact           identical canonical SMILES incl. stereo
                  stereo_differs  identical once stereo is ignored (SurfPro
                                  rarely specifies stereo, databases often do)
                  none
     in_<db>    = exact or stereo_differs
2. Candidate neighbours by Tanimoto similarity of Morgan bit fingerprints
   (r=2, 2048 bits) over the whole database, on the GPU: bits are unpacked to
   fp16 and |A & B| for all query x database pairs is one matrix product per
   block (exact: counts <= 2048 are representable in fp16, accumulation is
   fp32). The TOPK best per query are written out; finalise.py re-ranks them
   with count fingerprints, because bit fingerprints cannot tell chain-length
   homologues apart (C12 and C6 sulfonate share every bit: Tanimoto 1).

Writes <workdir>/exact_<db>.csv and candidates_<db>.parquet.
"""
import argparse, glob, os, time

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

ROOT = os.path.expanduser("~/data/chemdb")
HERE = os.path.dirname(os.path.abspath(__file__))
TOPK = 512       # candidates per query, re-ranked by count Tanimoto in finalise.py
BLOCK = 40_000   # database rows per GPU matrix product
# Database entries to ignore. SurfPro itself was deposited into PubChem
# (NORMAN Suspect List Exchange, xref doi 10.1039/D4DD00393D), wrong
# structures included; compounds whose only source is that deposition would
# make "found in PubChem" circular. Compounds also deposited by others stay.
EXCLUDE = {"pubchem": (os.path.join(ROOT, "pubchem_surfpro_deposit/surfpro_deposit_cids.json"),
                       "surfpro_only_cids")}


def excluded_ids(db):
    if db not in EXCLUDE or not os.path.exists(EXCLUDE[db][0]):
        return set()
    import json
    path, key = EXCLUDE[db]
    return {str(i) for i in json.load(open(path))[key]}


def shards(db):
    files = sorted(glob.glob(os.path.join(ROOT, db, "shard_*.npy")))
    files = [f for f in files if not f.endswith(".tmp.npy")]
    if not files:
        raise SystemExit(f"no shards for {db} in {ROOT}/{db}; run build_db.py first")
    return files


def unpack(packed_u8):
    """uint8 [n, 256] on the GPU -> fp16 bits [n, 2048] (np.packbits order)."""
    shifts = torch.arange(7, -1, -1, device=packed_u8.device, dtype=torch.uint8)
    bits = (packed_u8.unsqueeze(-1) >> shifts) & 1
    return bits.reshape(packed_u8.shape[0], -1).half()


def exact_lookup(db, q, exclude):
    """ids of database entries whose parent SMILES equals the query's, with and
    without stereo. Returns two dicts query_smiles -> [ids]."""
    want_iso = set(q["smiles"]); want_flat = set(q["smiles_nostereo"])
    iso, flat = {}, {}
    ex = pa_array(exclude) if exclude else None
    for f in shards(db):
        t = pq.read_table(f.replace(".npy", ".parquet"))
        if ex is not None:
            t = t.filter(pc.invert(pc.is_in(t["id"], value_set=ex)))
        for col, want, out in (("smiles", want_iso, iso), ("smiles_nostereo", want_flat, flat)):
            mask = pc.is_in(t[col], value_set=pa_array(want))
            if pc.any(mask).as_py():
                hit = t.filter(mask)
                for i, s in zip(hit["id"].to_pylist(), hit[col].to_pylist()):
                    out.setdefault(s, []).append(i)
    return iso, flat


def pa_array(values):
    import pyarrow as pa
    return pa.array(sorted(values))


def gpu_topk(db, q_packed, device, exclude, k=TOPK):
    """Top-k Tanimoto neighbours of every query over the whole database
    (minus the excluded ids). Returns similarities [nq, k] and global row
    indices of the neighbours."""
    ex = pa_array(exclude) if exclude else None
    Q = unpack(torch.from_numpy(q_packed).to(device))
    qc = Q.float().sum(1, keepdim=True)
    best_s = torch.full((len(Q), k), -1.0, device=device)
    best_i = torch.zeros((len(Q), k), dtype=torch.int64, device=device)
    files = shards(db)
    offsets, total, t0 = [], 0, time.time()
    for n, f in enumerate(files):
        D_np = np.load(f, mmap_mode="r")
        offsets.append(total)
        if len(D_np) == 0:
            continue
        P = torch.from_numpy(np.ascontiguousarray(D_np)).to(device)
        drop = None
        if ex is not None:
            ids = pq.read_table(f.replace(".npy", ".parquet"), columns=["id"])["id"]
            drop = np.flatnonzero(pc.is_in(ids, value_set=ex).to_numpy(zero_copy_only=False))
        for b in range(0, len(P), BLOCK):        # 8 GB card: [nq, BLOCK] at a time
            D = unpack(P[b:b + BLOCK])
            dc = D.float().sum(1)
            sim = (Q @ D.T).float()              # |A & B|, then Tanimoto in place
            union = qc + dc.unsqueeze(0) - sim
            sim.div_(union.clamp_(min=1))        # empty fingerprints -> 0
            if drop is not None and len(drop):
                d = drop[(drop >= b) & (drop < b + BLOCK)] - b
                if len(d):
                    sim[:, torch.from_numpy(d).to(device)] = -1.0
            s, i = sim.topk(min(k, sim.shape[1]), dim=1)
            best_s, j = torch.cat([best_s, s], 1).topk(k, dim=1)
            best_i = torch.cat([best_i, i + total + b], 1).gather(1, j)
            del D, sim, union
        total += len(D_np)
        del P
        if n % 50 == 0 or n == len(files) - 1:
            print(f"  {db}: shard {n + 1}/{len(files)}, {total:,} molecules, "
                  f"{time.time() - t0:.0f} s", flush=True)
    offsets = np.array(offsets + [total])
    return best_s.cpu().numpy(), best_i.cpu().numpy(), files, offsets


def fetch_rows(files, offsets, gidx):
    """id, smiles, smiles_nostereo for global row indices (any shape)."""
    flat = np.unique(gidx.ravel())
    shard_of = np.searchsorted(offsets, flat, side="right") - 1
    out = {}
    for s in np.unique(shard_of):
        rows = flat[shard_of == s] - offsets[s]
        t = pq.read_table(files[s].replace(".npy", ".parquet")).take(rows)
        for g, i, a, b in zip(flat[shard_of == s], t["id"].to_pylist(),
                              t["smiles"].to_pylist(), t["smiles_nostereo"].to_pylist()):
            out[g] = (i, a, b)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dbs", nargs="+", default=["chembl", "pubchem"])
    ap.add_argument("--workdir", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu'}")

    # standardised queries (build_db.py surfpro); ids are SurfPro.csv row numbers
    qf = shards("surfpro")
    q = pd.concat([pq.read_table(f.replace(".npy", ".parquet")).to_pandas() for f in qf])
    q_fp = np.concatenate([np.load(f) for f in qf])
    q["row"] = q["id"].astype(int)
    order = np.argsort(q["row"].to_numpy())
    q, q_fp = q.iloc[order].reset_index(drop=True), q_fp[order]

    os.makedirs(args.workdir, exist_ok=True)
    q[["row", "smiles", "smiles_nostereo"]].to_parquet(os.path.join(args.workdir, "queries.parquet"))
    for db in args.dbs:
        t0 = time.time()
        exclude = excluded_ids(db)
        if exclude:
            print(f"{db}: excluding {len(exclude):,} ids ({EXCLUDE[db][0]})")
        iso, flat = exact_lookup(db, q, exclude)
        m_iso = q["smiles"].map(lambda s: s in iso)
        m_flat = q["smiles_nostereo"].map(lambda s: s in flat)
        pd.DataFrame({
            "row": q["row"],
            "match": np.where(m_iso, "exact", np.where(m_flat, "stereo_differs", "none")),
            "ids": [";".join(sorted(set(iso.get(a, []) + flat.get(b, [])))[:10])
                    for a, b in zip(q["smiles"], q["smiles_nostereo"])],
        }).to_csv(os.path.join(args.workdir, f"exact_{db}.csv"), index=False)
        print(f"{db}: exact lookup {time.time() - t0:.0f} s, found {int((m_iso | m_flat).sum())}")

        sims, gidx, files, offsets = gpu_topk(db, q_fp, device, exclude)
        rows = fetch_rows(files, offsets, gidx)
        recs = [(q.at[qi, "row"], j, *rows[gidx[qi, j]], float(sims[qi, j]))
                for qi in range(len(q)) for j in range(gidx.shape[1]) if sims[qi, j] >= 0]
        pd.DataFrame(recs, columns=["row", "rank", "id", "smiles", "smiles_nostereo",
                                    "tanimoto_bits"]).to_parquet(
            os.path.join(args.workdir, f"candidates_{db}.parquet"))
        print(f"{db}: done in {(time.time() - t0) / 60:.1f} min")
    print(f"written to {args.workdir}; now run finalise.py in the RDKit venv")


if __name__ == "__main__":
    main()
