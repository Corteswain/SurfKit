#!/usr/bin/env python3
"""Standardise a whole SMILES collection into shards for exact and
nearest-neighbour matching (see standardise.py for what is done per molecule).

    python build_db.py chembl  ~/data/chemdb/raw/chembl_37_chemreps.txt.gz
    python build_db.py pubchem ~/data/chemdb/raw/CID-SMILES.gz
    python build_db.py surfpro ../../data/SurfPro.csv

Writes <outdir>/<name>/shard_00000.parquet (id, smiles, smiles_nostereo) and
shard_00000.npy (uint8 [n, 256] packed Morgan bits), CHUNK molecules per shard,
row-aligned. Molecules RDKit cannot parse and molecules outside the SurfPro
domain (larger than the largest SurfPro main molecule, or with elements no
SurfPro main molecule has; see standardise.py) are left out and counted in
stats.json. The domain is computed from --domain-from and saved as
domain.json and in each shard's parquet metadata. Existing shards are kept, so
an interrupted run resumes where it stopped; a shard built with a different
(or no) domain is re-filtered from its stored SMILES instead of recomputed.

Must run in the RDKit venv (~/data/chemdb/venv): the base conda RDKit is
built for numpy 1.x and crashes in GetFingerprintAsNumPy.
"""
import argparse, csv, gzip, itertools, json, os, resource, sys, threading, time
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from rdkit import Chem

from standardise import BAD_ELEMENTS, FP_BYTES, TOO_LARGE, out_of_domain, standardise, surfpro_domain

WORKER_MEM_GB = 6   # address-space cap per worker (idle RDKit worker: ~2.7 GB virtual)


DOMAIN = None       # set in each worker by init_worker


def init_worker(gb, domain):
    """Pool initializer. Memory cap: a runaway molecule raises MemoryError
    (caught in standardise) instead of pushing the machine into the OOM killer."""
    global DOMAIN
    resource.setrlimit(resource.RLIMIT_AS, (int(gb * 2**30), int(gb * 2**30)))
    DOMAIN = dict(domain, _allowed=set(domain["elements"]))


def domain_tag(domain):
    return json.dumps({k: domain[k] for k in ("max_heavy_atoms", "elements")}, sort_keys=True)


def write_shard(base, ids, iso, flat, fp, tag):
    t = pa.table({"id": ids, "smiles": iso, "smiles_nostereo": flat})
    pq.write_table(t.replace_schema_metadata({"domain": tag}), base + ".tmp.parquet")
    os.replace(base + ".tmp.parquet", base + ".parquet")
    np.save(base + ".tmp.npy", fp)
    os.replace(base + ".tmp.npy", base + ".npy")   # .npy marks the shard complete


def refilter(base, tag):
    """Apply the domain to a shard built without it. Returns (kept, too_large,
    bad_elements)."""
    t = pq.read_table(base + ".parquet")
    fp = np.load(base + ".npy")
    keep, why = [], {TOO_LARGE: 0, BAD_ELEMENTS: 0}
    for smi in t["smiles"].to_pylist():
        m = Chem.MolFromSmiles(smi)
        r = out_of_domain(m, DOMAIN) if m is not None else None
        keep.append(r is None)
        if r:
            why[r] += 1
    keep = np.array(keep, bool)
    if not keep.all() or (t.schema.metadata or {}).get(b"domain") != tag.encode():
        t = t.filter(pa.array(keep))
        write_shard(base, t["id"].to_pylist(), t["smiles"].to_pylist(),
                    t["smiles_nostereo"].to_pylist(), fp[keep], tag)
    return int(keep.sum()), why[TOO_LARGE], why[BAD_ELEMENTS]

CHUNK = 200_000


def read_records(name, path):
    """Yield (id, smiles) for the supported input formats."""
    if name == "surfpro":                       # row number of SurfPro.csv as id
        with open(path) as fh:
            for i, r in enumerate(csv.DictReader(fh)):
                yield str(i), r["SMILES"]
        return
    with gzip.open(path, "rt") as fh:
        if name == "chembl":                    # chembl_id, canonical_smiles, inchi, inchikey
            next(fh)
            for line in fh:
                f = line.split("\t", 2)
                yield f[0], f[1]
        elif name == "pubchem":                 # CID \t SMILES, no header
            for line in fh:
                cid, smi = line.rstrip("\n").split("\t", 1)
                yield cid, smi
        else:
            sys.exit(f"unknown database {name}")


def work(job):
    """Returns (shard, read, kept, too_large, bad_elements, status). For a
    resumed shard read = records in the chunk; the out-of-domain counts are
    complete, molecules RDKit failed on earlier are in read - kept - ..."""
    k, records, outdir = job
    base = os.path.join(outdir, f"shard_{k:05d}")
    tag = domain_tag(DOMAIN)
    if os.path.exists(base + ".npy"):           # done in an earlier run
        meta = pq.read_schema(base + ".parquet").metadata or {}
        if meta.get(b"domain") == tag.encode():
            n = pq.read_metadata(base + ".parquet").num_rows
            return k, len(records), n, 0, 0, "kept"
        return (k, len(records), *refilter(base, tag), "refiltered")
    ids, iso, flat, fps = [], [], [], []
    why = {TOO_LARGE: 0, BAD_ELEMENTS: 0}
    for i, smi in records:
        r = standardise(smi, DOMAIN)
        if r is None:
            continue
        if isinstance(r, str):
            why[r] += 1
            continue
        ids.append(i); iso.append(r[0]); flat.append(r[1]); fps.append(r[2])
    fp = np.vstack(fps) if fps else np.zeros((0, FP_BYTES), np.uint8)
    write_shard(base, ids, iso, flat, fp, tag)
    return k, len(records), len(ids), why[TOO_LARGE], why[BAD_ELEMENTS], "built"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name", choices=("chembl", "pubchem", "surfpro"))
    ap.add_argument("path")
    ap.add_argument("--outdir", default=os.path.expanduser("~/data/chemdb"))
    ap.add_argument("--workers", type=int, default=min(20, os.cpu_count()),
                    help="default 20: 30 workers left too little RAM for the desktop")
    ap.add_argument("--worker-mem-gb", type=float, default=WORKER_MEM_GB)
    ap.add_argument("--domain-from", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                          "../../data/SurfPro.csv"),
                    help="CSV whose main molecules define the size and element limits")
    args = ap.parse_args()
    outdir = os.path.join(args.outdir, args.name)
    os.makedirs(outdir, exist_ok=True)
    domain = surfpro_domain(args.domain_from)
    json.dump(dict(domain, source=os.path.abspath(args.domain_from)),
              open(os.path.join(outdir, "domain.json"), "w"), indent=1)
    print(f"domain: <= {domain['max_heavy_atoms']} heavy atoms, elements {domain['elements']}")

    recs = read_records(args.name, args.path)
    # Pool's feeder thread drains the iterator as fast as it can; bound the
    # number of chunks in flight so PubChem is not read into memory at once.
    in_flight = threading.BoundedSemaphore(2 * args.workers)

    def jobs():
        for k, chunk in enumerate(iter(lambda: list(itertools.islice(recs, CHUNK)), [])):
            in_flight.acquire()
            yield k, chunk, outdir
    n_in = n_ok = n_large = n_elem = 0
    status = {"built": 0, "refiltered": 0, "kept": 0}
    t0 = time.time()
    with Pool(args.workers, initializer=init_worker, initargs=(args.worker_mem_gb, domain),
              maxtasksperchild=20) as pool:
        for k, a, b, c, d, st in pool.imap_unordered(work, jobs()):
            in_flight.release()
            n_in += a; n_ok += b; n_large += c; n_elem += d; status[st] += 1
            print(f"shard {k:5d} {st:10s} {n_in:>12,d} read {n_ok:>12,d} kept  "
                  f"{n_large:>9,d} too large {n_elem:>10,d} other elements  "
                  f"{n_in / (time.time() - t0):8.0f} mol/s", flush=True)
    stats = {"input": args.path, "domain": domain, "read": n_in, "kept": n_ok,
             "too_large": n_large, "bad_elements": n_elem,
             "rdkit_failed": n_in - n_ok - n_large - n_elem, "shards": status,
             "minutes": (time.time() - t0) / 60}
    json.dump(stats, open(os.path.join(outdir, "stats.json"), "w"), indent=1)
    print(stats)


if __name__ == "__main__":
    main()
