#!/usr/bin/env python3
"""Rebuild single shards with a per-molecule timeout.

RDKit's C++ code cannot be interrupted from Python, and a few PubChem entries
keep it busy for hours (shard 615 stalled the full PubChem build). Here every
molecule is sent to a child process; if no answer arrives within --timeout
seconds the child is killed, the molecule skipped and a new child started.

    python rebuild_shard.py pubchem ~/data/chemdb/raw/CID-SMILES.gz 615
"""
import argparse, itertools, json, os, time
from multiprocessing import Pipe, Process

import numpy as np

import build_db as B
from standardise import BAD_ELEMENTS, TOO_LARGE, standardise, surfpro_domain


def child(conn, domain, mem_gb):
    B.init_worker(mem_gb, domain)
    while True:
        conn.send(standardise(conn.recv(), B.DOMAIN))


def rebuild(k, chunk, outdir, domain, timeout):
    tag = B.domain_tag(domain)
    proc = conn = None
    ids, iso, flat, fps = [], [], [], []
    why = {TOO_LARGE: 0, BAD_ELEMENTS: 0, "timeout": 0, "rdkit_failed": 0}
    skipped = []
    t0 = time.time()
    for i, smi in chunk:
        if proc is None:
            conn, c2 = Pipe()
            proc = Process(target=child, args=(c2, domain, B.WORKER_MEM_GB), daemon=True)
            proc.start()
        conn.send(smi)
        if not conn.poll(timeout):
            print(f"  timeout: id {i} ({len(smi)} chars) {smi[:80]}", flush=True)
            proc.kill(); proc.join(); proc = None
            why["timeout"] += 1; skipped.append(i)
            continue
        r = conn.recv()
        if r is None:
            why["rdkit_failed"] += 1
        elif isinstance(r, str):
            why[r] += 1
        else:
            ids.append(i); iso.append(r[0]); flat.append(r[1]); fps.append(r[2])
    if proc is not None:
        proc.kill()
    fp = np.vstack(fps) if fps else np.zeros((0, B.FP_BYTES), np.uint8)
    B.write_shard(os.path.join(outdir, f"shard_{k:05d}"), ids, iso, flat, fp, tag)
    return {"shard": k, "read": len(chunk), "kept": len(ids), **why,
            "timed_out_ids": skipped, "minutes": round((time.time() - t0) / 60, 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name", choices=("chembl", "pubchem", "surfpro"))
    ap.add_argument("path")
    ap.add_argument("shards", nargs="+", type=int)
    ap.add_argument("--outdir", default=os.path.expanduser("~/data/chemdb"))
    ap.add_argument("--timeout", type=float, default=10.0, help="seconds per molecule")
    ap.add_argument("--domain-from", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                          "../../data/SurfPro.csv"))
    args = ap.parse_args()
    domain = surfpro_domain(args.domain_from)
    outdir = os.path.join(args.outdir, args.name)
    os.makedirs(outdir, exist_ok=True)
    recs, pos = B.read_records(args.name, args.path), 0
    for k in sorted(set(args.shards)):
        start = k * B.CHUNK
        chunk = list(itertools.islice(recs, start - pos, start - pos + B.CHUNK))
        pos = start + len(chunk)
        info = rebuild(k, chunk, outdir, domain, args.timeout)
        print(info, flush=True)
        with open(os.path.join(outdir, "rebuilt_shards.jsonl"), "a") as fh:
            fh.write(json.dumps(info) + "\n")


if __name__ == "__main__":
    main()
