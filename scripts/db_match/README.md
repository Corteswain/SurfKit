# SurfPro vs. ChEMBL / PubChem

Is each SurfPro surfactant a known compound, and what is its closest known
analogue?

| Step | Script | Env | Time |
|---|---|---|---|
| standardise SurfPro and the databases | `build_db.py` | RDKit venv | ChEMBL 3 min, PubChem ~70 min (30 cores) |
| exact lookup + GPU Tanimoto top-512 | `match.py` | base conda (torch + CUDA) | ChEMBL 20 s, PubChem a few min |
| re-rank by count Tanimoto, write table | `finalise.py` | RDKit venv | < 1 min |
| redo a stalled shard with a per-molecule timeout | `rebuild_shard.py` | RDKit venv | ~2 min/shard |

```bash
VENV=~/data/chemdb/venv/bin/python
$VENV build_db.py surfpro ../../data/SurfPro.csv
$VENV build_db.py chembl  ~/data/chemdb/raw/chembl_37_chemreps.txt.gz
$VENV build_db.py pubchem ~/data/chemdb/raw/CID-SMILES.gz
python match.py
$VENV finalise.py            # -> ../../data/SurfPro_db_matches.csv
```

**Standardisation** (`standardise.py`, identical for both sides): largest
organic fragment (counterions removed), neutralised with
`rdMolStandardize.Uncharger`, RDKit canonical SMILES with and without stereo.
A SurfPro entry counts as found if a database entry has the same
standardised molecule, so a different salt or the free acid/base still
counts. `stereo_differs` means equal only once stereo is ignored (SurfPro
seldom specifies stereo).

**Domain filter.** Database molecules are dropped if their main molecule has
more heavy atoms than the largest SurfPro main molecule (319) or an element
that no SurfPro main molecule has (allowed: C, N, O, S, P, F, Cl, Si; Na, K,
Li, Br, I, B occur in SurfPro only as counterions). Limits are computed from
SurfPro.csv at build time (`domain.json`). Kept: ChEMBL 2,758,593 / 2,897,819,
PubChem 113,332,897 / 124,862,611 (`stats.json`).

**SurfPro is in PubChem.** It was deposited via the NORMAN Suspect List
Exchange (substances 513958393-513960151, xref doi 10.1039/D4DD00393D),
including its wrong structures (peroxide ethoxylates etc.). The 1,587
compounds whose *only* source is that deposition are excluded in `match.py`
(list in `~/data/chemdb/pubchem_surfpro_deposit/`); without that, 92% of
SurfPro would be "found", 53% with it.

**Same compound, different drawing.** A charged ring drawn with the charge on
another atom (imidazolium, pyridinium) or a tautomer has a different canonical
SMILES; `finalise.py` also counts a candidate with the same standard InChIKey
as found (`same_inchikey`).

**Nearest neighbours.** 2048-bit Morgan fingerprints (r=2) of every database
molecule are compared with all SurfPro entries on the GPU (bit Tanimoto as an
fp16 matrix product, exact). Bit fingerprints cannot separate homologues: a C12
and a C6 alkyl sulfate share every bit and score 1.0. `finalise.py` therefore
re-ranks the 512 best candidates by Tanimoto of unfolded Morgan *count*
fingerprints, which do record chain length. Both values are in the table
(`*_tanimoto` = count, `*_tanimoto_bits` = bits). `*_nn_*` is the nearest
entry (the molecule itself when found); `*_nn_other_*` is the nearest
*different* molecule.

**Data.** Raw downloads, the venv, shards (PubChem: ~30 GB of fingerprints) and
intermediate results live in `~/data/chemdb`, outside the repository:
ChEMBL 37 `chembl_37_chemreps.txt.gz` (sha256 checked) and PubChem
`Compound/Extras/CID-SMILES.gz` (md5 checked, downloaded 2026-11-12).

The venv exists because the base conda RDKit is built for numpy 1.x and
crashes in numpy-returning functions under numpy 2.
