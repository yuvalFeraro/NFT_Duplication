"""
analyse_distances.py
Investigates WHY DISC-21 results are so much better than expected.
"""
import csv
import shutil
import tempfile
from pathlib import Path

import imagehash
import openpyxl
import numpy as np

BASE_V2   = Path(r"c:\Users\ferar\OneDrive - Technion\NFT_Duplication_Prj\NFT_program_V2")
BASE_DATA = Path(r"c:\Users\ferar\OneDrive - Technion\NFT_Duplication_Prj\data_sets")
V1        = BASE_V2 / "NFT_program_V1"

def load_csv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return {r["filename"]: r for r in csv.DictReader(f)}

def safe_load_xlsx(src):
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tp = Path(tmp.name)
    shutil.copy2(src, tp)
    wb = openpyxl.load_workbook(tp, read_only=True, data_only=True)
    rows = list(wb.active.iter_rows(values_only=True))
    wb.close(); tp.unlink(missing_ok=True)
    return rows

def hamming_int(h1_hex, h2_hex, bits=64):
    try:
        return bin(int(h1_hex, 16) ^ int(h2_hex, 16)).count("1")
    except Exception:
        return None

def hsv_hamming(h1_hex, h2_hex):
    try:
        total_bits = 42
        b1 = bin(int(h1_hex, 16))[2:].zfill(total_bits)
        b2 = bin(int(h2_hex, 16))[2:].zfill(total_bits)
        return sum(c1 != c2 for c1, c2 in zip(b1, b2))
    except Exception:
        return None

# Load hash stores
refs = load_csv(V1 / "data" / "hashes" / "references_hashes.csv")
qrys = load_csv(V1 / "data" / "hashes" / "queries_hashes.csv")

# Load mapping
rows = safe_load_xlsx(BASE_DATA / "DISC-21" / "2409_data" / "combined_queries_distances.xlsx")
mapping = {r[0]: r[1] for r in rows[1:]}

print("=" * 65)
print("Analysis: query-to-TRUE-REFERENCE distances (our recomputed hashes)")
print("=" * 65)

a_dists, p_dists, hsv_dists = [], [], []
identical_hsv = 0
missing = 0

for q, r in mapping.items():
    qr = qrys.get(q)
    rr = refs.get(r)
    if not qr or not rr:
        missing += 1
        continue
    ad = hamming_int(qr["aHash"], rr["aHash"])
    pd = hamming_int(qr["pHash"], rr["pHash"])
    hd = hsv_hamming(qr["hsvHash"], rr["hsvHash"])
    if ad is not None: a_dists.append(ad)
    if pd is not None: p_dists.append(pd)
    if hd is not None:
        hsv_dists.append(hd)
        if hd == 0:
            identical_hsv += 1

print(f"Pairs analysed: {len(a_dists)}  (missing: {missing})")
print()

for name, dists in [("aHash", a_dists), ("pHash", p_dists), ("hsvHash", hsv_dists)]:
    import statistics
    print(f"{name}:")
    print(f"  min={min(dists)}  median={statistics.median(dists):.1f}  mean={statistics.mean(dists):.1f}  max={max(dists)}")
    for t in [5, 10, 15, 18]:
        n = sum(1 for d in dists if d < t)
        print(f"  < {t:2d} : {n:3d}/{len(dists)}  ({100*n/len(dists):.1f}%)")
    print()

print(f"Queries with IDENTICAL hsvHash to their true reference: {identical_hsv}/{len(hsv_dists)}")
print()

# Now check: could a query accidentally match a NON-reference in the base?
# Load train split
train_base_rows = list(csv.DictReader(open(V1/"data"/"splits"/"train_base.csv", encoding="utf-8")))
train_base_fns = set(r["filename"] for r in train_base_rows)
train_base_store = {fn: refs[fn] for fn in train_base_fns if fn in refs}

train_dup_rows = list(csv.DictReader(open(V1/"data"/"splits"/"train_duplications.csv", encoding="utf-8")))

print("=" * 65)
print("Checking: do train queries get their hsvHash min from TRUE ref or ANOTHER ref?")
print("=" * 65)
from_true_ref = 0
from_other_ref = 0
true_ref_dist_list = []
min_dist_list = []

for row in train_dup_rows:
    qfn = row["filename"]
    rfn = row["reference_filename"]
    qr = qrys.get(qfn)
    if not qr: continue
    q_hsv = qr["hsvHash"]
    true_ref_dist = hsv_hamming(q_hsv, refs[rfn]["hsvHash"]) if rfn in refs else None
    # Find min distance to all train base images
    min_d = None
    min_fn = None
    for bfn, br in train_base_store.items():
        d = hsv_hamming(q_hsv, br["hsvHash"])
        if d is not None and (min_d is None or d < min_d):
            min_d = d
            min_fn = bfn
    if true_ref_dist is not None:
        true_ref_dist_list.append(true_ref_dist)
    if min_d is not None:
        min_dist_list.append(min_d)
    if min_fn == rfn:
        from_true_ref += 1
    else:
        from_other_ref += 1

print(f"Min hsvHash comes from TRUE reference    : {from_true_ref}/{len(train_dup_rows)}")
print(f"Min hsvHash comes from ANOTHER base image: {from_other_ref}/{len(train_dup_rows)}")
print()
import statistics
if true_ref_dist_list:
    print(f"hsvHash dist to TRUE reference:  mean={statistics.mean(true_ref_dist_list):.1f}  median={statistics.median(true_ref_dist_list):.1f}  max={max(true_ref_dist_list)}")
if min_dist_list:
    print(f"hsvHash dist to CLOSEST base:    mean={statistics.mean(min_dist_list):.1f}  median={statistics.median(min_dist_list):.1f}  max={max(min_dist_list)}")
