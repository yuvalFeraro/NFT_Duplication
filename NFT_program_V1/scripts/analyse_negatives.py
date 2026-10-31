"""
analyse_negatives.py
Checks what the min hsvHash distances from NEGATIVES to base actually look like,
and verifies they are truly being evaluated in step4.
"""
import csv
import statistics
from pathlib import Path

V1 = Path(r"c:\Users\ferar\OneDrive - Technion\NFT_Duplication_Prj\NFT_program_V2\NFT_program_V1")

def load_csv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return {r["filename"]: r for r in csv.DictReader(f)}

def hsv_hamming(h1, h2):
    try:
        b1 = bin(int(h1, 16))[2:].zfill(42)
        b2 = bin(int(h2, 16))[2:].zfill(42)
        return sum(c1 != c2 for c1, c2 in zip(b1, b2))
    except Exception:
        return None

refs = load_csv(V1 / "data" / "hashes" / "references_hashes.csv")
qrys = load_csv(V1 / "data" / "hashes" / "queries_hashes.csv")

# Load splits
def load_split(name):
    return list(csv.DictReader(open(V1 / "data" / "splits" / name, encoding="utf-8")))

train_base     = load_split("train_base.csv")
train_negs     = load_split("train_negatives.csv")
train_dups     = load_split("train_duplications.csv")

base_store = {r["filename"]: refs[r["filename"]] for r in train_base if r["filename"] in refs}

print("=" * 65)
print(f"Train base size    : {len(base_store)}")
print(f"Train negatives    : {len(train_negs)}")
print(f"Train duplications : {len(train_dups)}")
print("=" * 65)

# --- Min hsvHash distances: NEGATIVES vs BASE ---
neg_min_dists = []
neg_caught_at_5 = 0
for row in train_negs:
    fn = row["filename"]
    if fn not in refs:
        continue
    h = refs[fn]["hsvHash"]
    min_d = min(
        (hsv_hamming(h, br["hsvHash"]) for br in base_store.values()
         if hsv_hamming(h, br["hsvHash"]) is not None),
        default=None
    )
    if min_d is not None:
        neg_min_dists.append(min_d)
        if min_d < 5:
            neg_caught_at_5 += 1

print("\nNegatives: min hsvHash distance to ANY base image")
print(f"  min={min(neg_min_dists)}  median={statistics.median(neg_min_dists):.1f}"
      f"  mean={statistics.mean(neg_min_dists):.1f}  max={max(neg_min_dists)}")
for t in [1, 2, 3, 4, 5, 10]:
    n = sum(1 for d in neg_min_dists if d < t)
    print(f"  < {t}: {n:3d}/{len(neg_min_dists)}  ({100*n/len(neg_min_dists):.1f}%)  -> FP if threshold={t}")

# --- Min hsvHash distances: DUPLICATES vs BASE ---
dup_min_dists = []
dup_caught_at_5 = 0
for row in train_dups:
    fn = row["filename"]
    if fn not in qrys:
        continue
    h = qrys[fn]["hsvHash"]
    min_d = min(
        (hsv_hamming(h, br["hsvHash"]) for br in base_store.values()
         if hsv_hamming(h, br["hsvHash"]) is not None),
        default=None
    )
    if min_d is not None:
        dup_min_dists.append(min_d)
        if min_d < 5:
            dup_caught_at_5 += 1

print("\nDuplicates: min hsvHash distance to ANY base image")
print(f"  min={min(dup_min_dists)}  median={statistics.median(dup_min_dists):.1f}"
      f"  mean={statistics.mean(dup_min_dists):.1f}  max={max(dup_min_dists)}")
for t in [1, 2, 3, 4, 5, 10]:
    n = sum(1 for d in dup_min_dists if d < t)
    print(f"  < {t}: {n:3d}/{len(dup_min_dists)}  ({100*n/len(dup_min_dists):.1f}%)  -> TP if threshold={t}")

print("\n" + "=" * 65)
print("KEY QUESTION: are negatives (R*) a different type of image than duplicates (Q*)?")
# Check if negatives are R-files and duplicates are Q-files
neg_fns = [r["filename"] for r in train_negs]
dup_fns = [r["filename"] for r in train_dups]
print(f"  Negatives filenames sample : {neg_fns[:5]}")
print(f"  Duplicates filenames sample: {dup_fns[:5]}")
print(f"  Negatives start with R: {sum(1 for f in neg_fns if f.startswith('R'))}/{len(neg_fns)}")
print(f"  Duplicates start with Q: {sum(1 for f in dup_fns if f.startswith('Q'))}/{len(dup_fns)}")
