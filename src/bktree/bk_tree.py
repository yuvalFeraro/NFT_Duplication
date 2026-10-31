"""
bk_tree.py
==========
Thin wrapper around the sibling-project BKTree implementation
(NFT_program/NFT-Duplications-.../bk_tree_parallel.py).

Provides:
    - parse_hash_from_hex()   -> reconstruct imagehash objects from stored hex strings
    - build_trees_from_hashes() -> build 4 BKTree objects from pre-computed hash hex strings
    - query_top_k_union()     -> query all 4 trees and return the union of candidates
    - compute_distance_vector() -> compute the 4-dim Hamming distance vector
    - save_trees / load_trees -> pickle I/O
"""

import sys
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any

import imagehash
from imagehash import ImageMultiHash

# ---------------------------------------------------------------------------
# Import BKTree & distance functions from the sibling project
# ---------------------------------------------------------------------------
_SIBLING_HASH_DIR = (
    Path(__file__).resolve().parents[3]   # NFT_Duplication_Prj
    / "NFT_program"
    / "NFT-Duplications-detection-using-Image-hash-functions-main"
)

if str(_SIBLING_HASH_DIR) not in sys.path:
    sys.path.insert(0, str(_SIBLING_HASH_DIR))

from bk_tree_parallel import BKTree                                     # noqa: E402
from minimal_distance_detector import (                                 # noqa: E402
    regulat_hash_hamming_distance,
    sHash_hamming_distance,
)

# Hash name -> distance function mapping
DISTANCE_FUNCS = {
    "aHash":   regulat_hash_hamming_distance,
    "pHash":   regulat_hash_hamming_distance,
    "hsvHash": regulat_hash_hamming_distance,
    "sHash":   sHash_hamming_distance,
}


# ---------------------------------------------------------------------------
# Reconstruct imagehash objects from stored hex strings
# ---------------------------------------------------------------------------

def parse_hash_from_hex(hash_name: str, hex_str: str):
    """
    Convert a hex string (as stored in the metadata CSV) back into
    the appropriate imagehash object.

    - aHash, pHash -> imagehash.hex_to_hash(hex_str) -> ImageHash (8x8)
    - hsvHash      -> custom reconstruction (14x3 bit array from hex)
    - sHash        -> comma-separated segment hashes -> ImageMultiHash
    """
    import numpy as np

    if hash_name == "sHash":
        segments = hex_str.split(",")
        segment_hashes = [imagehash.hex_to_hash(s.strip()) for s in segments]
        return ImageMultiHash(segment_hashes)
    elif hash_name == "hsvHash":
        # colorhash uses a 14x3 bit array; its __str__ outputs a hex string
        # that encodes 14*3=42 bits
        binbits = 3
        hash_size = 14
        total_bits = hash_size * binbits  # 42
        bit_string = bin(int(hex_str, 16))[2:].zfill(total_bits)
        arr = np.array([int(b) for b in bit_string]).reshape((hash_size, binbits))
        return imagehash.ImageHash(arr.astype(bool))
    else:
        return imagehash.hex_to_hash(hex_str)


# ---------------------------------------------------------------------------
# Build four BK-Trees from pre-computed hash hex strings
# ---------------------------------------------------------------------------

def build_trees_from_hashes(
    image_rows: List[Dict[str, Any]],
    distance_threshold: int = 100,
    verbose: bool = True,
) -> Tuple[Dict[str, "BKTree"], Dict[str, Any]]:
    """
    Build 4 BK-Trees using pre-computed hash hex strings from the metadata CSVs.

    Parameters
    ----------
    image_rows : list of dicts
        Each dict must have:
            - 'filename'  : str (unique image identifier)
            - 'aHash'     : str (hex)
            - 'pHash'     : str (hex)
            - 'sHash'     : str (comma-separated hex segments)
            - 'hsvHash'   : str (hex)
    distance_threshold : int
        Passed to BKTree.__init__ (only affects search_within_distance; top_k ignores it).
    verbose : bool
        Print progress every 50 images.

    Returns
    -------
    trees : dict  {hash_name: BKTree}
    hash_store : dict  {filename: {hash_name: hash_object, ...}}
    """
    hash_names = ["aHash", "pHash", "sHash", "hsvHash"]

    trees = {
        name: BKTree(DISTANCE_FUNCS[name], distance_threshold)
        for name in hash_names
    }
    hash_store: Dict[str, Dict[str, Any]] = {}

    total = len(image_rows)
    for idx, row in enumerate(image_rows):
        filename = row["filename"]

        hashes = {}
        for hash_name in hash_names:
            h = parse_hash_from_hex(hash_name, str(row[hash_name]))
            hashes[hash_name] = h
            trees[hash_name].add(h, filename)

        hash_store[filename] = hashes

        if verbose and ((idx + 1) % 50 == 0 or idx + 1 == total):
            print(f"  BK-Tree build: {idx + 1}/{total} images inserted ...")

    return trees, hash_store


# ---------------------------------------------------------------------------
# Query four BK-Trees for top-K candidates (union)
# ---------------------------------------------------------------------------

def query_top_k_union(
    trees: Dict[str, "BKTree"],
    query_hashes: Dict[str, Any],
    k: int = 50,
) -> List[Tuple[str, Dict[str, int]]]:
    """
    Query each BK-Tree for the top-K nearest neighbours to `query_hashes`,
    then return the union of all candidates.

    Returns
    -------
    list of (candidate_filename, {hash_name: distance, ...})
        Sorted by minimum distance across hashes.
    """
    candidate_distances: Dict[str, Dict[str, int]] = {}

    for hash_name, tree in trees.items():
        results = tree.top_k(query_hashes[hash_name], k=k)
        for dist, name in results:
            if name not in candidate_distances:
                candidate_distances[name] = {}
            candidate_distances[name][hash_name] = dist

    # Sort by minimum distance across any hash
    candidates = sorted(
        candidate_distances.items(),
        key=lambda item: min(item[1].values()),
    )
    return candidates


# ---------------------------------------------------------------------------
# Compute distance vector between a query and a candidate
# ---------------------------------------------------------------------------

def compute_distance_vector(
    query_hashes: Dict[str, Any],
    candidate_hashes: Dict[str, Any],
) -> Dict[str, float]:
    """
    Compute the 4-dimensional Hamming-distance vector between two images'
    pre-computed hashes.

    Returns
    -------
    dict with keys 'dist_aHash', 'dist_pHash', 'dist_sHash', 'dist_hsvHash'
    """
    result = {}
    for hash_name, dist_fn in DISTANCE_FUNCS.items():
        d = dist_fn(query_hashes[hash_name], candidate_hashes[hash_name])
        result[f"dist_{hash_name}"] = d
    return result


# ---------------------------------------------------------------------------
# Pickle I/O
# ---------------------------------------------------------------------------

def save_trees(
    trees: Dict[str, "BKTree"],
    hash_store: Dict[str, Dict[str, Any]],
    output_dir: Path,
    prefix: str = "train",
) -> None:
    """Save each BK-Tree and the hash_store to `output_dir` as pickle files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for hash_name, tree in trees.items():
        path = output_dir / f"{prefix}_tree_{hash_name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(tree, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved {path}")

    store_path = output_dir / f"{prefix}_hash_store.pkl"
    with open(store_path, "wb") as f:
        pickle.dump(hash_store, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved {store_path}")


def load_trees(
    input_dir: Path,
    prefix: str = "train",
) -> Tuple[Dict[str, "BKTree"], Dict[str, Dict[str, Any]]]:
    """Load 4 BK-Trees + hash_store from pickle files in `input_dir`."""
    hash_names = ["aHash", "pHash", "sHash", "hsvHash"]
    trees = {}
    for hash_name in hash_names:
        path = input_dir / f"{prefix}_tree_{hash_name}.pkl"
        with open(path, "rb") as f:
            trees[hash_name] = pickle.load(f)
        print(f"  Loaded {path}")

    store_path = input_dir / f"{prefix}_hash_store.pkl"
    with open(store_path, "rb") as f:
        hash_store = pickle.load(f)
    print(f"  Loaded {store_path}")

    return trees, hash_store
