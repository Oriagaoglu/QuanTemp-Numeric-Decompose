import os
import re
import json
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi


# =========================
# CONFIG (EDIT THESE PATHS)
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent

# QuanTemp paths
QUANTEMP_ROOT = PROJECT_ROOT / "data" / "QuanTemp" / "data"
RAW_DATA_DIR = QUANTEMP_ROOT / "raw_data"
BM25_REF_PATH = QUANTEMP_ROOT / "bm25_scored_evidence" / "bm25_top_100_claimdecomp.json"
CORPUS_PATH = PROJECT_ROOT / "final_english_corpus.json"

# Outputs / caches
EXPERIMENT_DIR = PROJECT_ROOT / "outputs" / "experiment_2026_01_15"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

# You said you already have this from Colab
QUANTEMP_DECOMP_CACHE = EXPERIMENT_DIR / "quantemp_decompositions.pkl"

# Caches to build/load locally
BM25_INDEX_CACHE = EXPERIMENT_DIR / "bm25_index.pkl"
BM25_CORPUS_CACHE = EXPERIMENT_DIR / "bm25_corpus_ids.pkl"
RETRIEVAL_CACHE = EXPERIMENT_DIR / "retrieval_results.pkl"


# =========================
# TOKENIZER (LOCAL SAFE)
# =========================
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

def tokenize(text: str):
    """
    Simple, deterministic tokenizer that tends to work fine for BM25 baselines.
    Replace with your original tokenize() if you want to keep parity.
    """
    if not text:
        return []
    text = text.lower()
    return _TOKEN_RE.findall(text)


# =========================
# LOAD CLAIMS
# =========================
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

print("Loading QuanTemp claims...")
train_claims = load_json(RAW_DATA_DIR / "train_claims_quantemp.json")
val_claims   = load_json(RAW_DATA_DIR / "val_claims_quantemp.json")
test_claims  = load_json(RAW_DATA_DIR / "test_claims_quantemp.json")
print(f"✓ Train: {len(train_claims)}")
print(f"✓ Val:   {len(val_claims)}")
print(f"✓ Test:  {len(test_claims)}")


# =========================
# LOAD REPO REFERENCE
# =========================
print("\nLoading repo-provided BM25+ClaimDecomp results...")
repo_bm25_data = load_json(BM25_REF_PATH)

# repo format: list[ { "query_id": ..., "docs": [...] }, ... ]
repo_evidence_dict = {}
for item in repo_bm25_data:
    qid = item.get("query_id")
    docs = item.get("docs", [])
    if qid is not None:
        repo_evidence_dict[qid] = docs[:100]

print(f"✓ Loaded evidence for {len(repo_evidence_dict)} query_ids (repo reference)")


# =========================
# LOAD CORPUS
# =========================
print(f"\nLoading evidence corpus from: {CORPUS_PATH}")
corpus_dict = load_json(CORPUS_PATH)

# Ensure stable doc_id ordering
# Some corpora use numeric string keys; if not numeric, fallback to lexical sort.
def _sort_key(x):
    try:
        return int(x)
    except:
        return x

corpus_ids = sorted(corpus_dict.keys(), key=_sort_key)
evidence_corpus = [corpus_dict[doc_id] for doc_id in corpus_ids]

print(f"✓ Loaded {len(evidence_corpus)} evidence documents")
print(f"  Sample doc[0]: {evidence_corpus[0][:120]}...")


# =========================
# BUILD / LOAD BM25 INDEX
# =========================
if BM25_INDEX_CACHE.exists() and BM25_CORPUS_CACHE.exists():
    print("\nLoading cached BM25 index...")
    with open(BM25_INDEX_CACHE, "rb") as f:
        bm25_index = pickle.load(f)
    with open(BM25_CORPUS_CACHE, "rb") as f:
        cached_corpus_ids = pickle.load(f)
    print(f"✓ Loaded BM25 index for {len(cached_corpus_ids)} documents")

    # Safety check: corpus alignment
    if len(cached_corpus_ids) != len(corpus_ids):
        raise RuntimeError(
            "Cached BM25 corpus_ids length differs from current corpus. "
            "Delete bm25_index.pkl and bm25_corpus_ids.pkl and rerun."
        )
else:
    print("\nBuilding BM25 index (local build)...")
    corpus_tokens = []
    for doc in tqdm(evidence_corpus, desc="Tokenizing corpus"):
        corpus_tokens.append(tokenize(doc))

    print("Creating BM25 index...")
    bm25_index = BM25Okapi(corpus_tokens)

    with open(BM25_INDEX_CACHE, "wb") as f:
        pickle.dump(bm25_index, f)
    with open(BM25_CORPUS_CACHE, "wb") as f:
        pickle.dump(corpus_ids, f)

    print(f"✓ Saved BM25 index to: {BM25_INDEX_CACHE}")
    print(f"✓ Saved corpus ids to: {BM25_CORPUS_CACHE}")


# =========================
# LOAD DECOMPOSITIONS PKL
# =========================
print("\nLoading cached decompositions...")
if not QUANTEMP_DECOMP_CACHE.exists():
    raise FileNotFoundError(
        f"Could not find {QUANTEMP_DECOMP_CACHE}. "
        f"Copy your Colab-generated quantemp_decompositions.pkl into {EXPERIMENT_DIR}."
    )

with open(QUANTEMP_DECOMP_CACHE, "rb") as f:
    quantemp_decomp = pickle.load(f)

# Basic validation
for split in ["train", "val", "test"]:
    if split not in quantemp_decomp:
        raise ValueError(f"quantemp_decomp missing split: {split}")

print("✓ Decompositions loaded.")


# =========================
# WEIGHTED RETRIEVAL CONFIG
# =========================
# Combine BM25 scores from:
# - raw claim query (baseline)
# - decomposed subqueries (max pooled)
# Final score = W_RAW * raw_score + W_DECOMP * decomp_score
W_RAW = 0.4
W_DECOMP = 0.6


# =========================
# RETRIEVAL FUNCTIONS
# =========================
def retrieve_baseline(claim_text: str, top_k: int = 100):
    query_tokens = tokenize(claim_text)
    scores = bm25_index.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [evidence_corpus[i] for i in top_indices]

def retrieve_decomposed(subqueries, top_k: int = 100):
    if not subqueries:
        return []

    # BM25Okapi.get_scores returns a numpy array-like for each subquery
    all_scores = []
    for sq in subqueries:
        sq_tokens = tokenize(sq)
        all_scores.append(bm25_index.get_scores(sq_tokens))

    # MAX pooling across subqueries
    max_scores = np.max(np.vstack(all_scores), axis=0)
    top_indices = np.argsort(max_scores)[::-1][:top_k]
    return [evidence_corpus[i] for i in top_indices]

def retrieve_baseline_with_scores(claim_text: str):
    """Return full BM25 scores over the corpus for the raw claim query."""
    query_tokens = tokenize(claim_text)
    return bm25_index.get_scores(query_tokens)


def retrieve_decomposed_with_scores(subqueries):
    """Return full BM25 scores over the corpus for decomposed subqueries (max pooled)."""
    if not subqueries:
        # return zeros so weighted combination still works
        return np.zeros(len(evidence_corpus), dtype=np.float32)

    all_scores = []
    for sq in subqueries:
        sq_tokens = tokenize(sq)
        all_scores.append(bm25_index.get_scores(sq_tokens))

    return np.max(np.vstack(all_scores), axis=0)


def retrieve_weighted(claim_text: str, subqueries, top_k: int = 100, w_raw: float = W_RAW, w_decomp: float = W_DECOMP):
    """Weighted combination of raw-query BM25 and decomposed-query BM25 scores."""
    raw_scores = retrieve_baseline_with_scores(claim_text)
    decomp_scores = retrieve_decomposed_with_scores(subqueries)

    # Ensure numpy arrays (rank_bm25 may return list-like)
    raw_scores = np.asarray(raw_scores, dtype=np.float32)
    decomp_scores = np.asarray(decomp_scores, dtype=np.float32)

    combined = (w_raw * raw_scores) + (w_decomp * decomp_scores)
    top_indices = np.argsort(combined)[::-1][:top_k]
    return [evidence_corpus[i] for i in top_indices]


# =========================
# RUN / LOAD RETRIEVAL
# =========================
retrieval_results = {
    "baseline": {"train": {}, "val": {}, "test": {}},
    "decomposed": {"train": {}, "val": {}, "test": {}},
    "repo": {"train": {}, "val": {}, "test": {}},
}

if RETRIEVAL_CACHE.exists():
    print("\nLoading cached retrieval results...")
    with open(RETRIEVAL_CACHE, "rb") as f:
        retrieval_results = pickle.load(f)
    print("✓ Loaded cached results")
else:
    print("\nPerforming BM25 retrieval for all claims...")

    splits = {"train": train_claims, "val": val_claims, "test": test_claims}

    for split_name, claims in splits.items():
        print(f"\n  Retrieving for {split_name}...")

        # R0: baseline
        for idx, claim_obj in enumerate(tqdm(claims, desc=f"    R0 {split_name}", leave=False)):
            claim_text = claim_obj["claim"]
            retrieval_results["baseline"][split_name][idx] = retrieve_baseline(claim_text, top_k=100)

        # R1: decomposed (WEIGHTED raw + subqueries)
        for idx in tqdm(range(len(claims)), desc=f"    R1 {split_name}", leave=False):
            claim_text = claims[idx]["claim"]
            subqueries = quantemp_decomp[split_name][idx].get("subqueries", [])
            retrieval_results["decomposed"][split_name][idx] = retrieve_weighted(
                claim_text,
                subqueries,
                top_k=100,
                w_raw=W_RAW,
                w_decomp=W_DECOMP,
            )

        # R2: repo (map claim id -> repo query_id)
        for idx, claim_obj in enumerate(claims):
            claim_id = (
                claim_obj.get("id")
                or claim_obj.get("query_id")
                or claim_obj.get("claim_id")
                or idx
            )
            retrieval_results["repo"][split_name][idx] = repo_evidence_dict.get(claim_id, [])

    with open(RETRIEVAL_CACHE, "wb") as f:
        pickle.dump(retrieval_results, f)

    print(f"\n✓ Saved retrieval results to: {RETRIEVAL_CACHE}")


# =========================
# STATS
# =========================
print("\nRetrieval Statistics:")
for method in ["baseline", "decomposed", "repo"]:
    train_count = sum(len(docs) > 0 for docs in retrieval_results[method]["train"].values())
    val_count   = sum(len(docs) > 0 for docs in retrieval_results[method]["val"].values())
    test_count  = sum(len(docs) > 0 for docs in retrieval_results[method]["test"].values())

    print(f"\n  {method.upper()}:")
    print(f"    Train: {train_count}/{len(train_claims)} claims with evidence")
    print(f"    Val:   {val_count}/{len(val_claims)} claims with evidence")
    print(f"    Test:  {test_count}/{len(test_claims)} claims with evidence")

print("\nDone.")
print(f"Artifacts written under: {EXPERIMENT_DIR}")
