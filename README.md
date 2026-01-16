# QuanTemp Numeric Decomposition (Group 17)

Claim decomposition experiments targeting **numeric/temporal** claims:
**decomposition → BM25 retrieval → NLI fact checking**.

This repo contains only our scripts/notebooks. Large datasets/corpora and model checkpoints are excluded.

## What’s in this repo
- `retrieval.py`: builds/loads a local BM25 index and writes retrieval caches
- `bart.py`: example HuggingFace training loop using cached retrieval outputs
- `train_and_evaluate.ipynb`: experiments / analysis notebook
- `requirements.txt`: dependencies

## What we commit vs not
- ✅ Commit: retrieval caches for reproducibility (under `outputs/`):
  - `bm25_index.pkl`, `bm25_corpus_ids.pkl`, `retrieval_results.pkl`
- ❌ Do not commit: datasets/corpora under `data/`, `models/`, large corpus JSON, training checkpoints

## Required data (not in git)
Place QuanTemp-format data locally (paths can be changed in scripts):
- Claims: `data/quantemp/data/raw_data/{train,val,test}_claims_quantemp.json`
- Repo BM25 reference (ClaimDecomp): `data/quantemp/data/bm25_scored_evidence/bm25_top_100_claimdecomp.json`
- Evidence corpus for BM25 indexing: `final_english_corpus.json`

## Reproduce (high level)
1) Install deps: `pip install -r requirements.txt`
2) Run retrieval: execute `retrieval.py` (produces caches under `outputs/`)
3) Run evaluation / analysis: use `train_and_evaluate.ipynb`
4) Fine-tune verifier: run `bart.py` (reads retrieval caches)

## Summary of findings
- Decomposition can hurt BM25 when queries become too “explorative” (Recall/MRR drop).
- A weighted approach combining raw + decomposed retrieval scores was explored; numeric-focused decomposition showed promising MRR.

## Results (test)

| Setup | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| BART baseline (no decomposition) | 0.6128 | 0.5730 | 0.6277 |
| RoBERTa-large + decomposed retrieval | 0.6373 | 0.5891 | 0.6494 |
| BART + decomposed retrieval | 0.6369 | 0.5889 | 0.6495 |

Notes:
- **Macro-F1** treats all classes equally.
- **Weighted-F1** accounts for class frequency.