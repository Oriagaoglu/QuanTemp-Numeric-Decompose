import pickle

PATH = "retrieval_results.pkl"

with open(PATH, "rb") as f:
    qd = pickle.load(f)

print(type(qd))
print(qd.keys())

for split in qd:
    print(f"\n{split.upper()}: {len(qd[split])} claims")
    # show one example key
    first_key = next(iter(qd[split]))
    print(f"  sample key: {first_key}")
    print(f"  sample value keys: {qd[split][first_key].keys()}")

import json

with open("../..//data/QuanTemp/data/raw_data/train_claims_quantemp.json") as f:
    train_claims = json.load(f)

for i in [0, 50, 500]:
    print("\nIDX:", i)
    print("CLAIM JSON:", train_claims[i]["claim"])
    print("DECOMP PKL:", qd["train"][i]["claim"])