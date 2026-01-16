import os
import json
import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    BartForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    get_last_checkpoint
)
from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score
from torch import nn

# ===== 0. PYTORCH 2.4 & H100 ACCELERATION =====
# Enable TF32 for high-performance matrix math
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# PyTorch 2.4 SDPA (Scaled Dot Product Attention) optimization
# This will automatically use FlashAttention-2 if compatible
torch.set_float32_matmul_precision('high')

# ===== CONFIGURATION =====
BASE_DIR = "/workspace" 
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "experiments")
RETRIEVAL_CACHE = os.path.join(DATA_DIR, "retrieval_results.pkl")
VERIFIER_MODEL = "facebook/bart-large"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== 1. LOAD DATA =====
print("Loading data...")
with open(os.path.join(DATA_DIR, "train_claims_quantemp.json")) as f:
    train_claims = json.load(f)
with open(os.path.join(DATA_DIR, "val_claims_quantemp.json")) as f:
    val_claims = json.load(f)
with open(os.path.join(DATA_DIR, "test_claims_quantemp.json")) as f:
    test_claims = json.load(f)

with open(RETRIEVAL_CACHE, "rb") as f:
    retrieval_results = pickle.load(f)

# ===== 2. PRE-PROCESSING =====
def normalize_label(label):
    l = str(label).lower()
    if any(x in l for x in ["support", "true", "correct"]): return 0
    if any(x in l for x in ["refute", "false", "pants"]): return 1
    return 2

def create_examples(claims, evidence_dict, top_k=3):
    examples = []
    for idx, obj in enumerate(claims):
        label = normalize_label(obj["label"])
        evs = evidence_dict.get(idx, [])
        for ev in evs[:top_k]:
            if len(ev.strip()) < 20: continue
            examples.append({"claim": obj["claim"], "evidence": ev[:1024], "label": label})
    return examples

trainset = create_examples(train_claims, retrieval_results["decomposed"]["train"])
valset = create_examples(val_claims, retrieval_results["decomposed"]["val"])

# ===== 3. TOKENIZATION =====
tokenizer = AutoTokenizer.from_pretrained(VERIFIER_MODEL)

def process_ds(examples):
    ds = Dataset.from_list(examples)
    return ds.map(lambda x: tokenizer(
        x["evidence"], 
        x["claim"], 
        truncation=True, 
        padding="max_length", 
        max_length=512
    ), batched=True)

train_ds = process_ds(trainset)
val_ds = process_ds(valset)

# ===== 4. H100 OPTIMIZED TRAINER =====
weights = compute_class_weight("balanced", classes=np.array([0,1,2]), y=[ex["label"] for ex in trainset])

class BartH100Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        # Using a fixed weight tensor on the correct device
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(DEVICE))
        loss = loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "bart_h100_v2"),
    
    # Precision & Speed (H100 + PyTorch 2.4)
    bf16=True,                          # Bfloat16 is superior on H100
    tf32=True,                          # Faster float32 operations
    optim="adamw_torch_fused",          # Fused kernel is much faster
    torch_compile=True,                 # PyTorch 2.x specific optimization for ~20% speedup
    
    # Efficiency & Batching
    per_device_train_batch_size=32,     # H100 has 80GB, we can go high
    gradient_accumulation_steps=1,      # Lower is better for throughput if VRAM allows
    dataloader_num_workers=8,           # Parallelize data loading
    
    # Loop Config
    learning_rate=1e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    report_to="none"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": f1_score(labels, preds, average="macro")}

trainer = BartH100Trainer(
    model=BartForSequenceClassification.from_pretrained(VERIFIER_MODEL, num_labels=3),
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics
)

# ===== 5. EXECUTION =====
print("Starting H100 Training...")
last_ckpt = get_last_checkpoint(training_args.output_dir)
trainer.train(resume_from_checkpoint=last_ckpt if last_ckpt else None)

# Save the best model
trainer.save_model(os.path.join(OUTPUT_DIR, "final_bart_h100_model"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_bart_h100_model"))

# ===== 6. EVALUATION =====
print("Evaluating on Test Set...")
model = trainer.model.eval()

def test_eval(claims, evidence_dict):
    preds, truths = [], []
    for idx, obj in enumerate(tqdm(claims)):
        label = normalize_label(obj["label"])
        evs = evidence_dict.get(idx, [])
        if not evs:
            preds.append(2); truths.append(label); continue
        
        # Inference with BF16 and Autocast
        batch = tokenizer([e[:1024] for e in evs[:5]], [obj["claim"]]*len(evs[:5]), 
                          padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(**batch).logits
            avg_prob = torch.softmax(logits, dim=1).mean(dim=0)
            preds.append(int(torch.argmax(avg_prob).cpu()))
            truths.append(label)
    
    return f1_score(truths, preds, average="macro")

final_f1 = test_eval(test_claims, retrieval_results["decomposed"]["test"])
print(f"BART-Large H100 Test Macro-F1: {final_f1:.4f}")