"""
Phase 1 v4: Fixed vocabulary extraction + domain-specificity filter + safe embedding warm-up.

Root causes fixed so far:
1. ByteLevel decoding: custom tokenizer vocab keys were byte-encoded garbage;
   now decoded back to real UTF-8 text before comparison.
2. SentencePiece prefix mismatch: words already known to the base tokenizer
   (with or without the leading "▁") are excluded via is_already_known().
3. Colloquial/common-word pollution: a stopword list + domain-vs-general
   frequency ratio filter removes everyday words that aren't truly domain-specific.
4. Embedding collapse: HF's mean-resizing (multivariate normal init) was
   producing near-identical vectors for all new rows (std=0.023), which
   corrupted softmax for the whole vocabulary via the tied lm_head.
   Disabled via mean_resizing=False.
"""
import os
import torch
from tokenizers import Tokenizer, decoders
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
from config import CONFIG
from utils import find_file
from hazm import stopwords_list
import urllib.request
from collections import Counter
import re

login(token=UserSecretsClient().get_secret("HF_TOKEN"))

# --- 1. Locate input files ---
CORPUS_PATH = find_file("bilingual_med_legal_corpus.txt")
TOKENIZER_JSON_PATH = find_file("med_legal_tokenizer.json")
assert CORPUS_PATH is not None, "Corpus file not found!"
assert TOKENIZER_JSON_PATH is not None, "Custom tokenizer file not found!"
print(f"[OK] Corpus: {CORPUS_PATH}")
print(f"[OK] Custom tokenizer: {TOKENIZER_JSON_PATH}")

# --- 2. Extract real, non-duplicate domain tokens ---
base_tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])

raw_tok = Tokenizer.from_file(TOKENIZER_JSON_PATH)
raw_tok.decoder = decoders.ByteLevel()

def is_already_known(word, tok):
    # Checks whether this word is already a single token in the base tokenizer,
    # both as a mid-word piece and as a word-start piece (SentencePiece's "▁" prefix)
    ids_mid = tok(word, add_special_tokens=False)["input_ids"]
    ids_start = tok(" " + word, add_special_tokens=False)["input_ids"]
    return len(ids_mid) == 1 or len(ids_start) == 1

tokens_to_add = []
for token_str, token_id in raw_tok.get_vocab().items():
    real_text = raw_tok.decode([token_id]).strip()
    if not real_text or len(real_text) < 3:
        continue
    if not is_already_known(real_text, base_tokenizer):
        tokens_to_add.append(real_text)

# ============================================================
# Final filter: remove common words via stopword list + frequency ratio
# ============================================================
print("Loading Persian common-word lists...")

# --- Source 1: stopword lists (prepositions, pronouns) ---
stopwords_hazm = set(stopwords_list())
url = "https://raw.githubusercontent.com/kharazi/persian-stopwords/master/persian"
urllib.request.urlretrieve(url, "persian_stopwords_github.txt")
with open("persian_stopwords_github.txt", "r", encoding="utf-8") as f:
    stopwords_github = set(line.strip() for line in f if line.strip())
PERSIAN_STOPWORDS = stopwords_hazm | stopwords_github
print(f"[OK] Loaded {len(PERSIAN_STOPWORDS)} Persian stopwords.")

# --- Source 2: word frequency in our own domain corpus ---
def word_counts(text_lines):
    counter = Counter()
    for line in text_lines:
        words = re.findall(r"[\w\u0600-\u06FF]+", line.lower())
        counter.update(words)
    return counter

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    domain_lines = f.readlines()
domain_counts = word_counts(domain_lines)
domain_total = sum(domain_counts.values())
print(f"[OK] Counted {domain_total:,} words in the domain corpus.")

# --- Source 3: word frequency in a general Persian corpus (Wikipedia, streamed) ---
from datasets import load_dataset

print("Downloading a sample of Persian Wikipedia for comparison...")
wiki = load_dataset("wikimedia/wikipedia", "20231101.fa", split="train", streaming=True)
general_lines = []
for i, row in enumerate(wiki):
    general_lines.append(row["text"])
    if i >= 3000:   # 3000 articles is enough for this comparison
        break

general_counts = word_counts(general_lines)
general_total = sum(general_counts.values())
print(f"[OK] Counted {general_total:,} words in the general corpus.")

# --- Filter based on domain-vs-general frequency ratio ---
RATIO_THRESHOLD = 5.0   # word must be at least 5x more frequent in-domain than in general Persian

def is_domain_specific(word, domain_counts, domain_total, general_counts, general_total, ratio_threshold):
    domain_freq = domain_counts.get(word, 0) / domain_total
    general_freq = general_counts.get(word, 0) / general_total
    if general_freq == 0:
        return domain_freq > 0   # word absent from general corpus -> likely domain-specific
    return (domain_freq / general_freq) >= ratio_threshold

filtered_tokens = []
removed_as_stopword = 0
removed_as_common = 0

for token in tokens_to_add:
    if token in PERSIAN_STOPWORDS:
        removed_as_stopword += 1
        continue
    if not is_domain_specific(token, domain_counts, domain_total, general_counts, general_total, RATIO_THRESHOLD):
        removed_as_common += 1
        continue
    filtered_tokens.append(token)

print(f"[AUDIT] Removed as stopword: {removed_as_stopword}")
print(f"[AUDIT] Removed as common word (low frequency ratio): {removed_as_common}")
print(f"[AUDIT] Remaining after filter: {len(filtered_tokens)}")

selected_new_tokens = list(dict.fromkeys(filtered_tokens))[:2000]
print(f"[OK] {len(selected_new_tokens)} final domain-specific tokens selected.")
print("Sample:", selected_new_tokens[:30])

# --- 3. Build a fresh tokenizer and model from the base ---
tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
num_added = tokenizer.add_tokens(selected_new_tokens)
print(f"[OK] Added {num_added} new tokens to tokenizer. Vocab size: {len(tokenizer)}")

model = AutoModelForMaskedLM.from_pretrained(CONFIG["base_model"])
OLD_VOCAB = model.get_input_embeddings().weight.shape[0]
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
print(f"[OK] Embedding rows: {OLD_VOCAB} -> {len(tokenizer)}")
# --- Sanity check: verify new rows are NOT a collapsed/near-identical cluster ---
emb_check = model.get_input_embeddings().weight
old_norms = emb_check[:OLD_VOCAB].norm(dim=1)
new_norms = emb_check[OLD_VOCAB:].norm(dim=1)
print(f"[NORM CHECK] old rows: mean={old_norms.mean().item():.3f}, std={old_norms.std().item():.3f}")
print(f"[NORM CHECK] new rows: mean={new_norms.mean().item():.3f}, std={new_norms.std().item():.3f}")

# --- 4. Sanity check before training ---
def check(tag):
    model.eval()
    text = "The capital of France is <mask>."
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    idx = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero()[0].item()
    probs = torch.softmax(logits[0, idx], dim=-1)
    top = probs.topk(3)
    print(tag, [(tokenizer.decode([i]).strip(), round(p.item(), 3))
                for p, i in zip(top.values, top.indices)])
    model.train()

check("[BEFORE TRAINING]")

# --- 5. Freeze everything, unfreeze only new embedding rows ---
for p in model.parameters():
    p.requires_grad = False
emb_weight = model.get_input_embeddings().weight
emb_weight.requires_grad = True

def zero_old_rows(grad):
    grad = grad.clone()
    grad[:OLD_VOCAB] = 0
    return grad
emb_weight.register_hook(zero_old_rows)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[AUDIT] Trainable tensor size: {trainable:,} (only new rows actually update)")

# --- 6. Build the dataset fresh, with the new tokenizer ---
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus_lines = [line.strip() for line in f if len(line.strip()) > 30][:25000]
raw_dataset = Dataset.from_dict({"text": corpus_lines})

def tokenize_batch(batch, tok):
    return tok(batch["text"], truncation=True, max_length=CONFIG["max_length"], padding="max_length")

tokenized_dataset = raw_dataset.map(
    tokenize_batch, batched=True, num_proc=1,
    remove_columns=["text"], fn_kwargs={"tok": tokenizer}
)
print(f"[OK] Tokenized dataset ready: {len(tokenized_dataset)} samples.")

# --- 7. Baseline before training (on 512 samples) ---
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=CONFIG["mlm_probability"])

training_args = TrainingArguments(
    output_dir="/kaggle/working/phase1_v4_checkpoint",
    num_train_epochs=3,
    per_device_train_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=CONFIG["grad_accum_steps"],
    learning_rate=CONFIG["learning_rate_phase1"],
    weight_decay=0.0,
    logging_steps=10,
    save_strategy="no",
    fp16=False,
    dataloader_num_workers=2,
    report_to="none",
)
from transformers import TrainerCallback

class EmbeddingNormWatcher(TrainerCallback):
    def on_before_optimizer_step(self, args, control, **kwargs):
        model = kwargs["model"]
        grad = model.get_input_embeddings().weight.grad
        if grad is None:
            print("  [GRAD CHECK] grad is None at step boundary!")
        else:
            print(f"  [GRAD CHECK] grad abs mean (new rows) = {grad[OLD_VOCAB:].abs().mean().item():.8f}")
            print(f"  [GRAD CHECK] grad abs mean (old rows) = {grad[:OLD_VOCAB].abs().mean().item():.8f}")

    def on_log(self, args, state, control, **kwargs):
        model = kwargs["model"]
        with torch.no_grad():
            new_norms = model.get_input_embeddings().weight[OLD_VOCAB:].norm(dim=1)
            print(f"  [WATCH] new-row mean norm = {new_norms.mean().item():.4f}, max = {new_norms.max().item():.4f}")


trainer = Trainer(model=model, args=training_args,
                  train_dataset=tokenized_dataset, data_collator=data_collator , callbacks=[EmbeddingNormWatcher()])

print("[BASELINE]", trainer.evaluate(eval_dataset=tokenized_dataset.select(range(512))))

# --- 8. Train ---
trainer.train()

check("[AFTER TRAINING]")

# --- 9. Push model, tokenizer, and dataset ---
model.push_to_hub(CONFIG["model_repo"], private=True)
tokenizer.push_to_hub(CONFIG["model_repo"], private=True)
tokenized_dataset.push_to_hub(CONFIG["dataset_repo"], private=True)
print(f"[SUCCESS] Pushed model+tokenizer to {CONFIG['model_repo']}")
print(f"[SUCCESS] Pushed dataset to {CONFIG['dataset_repo']}")