"""
Part 3: Pre-training sanity check.
Builds the model + expanded tokenizer, but does NOT train.
Checks whether the starting point is healthy before spending GPU time.
"""
import torch
from tokenizers import Tokenizer, decoders
from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from hazm import stopwords_list
import urllib.request
from collections import Counter
import re
from utils import find_file
from config import CONFIG
def is_digit_like(word):
    # Matches Western digits (0-9), Persian/Arabic-Indic digits (٠-٩), and Arabic digits (۰-۹)
    return bool(re.fullmatch(r"[0-9\u06F0-\u06F9\u0660-\u0669]+", word))
CORPUS_PATH = find_file("bilingual_med_legal_corpus.txt")
TOKENIZER_JSON_PATH = find_file("med_legal_tokenizer.json")


# --- Rebuild the same token list as part 2 (kept short here; assumes part 2 passed) ---
base_tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
raw_tok = Tokenizer.from_file(TOKENIZER_JSON_PATH)
raw_tok.decoder = decoders.ByteLevel()

def is_already_known(word, tok):
    ids_mid = tok(word, add_special_tokens=False)["input_ids"]
    ids_start = tok(" " + word, add_special_tokens=False)["input_ids"]
    return len(ids_mid) == 1 or len(ids_start) == 1

tokens_to_add = []
for token_str, token_id in raw_tok.get_vocab().items():
    real_text = raw_tok.decode([token_id]).strip()
    if real_text and len(real_text) >= 3 and not is_already_known(real_text, base_tokenizer):
        tokens_to_add.append(real_text)

stopwords_hazm = set(stopwords_list())
url = "https://raw.githubusercontent.com/kharazi/persian-stopwords/master/persian"
urllib.request.urlretrieve(url, "persian_stopwords_github.txt")
with open("persian_stopwords_github.txt", "r", encoding="utf-8") as f:
    stopwords_github = set(line.strip() for line in f if line.strip())
PERSIAN_STOPWORDS = stopwords_hazm | stopwords_github

def word_counts(text_lines):
    c = Counter()
    for l in text_lines:
        c.update(re.findall(r"[\w\u0600-\u06FF]+", l.lower()))
    return c

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    domain_lines = f.readlines()
domain_counts = word_counts(domain_lines)
domain_total = sum(domain_counts.values())

from datasets import load_dataset
wiki = load_dataset("wikimedia/wikipedia", "20231101.fa", split="train", streaming=True)
general_lines = [row["text"] for i, row in enumerate(wiki) if i < 3000]
general_counts = word_counts(general_lines)
general_total = sum(general_counts.values())

def is_domain_specific(word):
    df = domain_counts.get(word, 0) / domain_total
    gf = general_counts.get(word, 0) / general_total
    return (df / gf >= 5.0) if gf > 0 else df > 0

filtered = [t for t in tokens_to_add
            if not is_digit_like(t) and t not in PERSIAN_STOPWORDS and is_domain_specific(t)]
selected_new_tokens = list(dict.fromkeys(filtered))[:2000]
print(f"[OK] {len(selected_new_tokens)} tokens ready for injection.")

# --- Build tokenizer + model, no mean-resizing ---
tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
tokenizer.add_tokens(selected_new_tokens)

model = AutoModelForMaskedLM.from_pretrained(CONFIG["base_model"])
OLD_VOCAB = model.get_input_embeddings().weight.shape[0]
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
print(f"[OK] Embedding rows: {OLD_VOCAB} -> {len(tokenizer)}")

# --- Norm check ---
emb = model.get_input_embeddings().weight
old_norms = emb[:OLD_VOCAB].norm(dim=1)
new_norms = emb[OLD_VOCAB:].norm(dim=1)
print(f"[NORM CHECK] old: mean={old_norms.mean().item():.3f}, std={old_norms.std().item():.3f}")
print(f"[NORM CHECK] new: mean={new_norms.mean().item():.3f}, std={new_norms.std().item():.3f}")
print("  -> Healthy new-row std should be clearly non-zero and NOT a near-identical cluster.")

# --- fill-mask sanity (no training yet) ---
def check(tag):
    model.eval()
    text = "The capital of France is <mask>."
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    idx = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero()[0].item()
    probs = torch.softmax(logits[0, idx], dim=-1)
    top = probs.topk(3)
    print(tag, [(tokenizer.decode([i]).strip(), round(p.item(), 3)) for p, i in zip(top.values, top.indices)])

check("[UNTRAINED MODEL]")

# --- eval_loss baseline on a random domain sample ---
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus_lines = [l.strip() for l in f if len(l.strip()) > 30][:25000]
raw_dataset = Dataset.from_dict({"text": corpus_lines})

def tokenize_batch(batch, tok):
    return tok(batch["text"], truncation=True, max_length=CONFIG["max_length"], padding="max_length")

tokenized_dataset = raw_dataset.map(tokenize_batch, batched=True, num_proc=1,
                                     remove_columns=["text"], fn_kwargs={"tok": tokenizer})

import random
random.seed(42)
idxs = random.sample(range(len(tokenized_dataset)), 512)
subset = tokenized_dataset.select(idxs)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=CONFIG["mlm_probability"])
args = TrainingArguments(output_dir="/kaggle/working/sanity_v3", per_device_eval_batch_size=16, report_to="none")
trainer = Trainer(model=model, args=args, data_collator=data_collator, eval_dataset=subset)

print("[EVAL_LOSS on expanded model]", trainer.evaluate())

# --- Compare against a fully raw model with NO new tokens, same data ---
raw_model = AutoModelForMaskedLM.from_pretrained(CONFIG["base_model"])
raw_tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
raw_tokenized = Dataset.from_dict({"text": corpus_lines}).map(
    tokenize_batch, batched=True, num_proc=1, remove_columns=["text"], fn_kwargs={"tok": raw_tokenizer})
raw_subset = raw_tokenized.select(idxs)
raw_collator = DataCollatorForLanguageModeling(tokenizer=raw_tokenizer, mlm=True, mlm_probability=CONFIG["mlm_probability"])
raw_trainer = Trainer(model=raw_model, args=args, data_collator=raw_collator, eval_dataset=raw_subset)
print("[EVAL_LOSS on raw base model, no new tokens]", raw_trainer.evaluate())

print("\n[DONE] Part 3 complete. Compare the two eval_loss values above.")