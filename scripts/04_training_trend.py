"""
Part 4: Training trend check.
Only run this after parts 1-3 look healthy.
Trains for a meaningful number of steps and reports the loss trend in windows,
plus saves a simple trend chart.
"""
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import TrainerCallback, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tokenizers import Tokenizer, decoders
from datasets import Dataset
from hazm import stopwords_list
import urllib.request
from collections import Counter
import re
from utils import find_file
from config import CONFIG
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient

login(token=UserSecretsClient().get_secret("HF_TOKEN"))

CORPUS_PATH = find_file("bilingual_med_legal_corpus.txt")
TOKENIZER_JSON_PATH = find_file("med_legal_tokenizer.json")

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

filtered = [t for t in tokens_to_add if t not in PERSIAN_STOPWORDS and is_domain_specific(t)]
selected_new_tokens = list(dict.fromkeys(filtered))[:2000]

tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
tokenizer.add_tokens(selected_new_tokens)
model = AutoModelForMaskedLM.from_pretrained(CONFIG["base_model"])
OLD_VOCAB = model.get_input_embeddings().weight.shape[0]
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

for p in model.parameters():
    p.requires_grad = False
emb_weight = model.get_input_embeddings().weight
emb_weight.requires_grad = True

def zero_old_rows(grad):
    grad = grad.clone()
    grad[:OLD_VOCAB] = 0
    return grad
emb_weight.register_hook(zero_old_rows)

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus_lines = [l.strip() for l in f if len(l.strip()) > 30][:25000]
raw_dataset = Dataset.from_dict({"text": corpus_lines})

def tokenize_batch(batch, tok):
    return tok(batch["text"], truncation=True, max_length=CONFIG["max_length"], padding="max_length")

tokenized_dataset = raw_dataset.map(tokenize_batch, batched=True, num_proc=1,
                                     remove_columns=["text"], fn_kwargs={"tok": tokenizer})

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=CONFIG["mlm_probability"])

# --- Collect loss history for plotting ---
loss_history = []

class LossCollector(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            loss_history.append((state.global_step, logs["loss"]))

training_args = TrainingArguments(
    output_dir="/kaggle/working/trend_check",
    max_steps=300,                      # fixed step budget for this check, not full 3 epochs
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

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset,
                  data_collator=data_collator, callbacks=[LossCollector()])

print("[BASELINE]", trainer.evaluate(eval_dataset=tokenized_dataset.select(range(512))))

trainer.train()

# --- Report windowed averages ---
print("\n--- Loss in windows of 5 logs (~50 steps each) ---")
window = 5
for i in range(0, len(loss_history), window):
    chunk = loss_history[i:i+window]
    avg = sum(v for _, v in chunk) / len(chunk)
    step_range = f"{chunk[0][0]}-{chunk[-1][0]}"
    print(f"  steps {step_range}: avg loss = {avg:.3f}")

# --- Save a simple trend chart ---
steps = [s for s, _ in loss_history]
losses = [v for _, v in loss_history]
plt.figure(figsize=(8, 4))
plt.plot(steps, losses, marker="o", markersize=3)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Phase 1 training loss trend")
plt.grid(True, alpha=0.3)
plt.savefig("/kaggle/working/loss_trend.png", dpi=120, bbox_inches="tight")
print("\n[OK] Chart saved to /kaggle/working/loss_trend.png")

print("\n[DONE] Part 4 complete. If the trend is flat or rising, stop here and investigate further.")