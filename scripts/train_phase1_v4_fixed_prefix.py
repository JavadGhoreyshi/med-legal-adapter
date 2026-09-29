"""
Phase 1 v3: Fixed vocabulary extraction + safe embedding warm-up.

Root cause fixed here: the custom tokenizer used a ByteLevel pre-tokenizer,
so its raw vocab keys are byte-encoded garbage (e.g. "Ø¯Ø±Ø®ÙĪØ§Ø³Øª"),
not real UTF-8 text. Those garbage strings were being added to xlm-roberta's
vocabulary, poisoning the shared input/output embedding matrix and making
eval_loss jump from ~3.85 (healthy) to ~21 (broken).

This script decodes each custom token back to real text before comparing it
against the base vocabulary and adding it.
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

login(token=UserSecretsClient().get_secret("HF_TOKEN"))

# --- ۱. پیدا کردن فایل‌ها ---
CORPUS_PATH = find_file("bilingual_med_legal_corpus.txt")
TOKENIZER_JSON_PATH = find_file("med_legal_tokenizer.json")
assert CORPUS_PATH is not None, "فایل پیکره پیدا نشد!"
assert TOKENIZER_JSON_PATH is not None, "فایل توکنایزر تخصصی پیدا نشد!"
print(f"[OK] Corpus: {CORPUS_PATH}")
print(f"[OK] Custom tokenizer: {TOKENIZER_JSON_PATH}")
# --- ۲. استخراج صحیح توکن‌های تخصصی ---
base_tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])

raw_tok = Tokenizer.from_file(TOKENIZER_JSON_PATH)
raw_tok.decoder = decoders.ByteLevel()

def is_already_known(word, tok):
    # چک می‌کند آیا این کلمه از قبل به‌صورت یک توکن واحد در توکنایزر پایه وجود دارد
    # هم به‌شکل وسط کلمه، هم به‌شکل شروع کلمه (با فاصله، که SentencePiece آن را با ▁ ذخیره می‌کند)
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

selected_new_tokens = list(set(tokens_to_add))[:2000]
print(f"[OK] {len(selected_new_tokens)} real, truly-new domain tokens extracted.")
print("Sample:", selected_new_tokens[:15])

# --- ۳. ساخت توکنایزر و مدل جدید از پایه ---
tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
num_added = tokenizer.add_tokens(selected_new_tokens)
print(f"[OK] Added {num_added} new tokens to tokenizer. Vocab size: {len(tokenizer)}")

model = AutoModelForMaskedLM.from_pretrained(CONFIG["base_model"])
OLD_VOCAB = model.get_input_embeddings().weight.shape[0]
model.resize_token_embeddings(len(tokenizer))
print(f"[OK] Embedding rows: {OLD_VOCAB} -> {len(tokenizer)}")

# --- ۴. تست سلامت قبل از آموزش ---
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

# --- ۵. فریز کردن همه‌چیز، آزاد فقط ردیف‌های جدید embedding ---
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

# --- ۶. ساخت دیتاست از نو، با توکنایزر جدید ---
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

# --- ۷. baseline قبل از آموزش (روی ۵۱۲ نمونه) ---
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=CONFIG["mlm_probability"])

training_args = TrainingArguments(
    output_dir="/kaggle/working/phase1_v3_checkpoint",
    num_train_epochs=3,
    per_device_train_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=CONFIG["grad_accum_steps"],
    learning_rate=CONFIG["learning_rate_phase1"],
    weight_decay=0.0,
    logging_steps=10,
    save_strategy="no",
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=2,
    report_to="none",
)

trainer = Trainer(model=model, args=training_args,
                  train_dataset=tokenized_dataset, data_collator=data_collator)

print("[BASELINE]", trainer.evaluate(eval_dataset=tokenized_dataset.select(range(512))))

# --- ۸. آموزش ---
trainer.train()

check("[AFTER TRAINING]")

# --- ۹. آپلود مدل، توکنایزر و دیتاست جدید ---
model.push_to_hub(CONFIG["model_repo"], private=True)
tokenizer.push_to_hub(CONFIG["model_repo"], private=True)
tokenized_dataset.push_to_hub(CONFIG["dataset_repo"], private=True)
print(f"[SUCCESS] Pushed model+tokenizer to {CONFIG['model_repo']}")
print(f"[SUCCESS] Pushed dataset to {CONFIG['dataset_repo']}")
