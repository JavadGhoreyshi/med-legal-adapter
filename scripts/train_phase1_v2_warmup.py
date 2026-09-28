"""
Phase 1 v2: safer embedding warm-up.
- Start from the ORIGINAL xlm-roberta-base weights (not the damaged v1).
- Reuse the v1 tokenizer, so the tokenized dataset ids stay valid.
- New token rows are initialised from the mean of their base sub-token embeddings.
- Only the NEW embedding rows are trained; old rows and lm_head stay untouched.
"""
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
from config import CONFIG

login(token=UserSecretsClient().get_secret("HF_TOKEN"))

# --- tokenizers and model ---
base_tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
tokenizer = AutoTokenizer.from_pretrained(CONFIG["legacy_repo"])   # base + 2000 new tokens
model = AutoModelForMaskedLM.from_pretrained(CONFIG["base_model"])  # fresh, undamaged

OLD_VOCAB = model.get_input_embeddings().weight.shape[0]
assert OLD_VOCAB == len(base_tokenizer), "vocab size mismatch"
model.resize_token_embeddings(len(tokenizer))
print(f"[OK] Embedding rows: {OLD_VOCAB} -> {len(tokenizer)}")

# --- smart init: mean of base sub-token embeddings ---
emb = model.get_input_embeddings().weight
with torch.no_grad():
    for tok, tid in tokenizer.get_added_vocab().items():
        if tid < OLD_VOCAB:
            continue
        pieces = base_tokenizer(tok, add_special_tokens=False)["input_ids"]
        if pieces:
            emb[tid] = emb[pieces].mean(dim=0)

# --- sanity check helper ---
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

# --- freeze everything, train only the embedding matrix ---
for p in model.parameters():
    p.requires_grad = False
emb_weight = model.get_input_embeddings().weight
emb_weight.requires_grad = True

def zero_old_rows(grad):
    grad = grad.clone()
    grad[:OLD_VOCAB] = 0      # old tokens never change
    return grad
emb_weight.register_hook(zero_old_rows)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[AUDIT] Trainable tensor size: {trainable:,} (only new rows actually update)")

# --- data ---
tokenized_dataset = load_dataset(CONFIG["dataset_repo"], split="train")
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=CONFIG["mlm_probability"])

training_args = TrainingArguments(
    output_dir="/kaggle/working/phase1_v2_checkpoint",
    num_train_epochs=3,
    per_device_train_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=CONFIG["grad_accum_steps"],
    learning_rate=CONFIG["learning_rate_phase1"],
    weight_decay=0.0,          # decay would slowly shrink the "frozen" old rows
    logging_steps=10,
    save_strategy="no",
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=2,
    report_to="none",
)

trainer = Trainer(model=model, args=training_args,
                  train_dataset=tokenized_dataset, data_collator=data_collator)
trainer.train()

check("[AFTER TRAINING]")

model.push_to_hub(CONFIG["model_repo"], private=True)
tokenizer.push_to_hub(CONFIG["model_repo"], private=True)
print("[SUCCESS] Pushed to", CONFIG["model_repo"])
