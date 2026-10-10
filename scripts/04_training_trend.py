"""
Part 4: short training run with validation loss tracked during training.
Run this only after part 3 has told you which init method to use.
"""
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import (AutoTokenizer, Trainer, TrainingArguments,
                          DataCollatorForLanguageModeling)
from config import CONFIG
from utils import (find_file, hf_login, load_split, tokenize_lines, select_domain_tokens,
                   build_expanded_model, freeze_all_but_new_rows)

hf_login()

CORPUS_PATH = find_file("bilingual_med_legal_corpus.txt")
TOKENIZER_JSON_PATH = find_file("med_legal_tokenizer.json")
assert CORPUS_PATH is not None and TOKENIZER_JSON_PATH is not None, "Input files not found!"

base_tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
train_lines, val_lines = load_split(CORPUS_PATH, CONFIG["n_train"], CONFIG["n_val"], CONFIG["seed"])
selected = select_domain_tokens(TOKENIZER_JSON_PATH, base_tokenizer, train_lines, CONFIG)

model, tokenizer, OLD_VOCAB = build_expanded_model(
    CONFIG["base_model"], base_tokenizer, selected, init=CONFIG["init_method"])
print(f"[OK] init method: {CONFIG['init_method']} | new rows: {len(tokenizer) - OLD_VOCAB}")

freeze_all_but_new_rows(model, OLD_VOCAB)
emb = model.get_input_embeddings().weight
old_rows_snapshot = emb[:OLD_VOCAB].detach().clone()
init_new_rows = emb[OLD_VOCAB:].detach().clone()

train_ds = tokenize_lines(train_lines, tokenizer, CONFIG["max_length"])
val_ds = tokenize_lines(val_lines, tokenizer, CONFIG["max_length"])
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=CONFIG["mlm_probability"])

args = TrainingArguments(
    output_dir="/kaggle/working/trend_check",
    max_steps=CONFIG["max_steps_trend"],
    per_device_train_batch_size=CONFIG["batch_size"],
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,       # keeps the logged train loss easy to interpret
    learning_rate=CONFIG["learning_rate_phase1"],
    weight_decay=0.0,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="no",
    fp16=False,
    dataloader_num_workers=2,
    seed=42,
    report_to="none",
)

trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                  eval_dataset=val_ds, data_collator=collator)

baseline = trainer.evaluate()["eval_loss"]
print(f"[BASELINE] val loss before training: {baseline:.3f}")

trainer.train()

# --- Report ---
log = trainer.state.log_history
train_points = [(e["step"], e["loss"]) for e in log if "loss" in e]
eval_points = [(e["step"], e["eval_loss"]) for e in log if "eval_loss" in e]

print("\n--- Validation loss over training ---")
for step, value in eval_points:
    print(f"  step {step:4d}: val loss = {value:.3f}")

# --- Did the new rows actually move, and did the old rows stay untouched? ---
emb_after = model.get_input_embeddings().weight.detach().cpu()
delta = (emb_after[OLD_VOCAB:] - init_new_rows.cpu()).norm(dim=1)
print(f"\n[MOVEMENT] new rows: mean change in norm = {delta.mean().item():.4f}, "
      f"max = {delta.max().item():.4f}")
print(f"[INTEGRITY] old rows unchanged: {torch.equal(emb_after[:OLD_VOCAB], old_rows_snapshot.cpu())}")

# --- Chart ---
plt.figure(figsize=(8, 4))
plt.plot(*zip(*train_points), label="train loss", alpha=0.6)
plt.plot(*zip(*eval_points), marker="o", label="val loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title(f"Phase 1 warm-up (init={CONFIG['init_method']}, lr={CONFIG['learning_rate_phase1']})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("/kaggle/working/loss_trend.png", dpi=120, bbox_inches="tight")
print("[OK] Chart saved to /kaggle/working/loss_trend.png")
print("[DONE] Part 4 complete.")
