import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
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
print(f"[OK] init: {CONFIG['init_method']} | new rows: {len(tokenizer) - OLD_VOCAB}")

freeze_all_but_new_rows(model, OLD_VOCAB)
old_rows_snapshot = model.get_input_embeddings().weight[:OLD_VOCAB].detach().clone()

train_ds = tokenize_lines(train_lines, tokenizer, CONFIG["max_length"])
val_ds = tokenize_lines(val_lines, tokenizer, CONFIG["max_length"])
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=CONFIG["mlm_probability"])

args = TrainingArguments(
    output_dir="/kaggle/working/phase1",
    num_train_epochs=1,
    per_device_train_batch_size=CONFIG["batch_size"],
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,       # same as the tested run in part 4
    learning_rate=CONFIG["learning_rate_phase1"],
    warmup_steps=50,
    weight_decay=0.0,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=300,
    save_strategy="no",
    fp16=False,
    dataloader_num_workers=2,
    seed=42,
    report_to="none",
)

trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                  eval_dataset=val_ds, data_collator=collator)

trainer.train()
final_loss = trainer.evaluate()["eval_loss"]
print(f"[FINAL] val loss: {final_loss:.3f}")

# integrity check before uploading
emb_after = model.get_input_embeddings().weight.detach().cpu()
ok = torch.equal(emb_after[:OLD_VOCAB], old_rows_snapshot.cpu())
print(f"[INTEGRITY] old rows unchanged: {ok}")
assert ok, "Old rows changed! Not uploading."

# local backup (may be deleted by Kaggle), then upload to Hugging Face
model.save_pretrained("/kaggle/working/phase1_model")
tokenizer.save_pretrained("/kaggle/working/phase1_model")

repo = CONFIG["model_repo"]
model.push_to_hub(repo, commit_message="phase1 warmup, subtoken_mean init, 1 epoch")
tokenizer.push_to_hub(repo, commit_message="tokenizer for phase1")
print(f"[OK] uploaded to https://huggingface.co/{repo}")
print("[DONE] Part 5 complete.")
