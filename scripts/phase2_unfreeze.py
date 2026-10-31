import torch
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
from transformers import (AutoTokenizer, AutoModelForMaskedLM, Trainer,
                          TrainingArguments, DataCollatorForLanguageModeling)
from config import CONFIG
from utils import find_file, load_split, tokenize_lines

login(token=UserSecretsClient().get_secret("HF_TOKEN"))

repo1 = CONFIG["model_repo"]
repo2 = repo1 + "-phase2"

tok = AutoTokenizer.from_pretrained(repo1)
OLD = len(AutoTokenizer.from_pretrained(CONFIG["base_model"]))
model = AutoModelForMaskedLM.from_pretrained(repo1)

# 1) همه چیز فریز
for p in model.parameters():
    p.requires_grad = False

# 2) embedding (فقط ردیف‌های جدید)
emb = model.get_input_embeddings().weight
emb.requires_grad = True
emb_mask = torch.zeros(emb.shape[0], 1)
emb_mask[OLD:] = 1.0
emb.register_hook(lambda g: g * emb_mask.to(g.device, g.dtype))

# 3) bias خروجی (فقط ردیف‌های جدید)
bias = model.lm_head.bias
bias.requires_grad = True
bias_mask = torch.zeros(bias.shape[0])
bias_mask[OLD:] = 1.0
bias.register_hook(lambda g: g * bias_mask.to(g.device, g.dtype))
print("bias mean old/new:", bias[:OLD].mean().item(), bias[OLD:].mean().item())

# 4) لایه‌های آخر
n = CONFIG["num_layers_to_unfreeze_phase2"]
layer_params = []
for layer in model.roberta.encoder.layer[-n:]:
    for p in layer.parameters():
        p.requires_grad = True
        layer_params.append(p)

old_emb = emb[:OLD].detach().clone().cpu()
old_bias = bias[:OLD].detach().clone().cpu()

optimizer = torch.optim.AdamW([
    {"params": [emb, bias], "lr": 1e-4},
    {"params": layer_params, "lr": CONFIG["learning_rate_phase2"]},
], weight_decay=0.0)

# 5) داده
CORPUS_PATH = find_file("bilingual_med_legal_corpus.txt")
train_lines, val_lines = load_split(CORPUS_PATH, CONFIG["n_train"], CONFIG["n_val"], CONFIG["seed"])
train_ds = tokenize_lines(train_lines, tok, CONFIG["max_length"])
val_ds = tokenize_lines(val_lines, tok, CONFIG["max_length"])
collator = DataCollatorForLanguageModeling(
    tokenizer=tok, mlm=True, mlm_probability=CONFIG["mlm_probability"])

args = TrainingArguments(
    output_dir="/kaggle/working/phase2",
    num_train_epochs=1,
    per_device_train_batch_size=CONFIG["batch_size"],
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    warmup_steps=50,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="no",
    fp16=False,
    dataloader_num_workers=2,
    seed=42,
    report_to="none",
)

trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                  eval_dataset=val_ds, data_collator=collator,
                  optimizers=(optimizer, None))

print("[BASELINE] val loss before phase 2:", round(trainer.evaluate()["eval_loss"], 3))
trainer.train()
print("[FINAL] val loss after phase 2:", round(trainer.evaluate()["eval_loss"], 3))

# 6) چک سلامت و ذخیره
ok_emb = torch.equal(model.get_input_embeddings().weight[:OLD].detach().cpu(), old_emb)
ok_bias = torch.equal(model.lm_head.bias[:OLD].detach().cpu(), old_bias)
print("[INTEGRITY] old embedding rows unchanged:", ok_emb, "| old bias unchanged:", ok_bias)
assert ok_emb and ok_bias, "Old rows changed! Not saving."

model.save_pretrained("/kaggle/working/phase2_model")
tok.save_pretrained("/kaggle/working/phase2_model")
try:
    model.push_to_hub(repo2, commit_message="phase2: last 2 layers + new rows + new bias")
    tok.push_to_hub(repo2, commit_message="tokenizer for phase2")
    print(f"[OK] uploaded to https://huggingface.co/{repo2}")
except Exception as e:
    print("upload failed, model stays in /kaggle/working/phase2_model:", e)
