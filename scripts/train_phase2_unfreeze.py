"""
Phase 2: باز کردن N لایه‌ی آخر ترنسفورمر + آموزش با lr پایین‌تر
ورودی: مدل فاز ۱ از هاگینگ‌فیس  |  خروجی: نسخه‌ی جدید روی هاگینگ‌فیس
"""
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                           DataCollatorForLanguageModeling, Trainer, TrainingArguments)
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
from config import CONFIG

login(token=UserSecretsClient().get_secret("HF_TOKEN"))

model = AutoModelForMaskedLM.from_pretrained(CONFIG["model_repo"])
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_repo"])
tokenized_dataset = load_dataset(CONFIG["dataset_repo"], split="train")

# --- انفریز کردن N لایه‌ی آخر ---
for param in model.roberta.parameters():
    param.requires_grad = False
model.get_input_embeddings().weight.requires_grad = True

n = CONFIG["num_layers_to_unfreeze_phase2"]
total_layers = len(model.roberta.encoder.layer)
for layer in model.roberta.encoder.layer[total_layers - n:]:
    for param in layer.parameters():
        param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"[AUDIT] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=CONFIG["mlm_probability"])

training_args = TrainingArguments(
    output_dir="/kaggle/working/phase2_checkpoint",
    num_train_epochs=3,
    per_device_train_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=CONFIG["grad_accum_steps"],
    learning_rate=CONFIG["learning_rate_phase2"],
    weight_decay=0.01, logging_steps=10, save_steps=200, save_total_limit=1,
    fp16=torch.cuda.is_available(), dataloader_num_workers=2, report_to="none")

trainer = Trainer(model=model, args=training_args,
                   train_dataset=tokenized_dataset, data_collator=data_collator)
trainer.train()

model.push_to_hub(CONFIG["model_repo"])
tokenizer.push_to_hub(CONFIG["model_repo"])
print("[SUCCESS] Phase 2 pushed to hub.")
