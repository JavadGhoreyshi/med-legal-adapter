CONFIG = {
    "hf_username": "shokol1011",
    "model_repo": "shokol1011/med-legal-xlm-roberta",
    "dataset_repo": "shokol1011/med-legal-tokenized-dataset",
    "base_model": "xlm-roberta-base",
    "max_length": 128,
    "mlm_probability": 0.15,
    "batch_size": 16,
    "grad_accum_steps": 4,
    # فاز فعلی که روی هاگینگ‌فیس ذخیره شده
    "current_phase": "phase1_warmup",
    "num_layers_to_unfreeze_phase2": 2,
    "learning_rate_phase1": 1e-3,
    "learning_rate_phase2": 2e-5,
}
