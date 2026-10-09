CONFIG = {
    "hf_username": "JavadGhoreyshi",
    "model_repo": "JavadGhoreyshi/med-legal-xlm-roberta-v4",
    "legacy_repo": "JavadGhoreyshi/med-legal-xlm-roberta",
    "broken_v3_repo": "JavadGhoreyshi/med-legal-xlm-roberta-v3",
    "dataset_repo": "JavadGhoreyshi/med-legal-tokenized-dataset-v4",
    "base_model": "xlm-roberta-base",
    "max_length": 128,
    "mlm_probability": 0.15,
    "batch_size": 16,
    "grad_accum_steps": 4,
    # فاز فعلی که روی هاگینگ‌فیس ذخیره شده
    "current_phase": "phase1_warmup",
    "num_layers_to_unfreeze_phase2": 2,
    "learning_rate_phase1": 1e-5,
    "learning_rate_phase2": 2e-5,
}
