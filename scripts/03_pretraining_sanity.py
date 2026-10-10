"""
Part 3: compare starting points on the SAME validation set, before any training.
Models: raw base | expanded (random init) | expanded (HF mean init) |
        expanded (sub-token-mean init) | v1 from the Hub.
"""
import gc
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from config import CONFIG
from utils import (find_file, hf_login, load_split, tokenize_lines, select_domain_tokens,
                   build_expanded_model, eval_loss, fill_mask_check)

hf_login()

CORPUS_PATH = find_file("bilingual_med_legal_corpus.txt")
TOKENIZER_JSON_PATH = find_file("med_legal_tokenizer.json")
assert CORPUS_PATH is not None and TOKENIZER_JSON_PATH is not None, "Input files not found!"

base_tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
train_lines, val_lines = load_split(CORPUS_PATH, CONFIG["n_train"], CONFIG["n_val"], CONFIG["seed"])
selected = select_domain_tokens(TOKENIZER_JSON_PATH, base_tokenizer, train_lines, CONFIG)

results = {}
max_len, mlm_p = CONFIG["max_length"], CONFIG["mlm_probability"]


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- 1. Raw base model ---
model = AutoModelForMaskedLM.from_pretrained(CONFIG["base_model"])
fill_mask_check(model, base_tokenizer, "[raw base]")
val_ds = tokenize_lines(val_lines, base_tokenizer, max_len)
results["raw base"] = eval_loss(model, base_tokenizer, val_ds, mlm_p)
del model
cleanup()

# --- 2. Expanded model with each init method ---
for init in ["random", "hf_mean", "subtoken_mean"]:
    model, tok, old_vocab = build_expanded_model(CONFIG["base_model"], base_tokenizer, selected, init=init)
    emb = model.get_input_embeddings().weight
    new_norms = emb[old_vocab:].norm(dim=1)
    print(f"[{init}] old-row norm mean={emb[:old_vocab].norm(dim=1).mean().item():.3f} | "
          f"new-row norm mean={new_norms.mean().item():.3f}, std={new_norms.std().item():.3f}")
    fill_mask_check(model, tok, f"[{init}]")
    val_ds = tokenize_lines(val_lines, tok, max_len)
    results[f"expanded / {init}"] = eval_loss(model, tok, val_ds, mlm_p)
    del model
    cleanup()

# --- 3. v1 from the Hub (private repo, needs login) ---
try:
    tok = AutoTokenizer.from_pretrained(CONFIG["legacy_repo"])
    model = AutoModelForMaskedLM.from_pretrained(CONFIG["legacy_repo"])
    val_ds = tokenize_lines(val_lines, tok, max_len)
    results["v1 (legacy repo, trained on old split)"] = eval_loss(model, tok, val_ds, mlm_p)
    del model
    cleanup()
except Exception as e:
    print("[WARN] could not evaluate v1:", e)

print("\n=== Validation loss summary ===")
for name, value in results.items():
    print(f"{name:45s} {value:.3f}")

print("\n[NOTES]")
print("- Compare the three 'expanded' rows with each other: same tokenizer, fair comparison.")
print("- 'raw base' vs 'expanded' is NOT exactly apples-to-apples: the tokenizations differ, "
      "so the models predict slightly different units.")
print("- v1 was trained on lines that may overlap this val set, so its number is optimistic.")
print("[DONE] Part 3 complete.")
