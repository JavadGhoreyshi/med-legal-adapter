"""
Part 2: domain token selection. No model is loaded here.
Frequencies come from the TRAIN split only.
"""
import re
from transformers import AutoTokenizer
from config import CONFIG
from utils import find_file, load_split, select_domain_tokens, word_counts

CORPUS_PATH = find_file("bilingual_med_legal_corpus.txt")
TOKENIZER_JSON_PATH = find_file("med_legal_tokenizer.json")
assert CORPUS_PATH is not None, "Corpus file not found!"
assert TOKENIZER_JSON_PATH is not None, "Custom tokenizer file not found!"

base_tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
train_lines, val_lines = load_split(CORPUS_PATH, CONFIG["n_train"], CONFIG["n_val"], CONFIG["seed"])
print(f"[OK] train lines: {len(train_lines):,} | val lines: {len(val_lines):,}")

selected = select_domain_tokens(TOKENIZER_JSON_PATH, base_tokenizer, train_lines, CONFIG)

print("\n--- First 60 selected tokens (most frequent in train first) ---")
print(selected[:60])

# --- Composition ---
latin = sum(1 for t in selected if re.fullmatch(r"[A-Za-z]+", t))
persian = sum(1 for t in selected if re.fullmatch(r"[\u0600-\u06FF\u200c]+", t))
print(f"\n[COMPOSITION] Latin-only: {latin} | Persian-only: {persian} | "
      f"other: {len(selected) - latin - persian}")

# --- Coverage: do the selected tokens actually occur in train and in val? ---
train_counts = word_counts(train_lines)
val_counts = word_counts(val_lines)
train_total = sum(train_counts.values())
val_total = sum(val_counts.values())

in_train = sum(train_counts.get(t, 0) for t in selected)
in_val = sum(val_counts.get(t, 0) for t in selected)
val_seen = sum(1 for t in selected if val_counts.get(t, 0) > 0)

print(f"[COVERAGE] selected tokens cover {100 * in_train / train_total:.2f}% of train words")
print(f"[COVERAGE] selected tokens cover {100 * in_val / val_total:.2f}% of val words")
print(f"[COVERAGE] tokens appearing at least once in val: {val_seen}/{len(selected)}")
print("\n[NOTE] Colloquial words (e.g. forum-style verbs) can still pass the frequency "
      "filter, because the Wikipedia sample is formal text. Review the list above by eye.")
print("[DONE] Part 2 complete.")
