"""
Part 2: Domain token extraction quality check.
No model loading, no training — just tests the extraction pipeline.
"""
from tokenizers import Tokenizer, decoders
from transformers import AutoTokenizer
from hazm import stopwords_list
import urllib.request
from collections import Counter
import re
from utils import find_file
from config import CONFIG
def is_digit_like(word):
    # Matches Western digits (0-9), Persian/Arabic-Indic digits (٠-٩), and Arabic digits (۰-۹)
    return bool(re.fullmatch(r"[0-9\u06F0-\u06F9\u0660-\u0669]+", word))
CORPUS_PATH = find_file("bilingual_med_legal_corpus.txt")
TOKENIZER_JSON_PATH = find_file("med_legal_tokenizer.json")
assert CORPUS_PATH is not None, "Corpus file not found!"
assert TOKENIZER_JSON_PATH is not None, "Custom tokenizer file not found!"
  
base_tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
raw_tok = Tokenizer.from_file(TOKENIZER_JSON_PATH)
raw_tok.decoder = decoders.ByteLevel()

def is_already_known(word, tok):
    ids_mid = tok(word, add_special_tokens=False)["input_ids"]
    ids_start = tok(" " + word, add_special_tokens=False)["input_ids"]
    return len(ids_mid) == 1 or len(ids_start) == 1

# --- Step A: decode + dedup against base vocab ---
raw_vocab_size = len(raw_tok.get_vocab())
tokens_to_add = []
for token_str, token_id in raw_tok.get_vocab().items():
    real_text = raw_tok.decode([token_id]).strip()
    if not real_text or len(real_text) < 3:
        continue
    if not is_already_known(real_text, base_tokenizer):
        tokens_to_add.append(real_text)

print(f"[STEP A] Raw custom vocab size: {raw_vocab_size:,}")
print(f"[STEP A] After decode+length+dedup filter: {len(tokens_to_add):,}")

# --- Step B: stopword + frequency ratio filter ---
stopwords_hazm = set(stopwords_list())
url = "https://raw.githubusercontent.com/kharazi/persian-stopwords/master/persian"
urllib.request.urlretrieve(url, "persian_stopwords_github.txt")
with open("persian_stopwords_github.txt", "r", encoding="utf-8") as f:
    stopwords_github = set(line.strip() for line in f if line.strip())
PERSIAN_STOPWORDS = stopwords_hazm | stopwords_github

def word_counts(text_lines):
    counter = Counter()
    for line in text_lines:
        counter.update(re.findall(r"[\w\u0600-\u06FF]+", line.lower()))
    return counter

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    domain_lines = f.readlines()
domain_counts = word_counts(domain_lines)
domain_total = sum(domain_counts.values())

from datasets import load_dataset
wiki = load_dataset("wikimedia/wikipedia", "20231101.fa", split="train", streaming=True)
general_lines = [row["text"] for i, row in enumerate(wiki) if i < 3000]
general_counts = word_counts(general_lines)
general_total = sum(general_counts.values())

RATIO_THRESHOLD = 5.0

def is_domain_specific(word):
    domain_freq = domain_counts.get(word, 0) / domain_total
    general_freq = general_counts.get(word, 0) / general_total
    if general_freq == 0:
        # NOTE: known limitation — this lets through rare English words
        # that simply don't appear in the Persian Wikipedia sample.
        return domain_freq > 0
    return (domain_freq / general_freq) >= RATIO_THRESHOLD

filtered_tokens = []
removed_stopword = 0
removed_common = 0
removed_digit = 0

for token in tokens_to_add:
    if is_digit_like(token):
        removed_digit += 1
        continue
    if token in PERSIAN_STOPWORDS:
        removed_stopword += 1
        continue
    if not is_domain_specific(token):
        removed_common += 1
        continue
    filtered_tokens.append(token)

print(f"[STEP B] Removed as digit-like: {removed_digit}")
print(f"[STEP B] Removed as stopword: {removed_stopword}")
print(f"[STEP B] Removed as common (low domain ratio): {removed_common}")


print("\n--- Sample of final tokens (first 40) ---")
for t in selected[:40]:
    print(" ", t)

# --- Manual quality signal: how many look like real domain terms vs. noise ---
digit_like = sum(1 for t in selected if is_digit_like(t))
latin_tokens = sum(1 for t in selected if re.fullmatch(r"[A-Za-z]+", t))
persian_tokens = sum(1 for t in selected if re.fullmatch(r"[\u0600-\u06FF]+", t))
print(f"\n--- Composition of final token list ---")
print(f"All-digit tokens: {digit_like}")
print(f"All-Latin tokens: {latin_tokens}")
print(f"All-Persian tokens: {persian_tokens}")
print(f"Other/mixed: {len(selected) - digit_like - latin_tokens - persian_tokens}")

print("\n[DONE] Part 2 complete.")