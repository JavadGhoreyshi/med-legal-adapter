"""
Part 1: Raw data quality check.
No model, no tokenizer training — just inspects the raw corpus.
"""
import re
from collections import Counter
from utils import find_file

CORPUS_PATH = find_file("bilingual_med_legal_corpus.txt")
assert CORPUS_PATH is not None, "Corpus file not found!"
print(f"[OK] Corpus: {CORPUS_PATH}")

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()
# Where did the old "first 25,000 usable lines" training set come from?
kept_idx = [i for i, l in enumerate(lines) if len(l.strip()) > 30]
print(f"[SPLIT CHECK] index of the 25,000th kept line: {kept_idx[24999]}")
print("[SPLIT CHECK] the medical part is the first ~58,190 lines; "
      "if this index is below that, the old training set was medical-only")
print(f"[OK] Total lines: {len(lines):,}")

# --- Line length distribution ---
lengths = [len(l.strip()) for l in lines]
lengths_sorted = sorted(lengths)
n = len(lengths_sorted)
print("\n--- Line length distribution (characters) ---")
print(f"min: {lengths_sorted[0]}")
print(f"p25: {lengths_sorted[n//4]}")
print(f"median: {lengths_sorted[n//2]}")
print(f"p75: {lengths_sorted[3*n//4]}")
print(f"p95: {lengths_sorted[int(n*0.95)]}")
print(f"max: {lengths_sorted[-1]}")

very_short = sum(1 for l in lengths if l < 10)
very_long = sum(1 for l in lengths if l > 1000)
print(f"lines < 10 chars: {very_short}")
print(f"lines > 1000 chars: {very_long}")

# --- Latin vs Persian character ratio ---
def char_ratios(text):
    persian = len(re.findall(r"[\u0600-\u06FF]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return persian, latin

total_persian, total_latin = 0, 0
for l in lines:
    p, lat = char_ratios(l)
    total_persian += p
    total_latin += lat

print("\n--- Script composition ---")
print(f"Persian characters: {total_persian:,}")
print(f"Latin characters:   {total_latin:,}")
print(f"Latin/Persian ratio: {total_latin / max(total_persian,1):.3f}")

# --- Suspicious lines: unusual Latin diacritics (possible encoding issues) ---
suspicious_pattern = re.compile(r'[À-ž]')
suspicious_lines = [l for l in lines if suspicious_pattern.search(l)]
print(f"\n--- Suspicious lines (unusual diacritics) ---")
print(f"Count: {len(suspicious_lines)} / {len(lines)} ({100*len(suspicious_lines)/len(lines):.3f}%)")
for l in suspicious_lines[:5]:
    print(" ", repr(l.strip()[:120]))

# --- Most common words overall (sanity check for colloquial pollution) ---
def tokenize_words(text):
    return re.findall(r"[\w\u0600-\u06FF]+", text.lower())

counter = Counter()
for l in lines:
    counter.update(tokenize_words(l))

print("\n--- Top 30 most frequent words (sanity check) ---")
for word, count in counter.most_common(30):
    print(f"  {word}: {count}")

print("\n[DONE] Part 1 complete.")
