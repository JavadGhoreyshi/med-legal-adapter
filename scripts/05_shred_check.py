"""
Shred check: how many ordinary words get split differently once the new tokens are added?
"""
from transformers import AutoTokenizer
from config import CONFIG
from utils import find_file, load_split, select_domain_tokens

base = AutoTokenizer.from_pretrained(CONFIG["base_model"])
train_lines, _ = load_split(find_file("bilingual_med_legal_corpus.txt"),
                            CONFIG["n_train"], CONFIG["n_val"], CONFIG["seed"])
sel = select_domain_tokens(find_file("med_legal_tokenizer.json"), base,
                           train_lines, CONFIG, verbose=False)
sel_set = set(sel)

tok = AutoTokenizer.from_pretrained(CONFIG["base_model"])
tok.add_tokens(sel)

changed = set()
checked = set()
for line in train_lines[:2000]:
    for w in line.split():
        if w in checked:
            continue
        checked.add(w)
        if w not in sel_set and tok.tokenize(w) != base.tokenize(w):
            changed.add(w)

print(f"[SHRED CHECK] distinct words checked: {len(checked):,}")
print(f"[SHRED CHECK] words (excluding the added tokens) tokenized differently: {len(changed):,}")
for w in list(changed)[:30]:
    print(f"  {w}: base={base.tokenize(w)} | added={tok.tokenize(w)}")
