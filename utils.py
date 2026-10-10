import os
import re
import json
import random
import urllib.request
from collections import Counter

import torch


def find_file(filename, search_root="/kaggle/input"):
    for root, _, files in os.walk(search_root):
        if filename in files:
            return os.path.join(root, filename)
    if os.path.exists(filename):
        return filename
    return None


def hf_login():
    # Login with the Kaggle secret; skip silently if not on Kaggle
    try:
        from kaggle_secrets import UserSecretsClient
        from huggingface_hub import login
        login(token=UserSecretsClient().get_secret("HF_TOKEN"))
        print("[OK] Logged in to Hugging Face.")
    except Exception as e:
        print("[WARN] HF login skipped:", e)


def load_split(corpus_path, n_train=25000, n_val=2000, seed=42):
    # Shuffle all usable lines once (fixed seed) so that train and val
    # both contain medical AND legal text, and val is never used for training.
    with open(corpus_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if len(l.strip()) > 30]
    random.Random(seed).shuffle(lines)
    val_lines = lines[:n_val]
    train_lines = lines[n_val:n_val + n_train]
    return train_lines, val_lines


def _tok_batch(batch, tok, max_length):
    return tok(batch["text"], truncation=True, max_length=max_length, padding="max_length")


def tokenize_lines(lines, tokenizer, max_length):
    from datasets import Dataset
    ds = Dataset.from_dict({"text": lines})
    return ds.map(_tok_batch, batched=True, remove_columns=["text"],
                  fn_kwargs={"tok": tokenizer, "max_length": max_length})


def word_counts(lines):
    # Case-sensitive on purpose: add_tokens matches the exact surface form.
    # \u200c (ZWNJ) is included so words with a half-space are counted as one word.
    counter = Counter()
    for line in lines:
        counter.update(re.findall(r"[\w\u0600-\u06FF\u200c]+", line))
    return counter


def is_digit_like(word):
    # Western, Persian and Arabic-Indic digits
    return bool(re.fullmatch(r"[0-9\u06F0-\u06F9\u0660-\u0669]+", word))


def is_already_known(word, tok):
    # True if the base tokenizer already encodes this word as ONE token
    # (mid-word form, or word-start form with the SentencePiece "▁" prefix)
    ids_mid = tok(word, add_special_tokens=False)["input_ids"]
    ids_start = tok(" " + word, add_special_tokens=False)["input_ids"]
    return len(ids_mid) == 1 or len(ids_start) == 1


def load_general_counts(cache_path="general_counts.json", n_articles=3000):
    # Word counts from Persian Wikipedia; cached so we only stream it once
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return Counter(json.load(f))
    from datasets import load_dataset
    wiki = load_dataset("wikimedia/wikipedia", "20231101.fa", split="train", streaming=True)
    lines = []
    for i, row in enumerate(wiki):
        lines.append(row["text"])
        if i + 1 >= n_articles:
            break
    counts = word_counts(lines)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(counts, f, ensure_ascii=False)
    return counts


def load_stopwords():
    from hazm import stopwords_list
    stopwords = set(stopwords_list())
    try:
        url = "https://raw.githubusercontent.com/kharazi/persian-stopwords/master/persian"
        urllib.request.urlretrieve(url, "persian_stopwords_github.txt")
        with open("persian_stopwords_github.txt", "r", encoding="utf-8") as f:
            stopwords |= set(line.strip() for line in f if line.strip())
    except Exception as e:
        print("[WARN] could not download the GitHub stopword list:", e)
    return stopwords


def select_domain_tokens(tokenizer_json_path, base_tokenizer, train_lines, cfg, verbose=True):
    """
    Pick new tokens from the custom BPE vocab. Frequencies are computed on
    train_lines ONLY, so every selected token really appears in the data the
    model trains on.
    """
    from tokenizers import Tokenizer, decoders

    raw_tok = Tokenizer.from_file(tokenizer_json_path)
    raw_tok.decoder = decoders.ByteLevel()   # decode byte-level pieces back to real UTF-8 text

    train_counts = word_counts(train_lines)
    train_total = sum(train_counts.values())
    general_counts = load_general_counts()
    general_total = sum(general_counts.values())
    stopwords = load_stopwords()

    stats = Counter()
    seen = set()
    candidates = []
    # How often does each candidate appear as a substring inside the raw train text?
    # (add_tokens matches substrings, so a short token can cut through longer words)
    train_text = "\n".join(train_lines)
    for token_str, token_id in raw_tok.get_vocab().items():
        word = raw_tok.decode([token_id]).strip()

        if len(word.replace("\u200c", "")) < 4:
            stats["too_short"] += 1
            continue
        if word in seen:
            stats["duplicate"] += 1
            continue
        seen.add(word)
        if is_digit_like(word):
            stats["digit"] += 1
            continue

        count = train_counts.get(word, 0)
        if count < cfg["min_train_count"]:
            stats["rare_in_train"] += 1
            continue
         # If the word occurs far more often INSIDE longer words than as a standalone word,
        # adding it as a token would shred those longer words.
        substring_count = train_text.count(word)
        if substring_count > 1.5 * count:
            stats["cuts_longer_words"] += 1
            continue
        if word in stopwords:
            stats["stopword"] += 1
            continue

        # Add-one smoothing: a word missing from Wikipedia is treated as seen once
        domain_freq = count / train_total
        general_freq = (general_counts.get(word, 0) + 1) / (general_total + 1)
        if domain_freq / general_freq < cfg["ratio_threshold"]:
            stats["common_word"] += 1
            continue

        if is_already_known(word, base_tokenizer):   # most expensive check, run last
            stats["already_in_base_vocab"] += 1
            continue

        candidates.append((word, count))

    candidates.sort(key=lambda x: -x[1])   # most frequent in train first
    selected = [w for w, _ in candidates[:cfg["num_new_tokens"]]]

    if verbose:
        print("[SELECT] rejected by the first rule they failed:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        print(f"[SELECT] passed all filters: {len(candidates)} -> keeping {len(selected)}")
    return selected


def build_expanded_model(base_model_name, base_tokenizer, new_tokens, init="subtoken_mean"):
    """
    Returns (model, tokenizer, old_vocab_size).
    init: "random"        -> plain random rows (mean_resizing=False)
          "hf_mean"       -> HF default (multivariate normal around old embeddings)
          "subtoken_mean" -> each new row = mean of the base sub-token embeddings of that word
    """
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.add_tokens(new_tokens)

    model = AutoModelForMaskedLM.from_pretrained(base_model_name)
    old_vocab = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer), mean_resizing=(init == "hf_mean"))

    if init == "subtoken_mean":
        emb = model.get_input_embeddings().weight
        with torch.no_grad():
            for tok, tid in tokenizer.get_added_vocab().items():
                if tid < old_vocab:
                    continue
                pieces = base_tokenizer(tok, add_special_tokens=False)["input_ids"]
                if pieces:
                    emb[tid] = emb[pieces].mean(dim=0)

    return model, tokenizer, old_vocab


def freeze_all_but_new_rows(model, old_vocab):
    # Freeze everything, unfreeze the (tied) embedding matrix,
    # and zero the gradient of the old rows so only new rows can change.
    for p in model.parameters():
        p.requires_grad = False
    emb_weight = model.get_input_embeddings().weight
    emb_weight.requires_grad = True

    def zero_old_rows(grad):
        grad = grad.clone()
        grad[:old_vocab] = 0
        return grad

    emb_weight.register_hook(zero_old_rows)


def eval_loss(model, tokenizer, val_ds, mlm_probability=0.15, batch_size=16):
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True,
                                               mlm_probability=mlm_probability)
    args = TrainingArguments(output_dir="/kaggle/working/eval_tmp",
                             per_device_eval_batch_size=batch_size,
                             seed=42, report_to="none")
    trainer = Trainer(model=model, args=args, data_collator=collator, eval_dataset=val_ds)
    return trainer.evaluate()["eval_loss"]


def fill_mask_check(model, tokenizer, tag, text="The capital of France is <mask>."):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    idx = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero()[0].item()
    probs = torch.softmax(logits[0, idx], dim=-1)
    top = probs.topk(3)
    print(tag, [(tokenizer.decode([i]).strip(), round(p.item(), 3))
                for p, i in zip(top.values, top.indices)])
