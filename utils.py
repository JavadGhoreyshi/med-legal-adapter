import os

def find_file(filename, search_root="/kaggle/input"):
    for root, _, files in os.walk(search_root):
        if filename in files:
            return os.path.join(root, filename)
    if os.path.exists(filename):
        return filename
    return None

def tokenize_batch(batch, tokenizer, max_length):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
