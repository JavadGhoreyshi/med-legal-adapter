import json, random, torch, gc
import torch.nn.functional as F
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login, hf_hub_download
from transformers import AutoTokenizer, AutoModel
from config import CONFIG

login(token=UserSecretsClient().get_secret("HF_TOKEN"))
dev = "cuda"

path = hf_hub_download("JavadGhoreyshi/med-legal-bilingual-pairs",
                       "pairs_unseen.jsonl", repo_type="dataset")
rows = [json.loads(l) for l in open(path, encoding="utf-8")]
random.Random(42).shuffle(rows)
test, train = rows[:1000], rows[1000:]
print("train:", len(train), "| test:", len(test))

MAXLEN, BS, EPOCHS, LR, TEMP = 192, 16, 2, 2e-5, 0.05

def embed(model, tok, texts):
    enc = tok(texts, truncation=True, max_length=MAXLEN,
              padding=True, return_tensors="pt").to(dev)
    out = model(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1).float()
    return F.normalize((out * m).sum(1) / m.sum(1), dim=-1)

@torch.no_grad()
def evaluate(model, tok):
    model.eval()
    fa, en = [], []
    for i in range(0, len(test), 64):
        b = test[i:i + 64]
        fa.append(embed(model, tok, [r["fa_q"] for r in b]).cpu())
        en.append(embed(model, tok, [r["en_q"] for r in b]).cpu())
    fa, en = torch.cat(fa), torch.cat(en)
    sim = fa @ en.T
    res = {}
    for name, s in [("fa->en", sim), ("en->fa", sim.T)]:
        ranks = (s > s.diag().unsqueeze(1)).sum(1) + 1
        res[name] = {"top1": round((ranks == 1).float().mean().item(), 3),
                     "top5": round((ranks <= 5).float().mean().item(), 3)}
    return res

def run(name, repo):
    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModel.from_pretrained(repo).to(dev)
    print(f"\n[{name}] before fine-tuning:", evaluate(model, tok))

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * (len(train) // BS)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps))
    data, step = list(train), 0
    for ep in range(EPOCHS):
        random.Random(ep).shuffle(data)
        model.train()
        for i in range(0, len(data) - BS + 1, BS):
            b = data[i:i + BS]
            a = embed(model, tok, [r["fa_q"] for r in b])
            c = embed(model, tok, [r["en_q"] for r in b])
            logits = a @ c.T / TEMP
            labels = torch.arange(len(b), device=dev)
            loss = (F.cross_entropy(logits, labels) +
                    F.cross_entropy(logits.T, labels)) / 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 50 == 0:
                print(f"  step {step}/{steps} loss {loss.item():.3f}")
    print(f"[{name}] after fine-tuning:", evaluate(model, tok))
    del model, opt
    gc.collect(); torch.cuda.empty_cache()

run("base", CONFIG["base_model"])
run("phase2", CONFIG["model_repo"] + "-phase2")
