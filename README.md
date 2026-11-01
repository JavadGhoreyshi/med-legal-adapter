# Med-Legal XLM-RoBERTa Adapter

Adapts `xlm-roberta-base` to the bilingual (Persian + English) medical + legal domain by adding domain-specific tokens to the vocabulary and training in phases (embedding warm-up first, then gradual unfreezing of transformer layers).

This is a learning project. It is finished: the work stopped after the Phase 3 check, and the results below include a negative one (no measurable downstream advantage over the base model).

## Project Status

- [x] Phase 0: Build the corpus and extract domain tokens from the medical + legal corpus
- [x] Phase 1: Embedding warm-up (backbone frozen, only new embedding rows trainable). Saved at: `JavadGhoreyshi/med-legal-xlm-roberta-v4`
- [x] Phase 2: Unfreeze the last 2 transformer layers (plus new embedding rows and new output-bias rows). Saved at: `JavadGhoreyshi/med-legal-xlm-roberta-v4-phase2`
- [x] Phase 3: Downstream check (Persian-English retrieval): no measurable advantage over the base model

## Summary of Results

- The adapted tokenizer is about 9% more efficient on the validation text (tokens per word 1.38 -> 1.25).
- The new tokens were learned: median rank of the correct new token went from 462 (initialization) to 85 (phase 1) to 32 (phase 2).
- General ability was not visibly damaged (informal check on 5 simple sentences).
- On a Persian-English retrieval task, the adapted model showed no measurable advantage over `xlm-roberta-base`.

## Results

### Learning of the new tokens

Rank of the correct new token when it is masked in a validation sentence (553 validation sentences that contain at least one new token; lower rank is better):

| Stage | Median rank | Top-10 |
|---|---|---|
| Before training (subtoken-mean init) | 462 | 9.9% |
| After phase 1 | 85 | 25.7% |
| After phase 2 | 32 | 35.6% |

- Validation MLM loss during phase 2: 5.766 before, 5.311 after. This loss is not directly comparable to `xlm-roberta-base` (4.28 on the same text), because the tokenizers differ.
- Informal sanity check: on 5 simple general sentences, the phase 2 model gives the same top-1 prediction as `xlm-roberta-base`. This is a small check, not a benchmark.

### Tokenizer efficiency (2,000 validation lines)

| Text | tokens/word, base | tokens/word, adapted | truncated at 128 tokens, base -> adapted |
|---|---|---|---|
| All | 1.38 | 1.25 | 18.4% -> 14.8% |
| Persian | 1.34 | 1.22 | 22.6% -> 18.4% |
| English | 1.72 | 1.55 | 3.2% -> 1.6% |

### Phase 3: downstream check (Persian-English question retrieval)

Task: for each Persian medical question, find its English translation among 1,000 test questions (and the reverse). Pairs come from corpus items whose four lines were never part of the phase 1-2 train or validation splits (`JavadGhoreyshi/med-legal-bilingual-pairs`): 4,244 for training and 1,000 for testing. Both models were fine-tuned with the same contrastive setup (mean pooling, max length 192, batch size 16, temperature 0.05, 2 epochs, learning rate 2e-5, same seed), single run.

| Model | fa->en top-1 | en->fa top-1 |
|---|---|---|
| xlm-roberta-base, before fine-tuning | 43.0% | 28.7% |
| phase 2 model, before fine-tuning | 39.5% | 25.3% |
| xlm-roberta-base, after fine-tuning | 97.4% | 96.6% |
| phase 2 model, after fine-tuning | 96.4% | 95.4% |

Conclusion: on this task the adapted model shows no measurable advantage over the base model. The differences (about 1 point) are within the noise of a single run on 1,000 test pairs. The task is probably too easy (both models reach about 97%), so it cannot separate them.

## Limitations

- The downstream check uses medical pairs only. The legal side was not evaluated downstream.
- Single run, single seed, 1,000 test pairs. Differences of about 1 point are within noise.
- The retrieval task is nearly saturated (about 97%), so it cannot separate the models.
- The general-ability check uses only 5 sentences.
- MLM loss is not comparable between models with different tokenizers.
- 2,000 tokens were selected but 1,990 new embedding rows were added. The difference of 10 was not investigated.

## Data

- Medical: `MohammadJRanjbar/PersianMedQA` (train split). Persian and English question + options, written as 4 lines per item.
- Legal: `PerSets/iran-legal-persian-qa`. The first ~60 MB are streamed with `scripts/00_download_legal_data.py`, then cleaned in the notebook.
- The unified corpus has 142,964 lines (58,190 medical, 84,772 legal).
- The corpus (`bilingual_med_legal_corpus.txt`) and the custom tokenizer file (`med_legal_tokenizer.json`) are not in this repository (see `.gitignore`). Regenerate them with `scripts/00_build_corpus_and_tokenizer.ipynb`.
- The licenses of the source datasets were not checked here. Check them before sharing models or derived data.

## Repository Structure

- `config.py`: central settings (repo names, hyperparameters)
- `utils.py`: shared helper functions
- `requirements.txt`: Python dependencies
- `scripts/00_download_legal_data.py`: downloads the legal text
- `scripts/00_build_corpus_and_tokenizer.ipynb`: builds the unified corpus and trains the custom BPE tokenizer
- `scripts/01_data_quality.py`: raw corpus checks
- `scripts/02_tokenizer_extraction.py`: selects the new domain tokens
- `scripts/03_pretraining_sanity.py`: compares initialization methods on the validation set before training
- `scripts/04_training_trend.py`: short training run with validation loss tracking
- `scripts/05_shred_check.py`: checks how many ordinary words are split differently after adding the tokens
- `scripts/train_phase1_warmup.py`: Phase 1 (archived, already done; see the warning below)
- `scripts/train_phase2_unfreeze.py`: Phase 2
- `scripts/06_retrieval_eval.py`: Phase 3 retrieval comparison

## How to Run on Kaggle

### Setup (same for every script)

1. Enable Internet and GPU in the notebook settings.
2. Add your Hugging Face write token as a Kaggle Secret named `HF_TOKEN`.
3. Add `bilingual_med_legal_corpus.txt` and `med_legal_tokenizer.json` as Kaggle input datasets (the scripts look for them under `/kaggle/input`).
4. Run:

```python
!git clone https://github.com/JavadGhoreyshi/med-legal-adapter.git
%cd med-legal-adapter
!pip install -r requirements.txt --quiet
```

### Run a script

```python
# Phase 2
!PYTHONPATH=. python scripts/train_phase2_unfreeze.py

# Phase 3 retrieval comparison
!PYTHONPATH=. python scripts/06_retrieval_eval.py
```

## Phase Details

- Token selection: candidates come from the custom BPE vocabulary. A word is kept only if it has at least 4 characters, is not a digit string, appears at least 5 times in the training split, is not mostly a fragment of longer words, is not a stopword, is at least 5 times more frequent in the corpus than in a Persian Wikipedia sample, and is not already a single token in the base tokenizer. The 2,000 most frequent remaining words are selected.
- Phase 1: only the new embedding rows are trained (old rows are kept unchanged by zeroing their gradients). Learning rate 5e-4, 1 epoch, subtoken-mean initialization (each new row starts as the mean of the base sub-token embeddings of that word).
- Phase 2: loads the phase 1 model from the Hugging Face Hub. Trainable: the last 2 encoder layers (learning rate 2e-5), the new embedding rows and the new output-bias rows (learning rate 1e-4). Old embedding rows and old bias values are checked to be unchanged before uploading.

## Warning

`scripts/train_phase1_warmup.py` pushes to `model_repo` (`v4`). Do not re-run it: token selection would not reproduce the uploaded vocabulary and would overwrite `v4`.

## Reproducibility Note

Model `v4` was created with an earlier version of `utils.py`. When token selection was re-run, the resulting vocabulary differed from the uploaded tokenizer. The most likely cause is that tokens with equal counts were ordered arbitrarily (the order of `get_vocab()` is not fixed). Other possible causes are the stopword list downloaded at run time and the Wikipedia word counts rebuilt in a new session. This was not tested separately. The current `utils.py` breaks ties alphabetically, but a new run will still not necessarily match `v4`. Always load the tokenizer and model from the Hugging Face Hub, as phase 2 does, and do not re-run token selection for later phases.

## Where Things Are Stored

- Code: this GitHub repository
- Models and tokenizer: Hugging Face Hub (`JavadGhoreyshi/med-legal-xlm-roberta-v4` for phase 1, `JavadGhoreyshi/med-legal-xlm-roberta-v4-phase2` for phase 2)
- Phase 3 pairs: Hugging Face Hub (`JavadGhoreyshi/med-legal-bilingual-pairs`)
- Training runs: Kaggle (GPU)

Legacy Hugging Face repositories, not used by the current scripts: `med-legal-xlm-roberta` (v1, only referenced by `03_pretraining_sanity.py`), `med-legal-xlm-roberta-v3` (broken), `med-legal-tokenized-dataset`, `-v3` and `-v4`.
