# Med-Legal XLM-RoBERTa Adapter

Adapts `xlm-roberta-base` to the bilingual (Persian + English) medical + legal domain by adding domain-specific tokens to the vocabulary and training in phases (embedding warm-up first, then gradual unfreezing of transformer layers).

## Project Status

- [x] Phase 0: Extract domain tokens from the medical + legal corpus
- [x] Phase 1: Embedding warm-up (backbone frozen, only new embedding rows trainable). Saved at: `JavadGhoreyshi/med-legal-xlm-roberta-v4`
- [x] Phase 2: Unfreeze the last 2 transformer layers (plus new embedding rows and new output-bias rows). Saved at: `JavadGhoreyshi/med-legal-xlm-roberta-v4-phase2`
- [ ] Phase 3: Evaluation and/or fine-tuning on the final task

## Results So Far

Rank of the correct new token when it is masked in a validation sentence (553 validation sentences that contain at least one new token; lower rank is better):

| Stage | Median rank | Top-10 |
|---|---|---|
| Before training (subtoken-mean init) | 462 | 9.9% |
| After phase 1 | 85 | 25.7% |
| After phase 2 | 32 | 35.6% |

- Validation MLM loss during phase 2: 5.766 before, 5.311 after. This loss is not directly comparable to `xlm-roberta-base` (4.28 on the same text), because the tokenizers differ.
- Informal sanity check: on 5 simple general sentences, the phase 2 model gives the same top-1 prediction as `xlm-roberta-base`. This is a small check, not a benchmark.

## Phase Details

- Phase 1: only the new embedding rows are trained (old rows are kept unchanged by zeroing their gradients). Learning rate 5e-4, 1 epoch.
- Phase 2: loads the phase 1 model from the Hugging Face Hub. Trainable: the last 2 encoder layers (learning rate 2e-5), the new embedding rows and the new output-bias rows (learning rate 1e-4). Old embedding rows and old bias values are checked to be unchanged before uploading.

## Repository Structure

- `config.py`: central settings (repo names, hyperparameters)
- `utils.py`: shared helper functions
- `requirements.txt`: Python dependencies
- `scripts/train_phase1_warmup.py`: Phase 1 (archived, already done)
- `scripts/train_phase2_unfreeze.py`: Phase 2

## How to Run on Kaggle

### Setup (same for every phase)

1. Enable Internet and GPU in the notebook settings.
2. Add your Hugging Face write token as a Kaggle Secret named `HF_TOKEN`.
3. Run:

```python
!git clone https://github.com/JavadGhoreyshi/med-legal-adapter.git
%cd med-legal-adapter
!pip install -r requirements.txt --quiet
```

### Run a phase

```python
# Phase 2
!PYTHONPATH=. python scripts/train_phase2_unfreeze.py

# Phase 3 (add here when ready)
```

## Reproducibility Note

Model `v4` was created with an earlier version of `utils.py` whose token selection did not break ties between tokens with equal counts. The ids of those tokens could differ between runs, so re-running token selection does not reproduce the vocabulary of the uploaded model. Always load the tokenizer and model from the Hugging Face Hub (as phase 2 does), and do not re-run token selection for later phases. The current `utils.py` sorts candidates by count and then alphabetically, so future runs are deterministic.

## Where Things Are Stored

- Code: this GitHub repository
- Models and tokenizer: Hugging Face Hub (`JavadGhoreyshi/med-legal-xlm-roberta-v4` for phase 1, `JavadGhoreyshi/med-legal-xlm-roberta-v4-phase2` for phase 2)
- Tokenized dataset: Hugging Face Hub (`JavadGhoreyshi/med-legal-tokenized-dataset-v4`)
- Training runs: Kaggle (GPU)
