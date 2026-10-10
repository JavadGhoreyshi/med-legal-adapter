# Med-Legal XLM-RoBERTa Adapter

Adapts `xlm-roberta-base` to the bilingual medical + legal domain by adding
domain-specific tokens to the vocabulary and training in phases
(embedding warm-up first, then gradual unfreezing of transformer layers).

## Project Status

- [x] Phase 0: Extract domain tokens from the medical + legal corpus
- [ ] Phase 1: Embedding warm-up (backbone frozen, only embeddings trainable)
      Saved at: `JavadGhoreyshi/med-legal-xlm-roberta`
- [ ] Phase 2: Unfreeze the last 2 transformer layers with a lower learning rate
- [ ] Phase 3: Evaluation and/or fine-tuning on the final task

## Repository Structure

- `config.py`: central settings (repo names, hyperparameters)
- `utils.py`: shared helper functions
- `scripts/train_phase1_warmup.py`: Phase 1 (archived, already done)
- `scripts/train_phase2_unfreeze.py`: Phase 2
- 
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

## Where Things Are Stored

- Code: this GitHub repository
- Models and tokenizer: Hugging Face Hub (`JavadGhoreyshi/med-legal-xlm-roberta`)
- Tokenized dataset: Hugging Face Hub (`JavadGhoreyshi/med-legal-tokenized-dataset`)
- Training runs: Kaggle (GPU)
