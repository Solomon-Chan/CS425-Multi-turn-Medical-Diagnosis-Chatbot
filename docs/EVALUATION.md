# Evaluation guide — tokenizer-aligned evaluation

This document explains the updated evaluation procedure and commands for `scripts/evaluate_ner.py` and how to quickly run checks locally.

## Goals
- Ensure gold labels and model predictions share the exact same tokenization.
- Provide a fast feedback loop for evaluation on CPU via `--max-examples`.
- Provide troubleshooting steps when system metrics differ from fine-tuning logs.

## Key commands

Run a quick evaluation on 50 examples (fast on CPU):

```cmd
python scripts\evaluate_ner.py --max-examples 50
```

Run the full evaluation (may be slow on CPU):

```cmd
python scripts\evaluate_ner.py
```

## What changed
- The evaluator now encodes the pre-tokenized `tokens` field with the model tokenizer using `is_split_into_words=True` and uses `word_ids()` to produce aligned gold labels and predictions. This eliminates a common cause of off-by-one and subword-boundary mismatches between gold and predicted spans.

## Quick checks and troubleshooting

1. Check tokenizer vs stored tokens

If you want to inspect whether the tokenizer's tokenization differs from the `tokens` stored in `bio_outputs/validation.jsonl` for a sample:

```python
from transformers import AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained('models/biobert-ner')
with open('bio_outputs/validation.jsonl') as f:
    obj = json.loads(f.readline())
    text_tokens = obj['tokens']
    tok = tokenizer(text_tokens, is_split_into_words=True)
    print('stored tokens  :', text_tokens[:50])
    print('tokenizer ids :', tok.input_ids[:50])
    print('word ids      :', tok.word_ids())
```

2. Confirm model label mapping

Ensure the model's `id2label` matches the expected labels used during training:

```cmd
python -c "from transformers import AutoModelForTokenClassification; m=AutoModelForTokenClassification.from_pretrained('models/biobert-ner'); print(m.config.id2label)"
```

If the mapping differs, predictions will need to be remapped before evaluation.

3. Reproduce Trainer.evaluate (optional)

If you need exact parity with training-time metrics, re-run the HuggingFace `Trainer.evaluate()` pipeline using the same dataset and preprocessing used during fine-tuning. This requires loading the checkpoint and a dataset prepared the same way (tokenized using the same function). See the training notebook for the original preprocessing code, or ask for help to generate a `trainer_evaluate.py` snippet.

## Expected behavior
- After alignment, token-level and entity-level metrics should reflect correct matching across subwords. If recall is still low, inspect sample false negatives printed by the evaluator — many labels originate from silver (auto-labeled) data and can be noisy.

## Dependencies
Make sure you have the required dependencies installed (see `requirements.txt`). On Windows, prefer running evaluation on a machine with a GPU if available; otherwise use `--max-examples` for quick iteration.

## Contact / Next steps
If you want, I can:
- Run a short evaluation here (e.g., 30 examples) and paste the output and a few FP/FN examples.
- Add a `trainer_evaluate.py` snippet that reproduces the training evaluation with `Trainer.evaluate()`.

Choose which action you'd like next.
