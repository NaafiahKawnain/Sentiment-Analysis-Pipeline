# Sentiment Analysis: Classical ML vs Transformers

A sentiment classification project comparing a classical ML baseline against a pretrained Hugging Face transformer on the same dataset, to evaluate the real-world tradeoff between the two approaches.

## Overview
Two models are built and evaluated on identical train/test splits:
1. A TF-IDF + Logistic Regression baseline
2. A pretrained DistilBERT model via Hugging Face's `pipeline("sentiment-analysis")`

The goal isn't just to build a classifier, but to compare speed, interpretability, and accuracy tradeoffs between classical NLP and transformer-based approaches.

## Tech Stack
- **Vectorization**: TF-IDF (`scikit-learn`)
- **Classical Model**: Logistic Regression
- **Transformer**: Hugging Face `transformers` pipeline (pretrained DistilBERT, SST-2 fine-tuned)
- **Language**: Python (Jupyter Notebook)

## Pipeline
1. **Load & explore** — dataset loading, class balance check, sample inspection
2. **Preprocess** — text cleaning (lowercasing, HTML/punctuation removal)
3. **Baseline (Part A)** — TF-IDF vectorization + Logistic Regression, trained and evaluated
4. **Transformer (Part B)** — pretrained Hugging Face pipeline run on the same test set, no fine-tuning
5. **Comparison (Part C)** — side-by-side metrics and tradeoff analysis

## Results

| Metric | TF-IDF + Logistic Regression | HF Transformer (pretrained) |
|---|---|---|
| Accuracy | 85% | 87% |
| F1 | 85% | — |
| Precision (Negative) | — | 82% |
| Recall (Negative) | — | 95% |
| Precision (Positive) | — | 94% |
| Recall (Positive) | — | 78% |

## Key Takeaway
The pretrained transformer edges out the classical baseline on accuracy, but the gap is modest — not a dramatic win. The transformer leans heavily toward catching Negative sentiment (high recall) at some cost to Positive recall, while the TF-IDF baseline stays simpler, faster, and more interpretable. This reflects a realistic tradeoff: classical ML remains a strong, lightweight option, while transformers offer a modest accuracy edge at higher compute cost.

## Limitations
- No fine-tuning performed on the transformer — used purely as a pretrained pipeline
- Baseline preprocessing is minimal (no lemmatization/advanced NLP cleaning)
- Evaluated on a single dataset/domain — results may not generalize to other text domains

## How to Run
1. Install dependencies: `scikit-learn`, `transformers`, `pandas`
2. Run all cells in the notebook sequentially — Part A (baseline), then Part B (transformer), then Part C (comparison)
