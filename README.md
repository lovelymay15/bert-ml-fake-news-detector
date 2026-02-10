# Fake News Detection using BERT

A machine learning model leveraging BERT (Bidirectional Encoder Representations from Transformers) for detecting and classifying misinformation in news articles.

## Overview

Developed as a group project for the Machine Learning course in 2025, this system utilizes advanced natural language processing to distinguish between real and fake news headlines. The model demonstrates exceptional performance on government-related news, achieving higher accuracy rates compared to general news classification.

## Model Performance

- **Overall Accuracy:** 94.75%
- **Precision:** 93.95%
- **F1 Score:** 94.18%
- **Sensitivity (Recall):** 94.42%
- **Specificity:** 95.03%

*Note: The model performs significantly better on government-related news compared to general news topics.*

## Key Features

- BERT-based binary classification (real vs. fake news)
- Headline-focused classification approach
- Comprehensive data preprocessing pipeline (language filtering, tokenization, deduplication)
- GUI application for interactive testing
- Pre-trained model fine-tuning with AdamW optimizer

## Dataset

**Source:** WELFake dataset (Kaggle)

**Preprocessing:**
- Initial entries: 72,134
- Final preprocessed: 60,048 English-language titles
- Language filtering using langdetect library
- Removal of null values, duplicates, and ambiguous entries

## Technical Implementation

**Model:** bert-base-uncased (pre-trained)

**Training Configuration:**
- Optimizer: AdamW (learning rate: 3e-5)
- Epochs: 3
- Batch size: 32
- Train/Validation/Test split: 80/10/10

**Libraries & Tools:**
- PyTorch
- Hugging Face Transformers
- Python (VS Code)
- Pandas, NumPy
- Tkinter (GUI)

## Limitations

- Headline-only classification approach
- Binary classification (does not capture nuanced misinformation types)
- Optimized for government-related news; may have reduced accuracy on other topics
- Single dataset dependency (WELFake)

