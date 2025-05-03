# ChatGPT‑Generated‑Abstracts‑Detection

**Detect whether a research abstract is human‑written, AI‑generated, AI‑polished, or a human–AI mix.**

_Data sourced from the CHEAT dataset introduced by Xu et al. (2023): https://arxiv.org/pdf/2304.12008_

---

## 🔑 Key Features

- **3 binary tasks**:  
  - Human vs Generation  
  - Human vs Polish  
  - Human vs Mix  
- **Rich linguistic features**: length, burstiness, vocabulary metrics, TF–IDF patterns, readability, POS distributions, perplexity…  
- **Models**:  
  - Logistic Regression  
  - XGBoost  
  - DistilRoBERTa + LoRA (text‑only)

---

## 📊 Data Split

For each class (human, generation, polish, mix):  
- **Train**: 80% (human/gen/polish 12 316; mix 3 611)  
- **Test**: 20% (human/gen/polish 3 079; mix 903)  

> Human vs Mix tasks down‑sample humans to match mix counts.

---

## 📈 Results

| Model                    | Task                 | Accuracy | AUC    |
|--------------------------|----------------------|----------|--------|
| Logistic Regression      | Human vs Generation  | 97.19%   | 99.55% |
|                          | Human vs Polish      | 78.48%   | 85.84% |
|                          | Human vs Mix         | 65.73%   | 71.31% |
| XGBoost                  | Human vs Generation  | 97.73%   | 99.71% |
|                          | Human vs Polish      | 79.67%   | 88.49% |
|                          | Human vs Mix         | 66.17%   | 72.26% |
| DistilRoBERTa + LoRA     | Human vs Generation  | 91.86%   | 99.98% |
|                          | Human vs Polish      | 73.04%   | 99.02% |
|                          | Human vs Mix         | 62.18%   | 84.92% |
