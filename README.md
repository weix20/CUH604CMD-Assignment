## Project 1 – Binary Classification with FNN

## Task
- Predict a binary label (`y ∈ {0,1}`) from 15 binary features  
- Dataset: `data.csv`

## Data Preparation
- Train/test split: 80/20, with `stratify=y` and `random_state=24`  
- Features standardized using `StandardScaler` (fit on training set)

## Model
- Feedforward Neural Network (Dense MLP)  
- Hidden layers: Dense + ReLU  
- Output layer: Sigmoid  
- Regularization: Dropout  

## Training
- Optimizer: Adam  
- Loss: Binary cross-entropy  
- Batch size: 32  
- Epochs: up to 50 with EarlyStopping (`val_loss`, `patience=5`, `restore_best_weights=True`)  

## Hyperparameters
- **Model 1:** 3×64, dropout 0.2, lr=0.001  
- **Model 2:** 4×128, dropout 0.3, lr=0.0005  
- **Model 3:** 2×32, dropout 0.1, lr=0.002  

## Evaluation
- Metrics: Accuracy, Precision, Recall, F1-score, ROC AUC  
- ROC AUC computed from predicted probabilities  
- Best model selected by test accuracy (tie-breaker: F1/ROC AUC)  



# Project 2 — Fine-tuned Academic Tutor (5000CMD: Theory of Computation)

This project fine-tunes a compact causal language model (**distilgpt2**) on curated text from Coventry University’s **5000CMD (Theory of Computation)** pages to generate concise, topic-aligned explanations and examples (e.g., DFAs, regular languages). It demonstrates the full AI development lifecycle: data collection, preprocessing, model selection, fine-tuning, evaluation, and ethical considerations.

## 1. Problem Identification
Students often need quick, customized clarifications on course concepts. We aim to create a small, locally fine-tuned tutor model that can:
- Produce short explanations for 5000CMD topics
- Illustrate concepts with brief examples
- Be retrained easily as taught material evolves

**Expected output:** short, domain-grounded completions given academic prompts (e.g., “In automata theory, a deterministic finite automaton …”).

## 2. Data Collection & Preparation
- **Source:** https://github.coventry.ac.uk/pages/ab3735/5000CMD/
- **Method:** lightweight crawl (same-site HTML only), boilerplate stripping, normalization, deduplication.
- **Artifacts:** `clean/5000cmd_clean.jsonl` and `clean/5000cmd_clean.csv`.
- **Dataset:** built with `datasets.DatasetDict` and split into **train/val/test = 80/10/10** (fixed seed for reproducibility).
> Note: For production, respect `robots.txt`, add rate limiting, and keep a URL allowlist.

## 3. Model Selection
- **Model:** `distilgpt2` (a compact GPT-2 variant)
- **Why:** good trade-off between memory, speed, and fluency for Colab.
- **Tokenizer:** pad token mapped to eos (GPT-2 family lacks pad).

## 4. Fine-tuning
- **Approach:** Transformers model + **custom PyTorch loop** (AdamW, LR warmup+decay, AMP, gradient accumulation, periodic eval, checkpointing).
- **Input format:** concatenated text chunked to fixed `BLOCK_SIZE` for causal LM.

## 5. Evaluation
- **Metrics:** validation **loss** and **perplexity (PPL)**.
- **Qualitative:** sample generations from academic prompts.
- **(Optional)** report basic latency (tokens/sec) during generation to satisfy “response time”.

### Reading the results
- Lower **val loss/PPL** indicates better next-token modeling on course text.
- Qualitative generations should stay on-topic (DFAs, regular languages, etc.) and avoid hallucinating external facts.

## 6. Results Analysis (example template)
- The model converged to val PPL ≈ *X.Y* after *N* epochs with `BLOCK_SIZE=256`.
- Generations are mostly on-topic and use course terminology.
- Failure modes: occasional vague phrasing or incomplete proofs; benefits from more in-domain data.

## 7. Ethical Considerations
- **Privacy & Terms:** Only crawl course pages you are permitted to use; avoid personal data. Respect `robots.txt` and site load.
- **Bias & Academic Integrity:** Outputs may reflect source biases; the model is a *study aid*, not an assessment surrogate. Students must cite sources and follow university academic-integrity policies.
- **Misuse:** Limit generation length; instruct users to verify outputs against official materials.

## 8. How to Run (Colab)
1. Open the notebook in Colab.
2. Run the environment cell (pinned versions).
3. Run the **crawl → clean → dataset** cells (produces JSONL/CSV and an HF dataset on disk).
4. Run **tokenization → chunking → data collator**.
5. Run **training** (custom loop) and **evaluation**.
6. Use the final **generation cell** to sanity-check outputs on an academic prompt.

## 9. Model Card (Mitchell et al., 2018 — brief)
- **Model:** distilgpt2 fine-tuned on 5000CMD text
- **Intended Use:** educational assistance, short explanations/examples
- **Out-of-Scope:** grading, formal proofs without human verification
- **Data:** publicly accessible course pages (cleaned/deduplicated)
- **Metrics:** val loss/PPL; sample generations; (optional) latency
- **Ethics & Risks:** see Section 7
- **Maintainer & Contact:** *Your name/email*

