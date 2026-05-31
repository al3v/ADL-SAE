# ADL-SAE Project

This repository contains an Applied Deep Learning project for probing early-layer activations of Gemma 2 2B on factual QA prompts.

## Goal

The project investigates whether early-layer representations of factual prompts show differences between correctly and incorrectly produced factual answers, while controlling for tokenization artifacts.

## Current Pipeline

1. `src/01_generate_answers.py`  
   Runs Gemma 2 2B on factual QA prompts and saves model answers.

2. `src/02_tokenization_analysis.py`  
   Analyzes how correct answers and model answers are tokenized.

3. `src/03_collect_early_activations.py`  
   Extracts early-layer hidden activations from layers 1, 2, and 3.

4. `src/04_activation_summary.py`  
   Computes simple activation statistics such as mean, standard deviation, and vector norm.

5. `src/05_merge_tokenization_activation.py`  
   Merges tokenization statistics with activation summaries.

## Notes

Large model files, virtual environments, and activation tensor `.pt` files are excluded from Git.
## Current Baseline Results

The current cleaned dataset contains 109 factual QA prompts. After removing ambiguous questions and formatting-based false negatives, Gemma 2 2B produced:

- 100 exact-match correct answers
- 9 exact-match incorrect answers

The incorrect cases are mainly hard factual questions involving authors, painters, scientific discoveries, and physics history.

For each prompt, early-layer activations were extracted from Gemma 2 2B at the final prompt token:

```text
Question: ...
Answer:


Activations were saved for layers 1, 2, and 3. Since the dataset contains 109 questions and 3 layers, the pipeline produces 327 activation vectors.

Activation Norm Baseline

The average activation norms were:

Layer 1: 52.93
Layer 2: 57.90
Layer 3: 54.08

Single-token and multi-token answers showed similar activation norms, suggesting that simple activation magnitude alone does not strongly explain tokenization differences.

Correct and incorrect answers also showed similar activation norms:

Layer 1:
incorrect: 52.83
correct:   52.94

Layer 2:
incorrect: 57.80
correct:   57.91

Layer 3:
incorrect: 53.66
correct:   54.12

This suggests that raw activation magnitude is not enough to distinguish correct and incorrect factual cases.

Cosine Similarity Baseline

A cosine similarity analysis was also added. The raw activation vectors are highly similar across prompts, especially because all prompts share the same format:

Question: ...
Answer:

This motivates the next step: using Sparse Autoencoders to inspect more fine-grained feature-level differences rather than relying only on raw activation norms or cosine similarity.
