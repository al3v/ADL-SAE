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

## Type-Balanced PopQA Later-Layer SAE Results

To reduce dataset and question-type confounding, a larger PopQA pool was generated and evaluated with alias-aware answer matching using PopQA's possible_answers field.

From this pool, a type-balanced dataset was created with:

- 220 total samples
- 110 correct answers
- 110 incorrect answers
- 11 question types
- 10 correct and 10 incorrect samples per type

The balanced question types were:

- author
- capital
- capital of
- composer
- country
- director
- father
- place of birth
- producer
- screenwriter
- sport

This makes the experiment stronger than the earlier manually created dataset and the smaller PopQA subsets, because correct and incorrect examples are balanced both globally and within each question type.

### Later-Layer SAE Setup

Later-layer activations were extracted from Gemma 2 2B at the final prompt token.

The analyzed layers were:

- Layer 6
- Layer 9
- Layer 12

For each activation, a matching Gemma Scope SAE was loaded manually from google/gemma-scope-2b-pt-res.

The selected SAEs were:

- Layer 6: layer_6/width_16k/average_l0_36/params.npz
- Layer 9: layer_9/width_16k/average_l0_37/params.npz
- Layer 12: layer_12/width_16k/average_l0_22/params.npz

For each sample and layer, the top 100 SAE features were saved.

This produced:

- 220 samples x 3 layers x 100 features = 66000 rows

### SAE Sparsity and Feature Norms

Average number of nonzero SAE features:

- Layer 6:
  - incorrect: 247.44
  - correct: 247.17

- Layer 9:
  - incorrect: 119.68
  - correct: 118.93

- Layer 12:
  - incorrect: 154.00
  - correct: 151.92

Average SAE feature norm:

- Layer 6:
  - incorrect: 93.69
  - correct: 93.66

- Layer 9:
  - incorrect: 88.58
  - correct: 88.27

- Layer 12:
  - incorrect: 139.28
  - correct: 138.64

These results show that simple SAE statistics such as feature count and feature norm still do not strongly separate correct and incorrect answers.

### Feature-Level Enrichment

Feature-level SAE analysis was more informative than simple magnitude statistics.

In layer 12, several features were enriched in incorrect examples.

The strongest global wrong-enriched candidates were:

- Feature 13338:
  - incorrect frequency: 39.1%
  - correct frequency: 16.4%
  - difference: +22.7%

- Feature 33:
  - incorrect frequency: 21.8%
  - correct frequency: 0.9%
  - difference: +20.9%

- Feature 3830:
  - incorrect frequency: 50.9%
  - correct frequency: 32.7%
  - difference: +18.2%

- Feature 8649:
  - incorrect frequency: 36.4%
  - correct frequency: 19.1%
  - difference: +17.3%

- Feature 4820:
  - incorrect frequency: 49.1%
  - correct frequency: 33.6%
  - difference: +15.5%

Correct-enriched features also appeared. The strongest one was:

- Feature 2987:
  - incorrect frequency: 11.8%
  - correct frequency: 52.7%
  - difference: -40.9%

### Type-Controlled Findings

Because the dataset is balanced within each question type, type-controlled comparisons are more meaningful.

Some layer-12 examples:

- author, feature 3830:
  - incorrect: 9 / 10
  - correct: 1 / 10

- author, feature 13651:
  - incorrect: 8 / 10
  - correct: 0 / 10

- author, feature 8649:
  - incorrect: 10 / 10
  - correct: 3 / 10

- producer, feature 5012:
  - incorrect: 9 / 10
  - correct: 2 / 10

- screenwriter, feature 8523:
  - incorrect: 10 / 10
  - correct: 4 / 10

These results suggest that later-layer SAE features can capture meaningful differences between correct and incorrect factual answers, especially when controlling for question type.

### Interpretation

The main conclusion is:

Early layers 1 to 3 mostly showed weak or unstable correctness signals. Later layers 6, 9, and especially 12 showed stronger feature-level differences. Simple activation norms and SAE norms were not enough. Type-balanced SAE feature enrichment was the most informative analysis.

This supports the idea that factual correctness is not easily visible through raw activation magnitude, but may be partially reflected in later-layer sparse feature patterns.

