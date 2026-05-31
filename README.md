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
