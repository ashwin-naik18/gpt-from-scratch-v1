# GPT From Scratch — V1

My first implementation of a GPT-like language model built completely from scratch using PyTorch.

The main goal of this project was not to build a large or highly optimized language model, but to understand how the core components of a Transformer-based language model work internally.

This project was the starting point of my journey toward building and improving my own language models.

---

## Overview

This model is a small decoder-only Transformer that learns to predict the next token in a sequence.

The overall pipeline is:

Text Dataset

↓

Custom Vocabulary

↓

Token IDs

↓

Token Embeddings

↓

Positional Embeddings

↓

Transformer Blocks

↓

Causal Self-Attention

↓

Multi-Head Attention

↓

Feed Forward Network

↓

Language Model Head

↓

Next Token Prediction

The model is trained using cross-entropy loss.

---

## Dataset

The model is trained using the `SimpleStories` dataset.

Dataset:

`SimpleStories/SimpleStories`

The stories are converted into a simple word-level representation before training.

---

## Tokenization

V1 uses a very simple custom word-level tokenizer.

Instead of using a subword tokenizer such as BPE, the dataset is split using:

```python
story.split()
