
# Retrieval-Augmented Generation (RAG)

## Research paper
Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks was published by Patrick Lewis and collaborators in 2020.

## Main idea
RAG combines information stored in model parameters with an external source of knowledge. A retriever finds relevant passages, and a language model uses retrieved information when generating a response.

## Why retrieval matters
A model may not reliably know every fact or have access to updated information. Retrieving relevant evidence can help make answers more specific and provide a traceable source for further checking.

## Important limitation
Retrieval does not automatically guarantee that the final generated answer is correct. The retrieved passages must be relevant, and the generated answer should be checked against them.

## Source
Research paper:
https://arxiv.org/abs/2005.11401

Title: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
Authors: Patrick Lewis and collaborators
Published: 2020

Source type: Research paper abstract
Accessed: 2026-10-13
