<div align="center">

# RAG Learning

![Status](https://img.shields.io/badge/status-actively%20updating-299957?style=for-the-badge)
![Python](https://img.shields.io/badge/python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge)

</div>

## Purpose

A Python project for exploring Retrieval-Augmented Generation (RAG) with LangChain.

## Goal

- Load documents
- Prepare text for retrieval
- Explore embeddings and semantic search
- Generate answers using retrieved context

## Repo structure

```text
RAG_learning/
├── document_loaders/
│   └── text_loader.py
├── main.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## Tech stack

Python · LangChain · Google / Groq integrations · dotenv · Rich

## Setup

Install dependencies:

```bash
uv sync
```

Run the project:

```bash
uv run python main.py
```

> [!NOTE]
> This repository is a work in progress focused on learning the core steps of a RAG workflow.
