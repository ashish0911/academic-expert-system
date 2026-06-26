# Academic Expert System

An intelligent research assistant that transforms a simple topic into a deep-dive knowledge base. This app doesn't just search for papers; it reads them, understands their content, and provides an interactive "Expert Chat" interface with clickable academic citations.

## Overview

Researching a new field is often overwhelming. This tool automates the "grunt work" by:

1. **Expanding Queries:** Using LLM to turn a broad interest into 3 precise scientific search queries.
2. **Autonomous Harvesting:** Automatically fetching the top papers from the **arXiv** repository.
3. **Content Ingestion:** Downloading PDFs, extracting text, and breaking them into semantic chunks.
4. **Expert Interface:** Providing a RAG-powered chat where answers are backed by real text snippets and **clickable in-text citations**.

---

## Project Structure

The repository is organized following professional modular standards:

```text
paper-expert-system/
├── .env                # API Keys (git-ignored)
├── .gitignore          # Rules for Git
├── requirements.txt    # Python dependencies
├── Readme.md           # Documentation
├── models.py           # Pydantic data schemas
├── utils.py            # PDF parsing & UI formatting helpers
├── engine.py           # The LangChain RAG pipeline
└── app.py              # Streamlit Frontend

```

---

## How to Run It

### 1. Prerequisites

Ensure you have **Python 3.10+** installed and an **OpenAI API Key**.

### 2. Clone and Install

Navigate to your project folder and run:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

### 3. Set Up Secrets

Create a file named `.env` in the root directory and add your key:

```text
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxx

```

### 4. Launch the App

```bash
docker compose up -d
streamlit run app.py

```

---

## Key Insights & Value

What makes this system better than a standard search bar?

* **Semantic Retrieval vs. Keyword Search:** Standard searches look for exact words. This system uses **Vector Embeddings**, meaning if you search for "Feline behavior," it will find papers about "Cat biology" even if the word "feline" never appears.
* **The "Hallucination" Guard:** Most LLMs "make up" facts. By using **RAG (Retrieval-Augmented Generation)**, the LLM is restricted to only using the text within the 6 downloaded papers. If the answer isn't there, it won't invent it.
* **Programmatic Citations:** Our "Bulletproof Renderer" ensures that every citation `[1]` is a verified link. It maps the LLM's logic back to our Python-controlled source list, so a link never breaks and never points to the wrong paper.

---

## Technology Stack

* **Interface:** [Streamlit](https://streamlit.io/)
* **Orchestration:** [LangChain](https://www.langchain.com/)
* **Database:** [ArXiv API](https://www.google.com/search?q=https://arxiv.org/help/api/index)
* **Vector Store:** [FAISS]()
* **LLM:** OpenAI GPT-5-mini
* **Embeddings:** OpenAI `text-embedding-3-small`