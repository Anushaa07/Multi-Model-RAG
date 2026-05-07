

# Multi-Modal Document Intelligence using RAG

## Overview
This project implements a **Multi-Modal Retrieval-Augmented Generation (RAG) system** that can process complex documents (PDFs with text, tables, images, and OCR), retrieve relevant context, and generate **context-grounded answers with citations**.

It is designed for **financial and policy reports** such as IMF Article IV reports but can be adapted to other structured documents.

---

## Features
- **Multi-modal ingestion:** text, tables, images, OCR
- **Chunking & Embeddings:** sentence-transformers embeddings, 200-word chunks with overlap
- **Vector Retrieval:** FAISS-based search
- **QA Generator:** Groq LLM integration (llama-3.1-8b-instant)
- **Citation-aware answers:** retrieved page references included
- **Interactive UI:** Streamlit app

---

## Folder Structureproject/
│── ingestion/ # PDF parsing, tables, OCR, image extraction
│── embeddings/ # Chunking, embedding generation
│── retrieval/ # FAISS index, retriever, optional reranker
│── app/ # Streamlit interface
│ └── streamlit_app.py
│── utils/ # Config and helper functions
│── requirements.txt
│── README.md
│── report.pdf # 2-page technical report
│── demo_video.mp4 # optional video demonstration


---

## Setup Instructions

1. **Clone the repository**
```bash
git clone <repo-url>
cd project


Create virtual environment

python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows


Install dependencies

pip install -r requirements.txt


Install Tesseract OCR

Windows: Download and install Tesseract OCR

Mac: brew install tesseract

Linux: sudo apt install tesseract-ocr

Set Tesseract path in streamlit_app.py

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


Run Streamlit App

streamlit run app/streamlit_app.py

Usage

Upload a PDF file

View extracted images/tables

Type a query in the input box

See retrieved snippets and final Groq-generated answer with page citations

Notes


If Groq is not configured, the system returns extractive summaries from retrieved chunks

Caches embeddings & FAISS index for faster subsequent runs

Future Work

Cross-modal reranking with vision-text embeddings

Hybrid retrieval (BM25 + FAISS)

Summarization / briefing generation

Evaluation dashboard for latency & retrieval metrics
