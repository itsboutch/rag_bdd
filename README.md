ChatGPT a dit :
# RAG Pipeline for Hospital Data Extraction

This project transforms unstructured hospital data from PDF documents into structured, machine-readable formats using a Retrieval-Augmented Generation (RAG) workflow powered by the OpenAI API.

---

## Features

- Extract text from unstructured clinical and administrative PDFs
- Generate vector embeddings using **text-embedding-3-small**
- Store and query embeddings with **ChromaDB**
- Use **GPT-4o-mini** for context-aware structured output generation
- Export results as **CSV** or **JSON**
- Post-process and enrich data using SQL transformations

---

## Pipeline Overview

### 1. Document Ingestion

- Parse and extract text from PDF files  
- Clean and preprocess extracted content  
- Chunk documents into smaller segments  
- Assign metadata (e.g. document name, page index)

### 2. Embedding & Vector Storage

- Generate embeddings for each text chunk using **llm-extract-embedding-3**  
- Store embeddings in **ChromaDB** for efficient similarity search and retrieval

### 3. RAG Querying

- Retrieve the most relevant chunks from **ChromaDB** based on the user query  
- Provide retrieved context to **GPT-4o-mini**  
- Generate structured answers tailored to specific information needs  
- Support for **CSV** and **JSON** formatted outputs

### 4. Post-Processing

- Clean and normalize the structured outputs  
- Enrich data using SQL transformations  
- Prepare final datasets for analytics or downstream applications






