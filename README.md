RAG Pipeline for Hospital Data Extraction

This project processes unstructured hospital data stored in multiple PDF files and converts it into structured information. The workflow uses a Retrieval-Augmented Generation (RAG) approach powered by the OpenAI API.

Overview

Extract text from hospital PDFs containing unstructured clinical and administrative information.

Build a retrieval pipeline using GPT-4o-mini for generation and llm-extract-embedding-3 for vector embeddings.

Store embeddings in ChromaDB to enable efficient similarity search and context retrieval.

Query the LLM to produce structured outputs (CSV or JSON) based on user-defined information needs.

Pipeline Components

Document Ingestion

PDF parsing and text preprocessing.

Chunking and metadata assignment.

Embedding & Vector Storage

Text chunks embedded using llm-extract-embedding-3.

Embeddings stored in ChromaDB for fast vector search.

LLM Querying

RAG workflow that retrieves relevant document chunks.

GPT-4o-mini generates structured answers tailored to the query.

Outputs formatted as CSV or JSON.

Post-Processing

Structured outputs cleaned and enriched using SQL transformations.

Data prepared for analytics or downstream applications.

Objectives

Transform unstructured hospital documents into machine-readable datasets.

Enable reliable and context-aware information extraction using RAG.

Provide a modular foundation for future data cleaning, enrichment, and analysis steps.
