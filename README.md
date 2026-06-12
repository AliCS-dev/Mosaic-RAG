# Mosaic-RAG

Mosaic-RAG is an experimental  modular RAG framework.

The current prototype separates the RAG pipeline into independent services:

- Retriever Service
- Generator Service

The aim is to explore a service-oriented RAG architecture where each component can be developed, tested, replaced, and deployed independently.

## Current Architecture

```text
User Query
   ↓
Retriever Service
   ↓
Retrieved Documents
   ↓
Generator Service
   ↓
Generated Answer
