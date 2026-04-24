# Product Overview

FIND AI is a company knowledge assistant that answers employee questions using
approved internal documents.

## Core Workflow

1. Add markdown documents to the knowledge base.
2. Run ingestion to chunk and embed the documents.
3. Store vectors and metadata in ChromaDB.
4. Ask a question in the Streamlit app.
5. Retrieve relevant document chunks.
6. Generate an answer with the local Ollama model.
7. Show consulted sources so users can verify the answer.

## Primary Users

- Employees who need quick answers from company documents.
- Customer success teams who need accurate product and process information.
- Operations teams who maintain policies and onboarding guides.
- Founders or managers who want a lightweight internal assistant.

## Product Principles

- The assistant should answer only from retrieved context.
- Every answer should make source documents visible.
- Users should be able to rebuild the index after document updates.
- Local model support is preferred for privacy and cost control.

## Current Interface

The product uses Streamlit for the user interface. The UI includes a sidebar for
model and retrieval settings, a chat area, status metrics, and a source panel.
