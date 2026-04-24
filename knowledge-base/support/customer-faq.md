# Customer FAQ

This FAQ describes common customer questions for FIND AI.

## Why are answers limited to the knowledge base?

The assistant is designed to answer from approved company documents. This helps
reduce hallucinations and makes answers easier to verify.

## What should I do when the assistant says it does not know?

Add or update the relevant markdown document, then rebuild the index. If the
information already exists, improve the document heading or wording so retrieval
can find it more easily.

## How often should I rebuild the index?

Rebuild the index whenever documents are added, deleted, or meaningfully changed.

## Can the assistant answer from PDFs?

The current demo focuses on markdown files. PDF support is listed as a roadmap
item.

## Why does the first answer take longer?

The first answer can be slower because the local Ollama model may need to load
into memory.
