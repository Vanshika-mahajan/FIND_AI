# Data Handling Policy

FIND AI treats company and customer documents as sensitive by default.

## Approved Data

Approved data may include public product notes, internal policies, onboarding
guides, and demo customer documents that have been cleared for testing.

## Restricted Data

Restricted data includes passwords, API keys, payment information, private
customer records, personal identity documents, and confidential legal material.
Restricted data should not be added to the demo knowledge base.

## Local Development

Developers should keep `.env` files local and avoid committing generated vector
databases. The `.gitignore` file excludes `.env` and `preprocessed_db/`.

## Model Usage

The default setup uses Ollama for local generation. If hosted model providers
are added later, users must review what data may leave the local environment.

## Incident Response

If restricted data is accidentally committed or indexed, stop using the affected
index, remove the document, rotate any exposed secrets, and rebuild the index.
