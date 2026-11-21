# Invoice Triage & Extraction Pipeline

Local workflow using **n8n**, **FastAPI**, **Docker Compose**, and **Ollama** (`llama3.2:3b`) to extract structured fields from receipt and invoice text.

## Stack
- **n8n** (`invoice-n8n`) — webhook trigger and field shaping
- **FastAPI** (`invoice-extractor-api`) — `/extract` endpoint on port `8000`
- **Ollama** (`llama3.2:3b`) — local LLM running on host (`host.docker.internal:11434`)
- **Docker Compose** — container orchestration
