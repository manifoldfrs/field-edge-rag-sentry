# field‑edge‑rag‑sentry

Prove—fast—that a single M‑series MacBook can run a **field‑edge Retrieval‑Augmented‑Generation (RAG) system** end‑to‑end:

- local LLM inference (llama.cpp + Metal)
- on‑device LoRA fine‑tuning
- vector search (FAISS)
- real‑time vision via CoreML
- airtight evaluation that catches hallucinations before your interviewer does.

Think of it as a _micro‑demo_: minimal surface area, maximum evidence.

---

## 1 — Why this matters

| Field constraint                       | Design response                                                             |
| -------------------------------------- | --------------------------------------------------------------------------- |
| **No cloud** in contested environments | Everything runs on‑device; zero external calls.                             |
| **Limited GPU VRAM** (16 GB)           | Quantized 7 B LLM + LoRA adapters ≤ 6 GB.                                   |
| **Interviewer skepticism**             | Ragas + unit tests deliver hard metrics (precision, faithfulness, latency). |
| **Edge sensors** (camera)              | YOLO‑v5s → CoreML leverages Apple Neural Engine; keeps GPU free for LLM.    |

---

## 2 — Core features

- **Metal‑accelerated LLM** – `make llama` hits 15‑25 tok/s on 7‑13 B models.
- **LoRA fine‑tuning** – fits in RAM, field‑upgrade‑able with fresh data.
- **FAISS index** – blister‑fast search for <20 k chunks.
- **ZeroMQ comms** – lightweight PUB/SUB, no Docker networking pain.
- **Hard‑mode eval** – Ragas metrics + adversarial "no‑answer" prompts.
- **Tight codebase** – no LangChain bloat; every line is visible and debuggable.

---

## 3 — Quick‑start (macOS 14 / Apple Silicon)

```bash
# Install system dependencies
make deps     # installs cmake, faiss via Homebrew

# Build llama.cpp with Metal acceleration
make llama    # clones & compiles llama.cpp with Metal backend

# Run FAISS vector search smoke test
make test-faiss

# Or run everything at once
make          # builds llama.cpp + runs FAISS test

# Install Python dependencies (if needed)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
