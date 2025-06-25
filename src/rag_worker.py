"""
rag_worker.py
-------------
ZeroMQ SUB -> PUB bridge that exposes the Retrieval-Augmented-Generation (RAG)
pipeline over lightweight field comms.

* SUB socket:  tcp://127.0.0.1:5555  (topic "query")
* PUB socket:  tcp://127.0.0.1:5556  (topic "answer")

Message format (UTF-8 strings, no frames):
    "query|<uuid>|<question>"
    "answer|<uuid>|<text>"
"""

from __future__ import annotations

import signal
import sys
import uuid
from pathlib import Path

import zmq
from llama_cpp import Llama

from retriever import retrieve

MODEL_PATH = Path("models/llama-2-7b-chat-q4_0.gguf")
CTX = zmq.Context()


def build_prompt(question: str, passages: list[str]) -> str:
    ctx = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    return (
        "You are a military doctrine assistant. Use ONLY the context provided.\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {question}\n"
        "Answer (cite passage numbers, avoid hallucination):"
    )


def main() -> None:
    if not MODEL_PATH.exists():
        sys.exit(f"LLM not found at {MODEL_PATH}. Run `make download-model` first.")

    print("[RAG-WORKER] Loading model & index ...")
    llm = Llama(model_path=str(MODEL_PATH), n_ctx=2048, n_gpu_layers=-1, verbose=False)

    sub = CTX.socket(zmq.SUB)
    sub.connect("tcp://127.0.0.1:5555")
    sub.setsockopt_string(zmq.SUBSCRIBE, "query")

    pub = CTX.socket(zmq.PUB)
    pub.bind("tcp://127.0.0.1:5556")

    def cleanup(*args):
        sub.close()
        pub.close()
        CTX.term()
        print("\n[RAG-WORKER] Shutdown.")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)

    print("[RAG-WORKER] Ready. Waiting for queries ...")
    while True:
        msg = sub.recv_string()
        _, qid, question = msg.split("|", 2)

        passages = [r["text"] for r in retrieve(question, k=3)]
        prompt = build_prompt(question, passages)
        out = llm(prompt, max_tokens=256, stop=["</s>"])
        answer = out["choices"][0]["text"].strip()

        pub.send_string(f"answer|{qid}|{answer}")
