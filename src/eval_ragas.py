#!/usr/bin/env python3
"""
eval_ragas.py
-------------
Evaluate the end-to-end RAG pipeline with Ragas metrics.

Expected eval set: `data/eval_set.jsonl`, one JSON per line:
    {"question": "...", "answer": "ground-truth answer text"}

Run:
    $ make eval              # alias in Makefile
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import List, Tuple

from llama_cpp import Llama
from ragas.metrics import answer_correctness, context_precision, faithfulness
from ragas.metrics.aggregators import mean_agg
from ragas.metrics.base import RagasMetric
from ragas.schema import Dataset, GroundTruth
from tqdm import tqdm

from generate import build_prompt  # reuse prompt fn
from retriever import retrieve

MODEL_PATH = Path("models/llama-2-7b-chat-q4_0.gguf")
EVAL_PATH = Path("data/eval_set.jsonl")

METRICS: List[RagasMetric] = [
    context_precision,
    faithfulness,
    answer_correctness,
]


def load_eval() -> List[Tuple[str, str]]:
    if not EVAL_PATH.exists():
        raise SystemExit(
            f"Eval set not found at {EVAL_PATH}. "
            "Provide 25 Q-A pairs in JSONL format."
        )
    rows = []
    with EVAL_PATH.open() as f:
        for line in f:
            obj = json.loads(line)
            rows.append((obj["question"], obj["answer"]))
    return rows


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit("LLM model missing — run `make download-model` first.")

    print("[EVAL] Loading model …")
    llm = Llama(model_path=str(MODEL_PATH), n_ctx=2048, n_gpu_layers=-1, verbose=False)

    qas = load_eval()
    gt = GroundTruth(questions=[q for q, _ in qas], answers=[a for _, a in qas])
    dataset = Dataset.from_groundtruth(gt)

    answers, contexts, latencies = [], [], []
    print(f"[EVAL] Running {len(dataset)} queries …")

    for q in tqdm(dataset.questions):
        t0 = perf_counter()
        retrieved = retrieve(q, k=3)
        ctx_passages = [r["text"] for r in retrieved]
        prompt = build_prompt(q, ctx_passages)
        out = llm(prompt, max_tokens=256, stop=["</s>"])
        answer = out["choices"][0]["text"].strip()

        answers.append(answer)
        contexts.append(" ".join(ctx_passages))
        latencies.append(perf_counter() - t0)

    dataset.predicted_answers = answers
    dataset.contexts = contexts

    results = {m.name: m.aggregate(m.score(dataset)) for m in METRICS}
    results["avg_latency_s"] = sum(latencies) / len(latencies)

    print("\n=== Ragas metrics ===")
    for k, v in results.items():
        print(f"{k:20s}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()
