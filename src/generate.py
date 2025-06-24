"""
generate.py
-----------
End-to-end QA:
    1. retrieve top-k passages from FAISS
    2. craft prompt
    3. run llama.cpp (gguf, q4_0) and print answer

Example
-------
$ python src/generate.py "Describe the principles of mission-type orders."
"""

import argparse
from pathlib import Path

from llama_cpp import Llama

from retriever import retrieve

MODEL_PATH = Path("models/llama-2-7b-chat-q4_0.gguf")


def build_prompt(question: str, passages: list[str]) -> str:
    context = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    return (
        "You are a military doctrine assistant. Use ONLY the context provided.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer (cite passage numbers, avoid hallucination):"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", help="User question")
    ap.add_argument("-k", type=int, default=3, help="top-k passages")
    ap.add_argument("-max_tokens", type=int, default=256, help="generation limit")
    args = ap.parse_args()

    results = retrieve(args.question, k=args.k)
    passages = [r["text"] for r in results]

    prompt = build_prompt(args.question, passages)
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Model not found at {MODEL_PATH}. "
            "Download a 7B q4_0 gguf and place it there."
        )

    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=2048,
        n_gpu_layers=-1,  # Metal offload
        logits_all=False,
        verbose=False,
    )

    output = llm(prompt, max_tokens=args.max_tokens, stop=["</s>"])
    answer = output["choices"][0]["text"].strip()
    print("\n=== Answer ===\n")
    print(answer)


if __name__ == "__main__":
    main()
