#!/usr/bin/env python3
"""
vision_rag.py
-------------
Web-cam object detection + Retrieval-Augmented-Generation advisor.

Whenever YOLO-v5s (exported to CoreML) detects an object, we:
    • formulate a doctrinal question about that object,
    • retrieve supporting passages from the FAISS index, and
    • ask the Llama-2-7B-Chat Q4_0 model for recommended orders.

Answers are printed to the terminal (not overlaid on the frame to keep things
light-weight).

Run:
    $ make vision-rag
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import cv2
from llama_cpp import Llama
from ultralytics import YOLO

from generate import build_prompt
from retriever import retrieve
from vision_demo import export_if_missing

MODEL_PATH = Path("models/llama-2-7b-chat-q4_0.gguf")
CACHE_TTL = timedelta(seconds=30)  # don't repeat advice for same class too often

# class_name -> (answer, last_print_time)
_advice_cache: Dict[str, tuple[str, datetime]] = defaultdict(lambda: ("", datetime.min))


def advise_for(cls_name: str, llm: Llama) -> str:
    """
    Retrieve doctrine passages & ask LLM for recommended orders
    concerning the detected object class.
    Uses a simple TTL cache to avoid spamming the model.
    """
    answer, ts = _advice_cache[cls_name]
    now = datetime.utcnow()
    if now - ts < CACHE_TTL:
        return answer  # still fresh

    question = f"What orders should be given when encountering '{cls_name}' on the battlefield?"
    passages = [r["text"] for r in retrieve(question, k=3)]
    prompt = build_prompt(question, passages)
    out = llm(prompt, max_tokens=256, stop=["</s>"])
    answer = out["choices"][0]["text"].strip()

    _advice_cache[cls_name] = (answer, now)
    stamped = now.strftime("%H:%M:%S")
    print(f"\n[{stamped}] OBJECT: {cls_name.upper()}\n{answer}\n{'-'*60}")
    return answer


def main() -> None:
    if not MODEL_PATH.exists():
        sys.exit("LLM model missing - run `make download-model` first.")

    print("[VISION-RAG] Loading CoreML detector & Llama model ...")
    coreml_path = export_if_missing()
    detector = YOLO(str(coreml_path))
    llm = Llama(model_path=str(MODEL_PATH), n_ctx=2048, n_gpu_layers=-1, verbose=False)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("ERROR: could not open webcam (permissions?).")

    print("[VISION-RAG] Ready. Press q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("WARNING: failed to read frame; exiting.")
            break

        results = detector.predict(source=frame, verbose=False)[0]
        classes_in_frame = {detector.names[int(b.cls[0])] for b in results.boxes}

        for cls_name in classes_in_frame:
            # fire-and-forget; user can read answers in terminal
            advise_for(cls_name, llm)

        cv2.imshow("YOLOv5s (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
