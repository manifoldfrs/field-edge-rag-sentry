"""
zmq_query.py --> A client that sends questions to the RAG worker and waits for the answer.
"""

from __future__ import annotations

import argparse
import time
import uuid

import zmq

CTX = zmq.Context()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", help="Question to ask the RAG worker")
    args = ap.parse_args()

    qid = str(uuid.uuid4())

    pub = CTX.socket(zmq.PUB)
    pub.bind("tcp://127.0.0.1:5555")

    sub = CTX.socket(zmq.SUB)
    sub.connect("tcp://127.0.0.1:5556")
    sub.setsockopt_string(zmq.SUBSCRIBE, f"answer|{qid}")

    # Give sockets a moment to join
    time.sleep(0.2)

    pub.send_string(f"query|{qid}|{args.question}")
    print("[CLIENT] Sent:", args.question)

    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    if dict(poller.poll(10_000)):  # 10-second timeout
        msg = sub.recv_string()
        _, _, answer = msg.split("|", 2)
        print("\n=== Answer ===\n")
        print(answer)
    else:
        print("Timeout: no answer received.")

    pub.close()
    sub.close()
    CTX.term()


if __name__ == "__main__":
    main()
