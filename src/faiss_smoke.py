#!/usr/bin/env python3
import numpy as np, faiss

dim, n = 384, 2_000  # 2 k × 384‑d vectors
np.random.seed(0)  # deterministic demo
data = np.random.random((n, dim)).astype("float32")

index = faiss.IndexFlatL2(dim)  # exact L2 search
index.add(data)

D, I = index.search(data[:5], 5)  # query first 5 vectors for 5 NNs
print("FAISS smoke test OK - nearest-neighbour indices:")
print(I)
