import numpy as np, faiss

# 2 k × 384‑d vectors
dim, n = 384, 2_000
np.random.seed(0)
data = np.random.random((n, dim)).astype("float32")

index = faiss.IndexFlatL2(dim)
index.add(data)

# query first 5 vectors for 5 NNs
D, I = index.search(data[:5], 5)
print("FAISS smoke test OK - nearest-neighbour indices:")
print(I)
