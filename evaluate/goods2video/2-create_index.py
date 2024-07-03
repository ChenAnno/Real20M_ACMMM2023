import faiss
import numpy as np
import sys
import os

feat_dir = sys.argv[1]

feat = np.load(os.path.join(feat_dir, "doc.npy"))
feat = np.array(feat, dtype=np.float32)
print(feat.shape)
print("create index")
dim = feat.shape[1]
cpu_index = faiss.IndexFlatL2(dim)
#cpu_index = index = faiss.IndexHNSWFlat(1024, 32)
cpu_index.add(feat)
faiss.write_index(cpu_index, os.path.join(feat_dir, "doc.index"))
