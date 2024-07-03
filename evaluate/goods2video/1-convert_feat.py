#!/usr/bin/env python
# -*- coding=utf8 -*-
from tqdm import tqdm
import json
import numpy as np
import sys
import os

query_path = sys.argv[1]
doc_path = sys.argv[2]
dim = int(sys.argv[3])
save_dir = sys.argv[4]

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

data = open(query_path).readlines()
x = np.zeros((len(data), dim))
print(x.shape)
i = 0
for line in tqdm(data):
    l = line.strip().split('\t')
    feat = json.loads('['+l[1]+']')
    x[i,:] = feat
    i+=1
np.save(os.path.join(save_dir, "query.npy"),x)

data = open(doc_path).readlines()
x = np.zeros((len(data), dim))
print(x.shape)
i = 0
for line in tqdm(data):
    l = line.strip().split('\t')
    feat = json.loads('['+l[1]+']')
    x[i,:] = feat
    i+=1
np.save(os.path.join(save_dir, "doc.npy"),x)
