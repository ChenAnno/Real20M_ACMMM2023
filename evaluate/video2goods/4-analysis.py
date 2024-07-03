import os, sys
import random
import numpy as np
from tqdm import tqdm


def analysis(res_np, dis_np, query_path_label, doc_path_label, gt_path):
    res = np.load(res_np)
    #dis = np.load(dis_np)

    query_path = []
    with open(query_path_label, 'r') as fr:
        lines = fr.readlines()
        for line in tqdm(lines):
            path = line.strip().split("\t")[0]
            query_path.append(path)

    doc_path = []
    with open(doc_path_label, 'r') as fr:
        lines = fr.readlines()
        for line in tqdm(lines):
            path = line.strip().split("\t")[0]
            doc_path.append(path)

    query2gt = {}
    with open(gt_path) as fr:
        lines = fr.readlines()
        for line in tqdm(lines):
            values = line.strip().split("\t")
            queryid = values[0].split("/")[-1].split("_")[0]
            for doc in values[1].split(","):
                docid = doc.split("/")[-1].split(".")[0]
                if queryid not in query2gt:
                    query2gt[queryid] = []
                query2gt[queryid].append(docid)

    for topk in [10, 20, 50, 100, 200, 500, 1000]:
        recall_ratio_list = []
        for i, r in enumerate(res):
            #d_list = dis[i]
            key = query_path[i]
            gt_list = query2gt[key]
            recall_num = 0
            recall_total = 0
            for j, idx in enumerate(r[:topk]):
                itemid = doc_path[idx]
                if itemid in gt_list:
                    recall_num += 1
            recall_total += len(gt_list)
            if recall_total == 0:
                continue
            recall_ratio_list.append(float(recall_num) / float(recall_total))
        print("recall@{}: {}".format(topk, sum(recall_ratio_list) / len(recall_ratio_list)))


dir_name = sys.argv[1]
query_feat_path = sys.argv[2]
doc_feat_path = sys.argv[3]
gt_path = sys.argv[4]
res_np = os.path.join(dir_name, "search_res.npy")
dis_np = os.path.join(dir_name, "search_res_dis.npy")
analysis(res_np, dis_np, query_feat_path, doc_feat_path, gt_path)
