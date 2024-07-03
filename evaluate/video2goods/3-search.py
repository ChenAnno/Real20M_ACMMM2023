import faiss # use gpu version
import os, sys
import numpy as np

def search(query_path, index_path, save_path, topk):
    query_feat = np.load(query_path)
    query_feat = np.array(query_feat, dtype=np.float32)
    print("query shape: ", query_feat.shape)

    print("start to load index")
    cpu_index = faiss.read_index(index_path)
    print("load index successfully.")

    #ngpus = faiss.get_num_gpus()
    #print("number of GPUs:", ngpus)
    #gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    #D, I = gpu_index.search(query_feat, topk) # actual search
    D, I = cpu_index.search(query_feat, topk) # actual search

    dis_save_path = save_path.replace("search_res", "search_res_dis")
    np.save(save_path, I)
    np.save(dis_save_path, D)
    return I, D

dir_name = sys.argv[1]
topk = int(sys.argv[2])
query_path = os.path.join(dir_name, "query.npy")
index_path = os.path.join(dir_name, "doc.index")
save_path = os.path.join(dir_name, "search_res")
search(query_path, index_path, save_path, topk)
