import os
import sys
import random
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import math
from tqdm import tqdm
from transformers import BertTokenizer, AutoTokenizer


def image_loader(filename):
    try:
        img = Image.open(filename).convert('RGB')
        width, height = img.size
        if width <= 0 or height <= 0:
            raise Exception("width <= 0 or height <= 0")
        return img
    except Exception as e:
        print(e)
        return None


class GoodsClassificationDataset(data.Dataset):
    def __init__(self, ann_file, goods_image_root, goods_text_root, transform):
        self.ann_file = ann_file
        self.goods_image_root = goods_image_root
        self.goods_text_root = goods_text_root
        self.transform = transform

        self.goods_name_labels_list = []
        for line in open(self.ann_file, 'r'):
            goods_name_labels = line.strip().split('\t')
            self.goods_name_labels_list.append(goods_name_labels)

    def __len__(self):
        return len(self.goods_name_labels_list)

    def __getitem__(self, index):
        goods_name_labels = self.goods_name_labels_list[index]
        goods_name, goods_labels = goods_name_labels

        goods_image_path = os.path.join(self.goods_image_root, str(int(goods_name) % 50000), goods_name + ".jpg")
        goods_image = [image_loader(goods_image_path)]

        if goods_image[0] is None:
            goods_name_labels = self.goods_name_labels_list[index - 1]
            goods_name, goods_labels = goods_name_labels
            goods_image_path = os.path.join(self.goods_image_root, str(int(goods_name) % 50000), goods_name + ".jpg")
            goods_image = [image_loader(goods_image_path)]

        goods_text_path = os.path.join(self.goods_text_root, str(int(goods_name) % 50000), goods_name + ".txt")
        goods_text = ""
        if os.path.exists(goods_text_path):
            lines = open(goods_text_path, 'r').readlines()  # itemid, category, brand, entity, title, desc
            if len(lines) == 6:
                goods_text = lines[4].strip()
        if self.transform is not None:
            goods_image = self.transform(goods_image)

        label1, label2, label3 = goods_labels.split("---")
        if label2 == "":
            label2 = label1
        if label3 == "":
            label3 = label2

        return goods_name, goods_image, goods_text, label1, label2, label3


class CollateFnTestClassification(object):
    def __init__(self):
        # self.tokenizer1 = AutoTokenizer.from_pretrained('hfl/rbt3')
        self.tokenizer2 = AutoTokenizer.from_pretrained('hfl/rbt6')

    def __call__(self, data):
        batch_data = list(zip(*data))
        # pid, imgs, doc_text
        # batch_data[0] = torch.cat(batch_data[0], 0)
        batch_data[1] = torch.cat(batch_data[1], 0)
        batch_data[2] = self.tokenizer2(batch_data[2], return_tensors='pt', padding=True, truncation=True,
                                        max_length=60)

        return batch_data


if __name__ == '__main__':
    ann_file = "./test.txt"
    # query_dataset = Query2GoodsDataset_query(ann_file)
    # print(query_dataset[0])
