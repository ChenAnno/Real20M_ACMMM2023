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
import cv2
from tqdm import tqdm
import random
import base64


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


def title_augmentation(title):
    if (len(title) >= 10) and (random.uniform(0, 1) > 0.4):
        rnd_index = list(range(len(title)))
        random.shuffle(rnd_index)
        rnd_index = rnd_index[:int(0.2 * len(title))]
        title_aug = []
        for i, elem in enumerate(title):
            if i in rnd_index:
                rnd = random.uniform(0, 1)
                if rnd < 0.333:
                    title_aug.append(elem * 2)
                elif rnd < 0.666:
                    title_aug.append('')
                else:
                    title_aug.append(elem + ' ')
            else:
                title_aug.append(elem)
        title = ''.join(title_aug)
    return title


def get_indices(query2idlist, batch_size, num_replicas):
    # index, itemid_list, pid_list
    indices_list = []
    cate_group = {
        "goods": [],
        "photo": [],
    }
    for query, idlist in query2idlist.items():
        index, itemid_list, pid_list = idlist
        if itemid_list != 'None' and pid_list != 'None':
            rnd = random.uniform(0, 1)
            if rnd < 0.5:  # 50%选择商品
                cate_group['goods'].append((index, 'goods'))
            else:
                cate_group['photo'].append((index, 'photo'))
        elif itemid_list != 'None':
            cate_group['goods'].append((index, 'goods'))
        elif pid_list != 'None':
            cate_group['photo'].append((index, 'photo'))
        else:
            continue

    for data_type, cate_list in cate_group.items():
        random.shuffle(cate_list)
        num = int(len(cate_list) / batch_size / num_replicas)
        cate_list = cate_list[: num * batch_size * num_replicas]
        batch_num = int(len(cate_list) / batch_size)
        for i in range(batch_num):
            indices_list.append(cate_list[i * batch_size: (i + 1) * batch_size])

    random.shuffle(indices_list)
    indices = []
    for tmp in indices_list:
        indices.extend(tmp)
    return indices


class CrossDomainEmbDataset(data.Dataset):
    def __init__(self, ann_file, len_clip,
                 goods_image_root, goods_text_root,
                 photo_image_root, photo_text_root,
                 live_image_root, live_text_root,
                 frameid_root,
                 shuffle=False, transform=None, return_img_id=False, is_train=True, cls_file=None):
        self.ann_file = os.path.expanduser(ann_file)
        self.goods_image_root = os.path.expanduser(goods_image_root)
        self.goods_text_root = os.path.expanduser(goods_text_root)
        self.photo_image_root = os.path.expanduser(photo_image_root)
        self.photo_text_root = os.path.expanduser(photo_text_root)
        self.live_image_root = os.path.expanduser(live_image_root)
        self.live_text_root = os.path.expanduser(live_text_root)
        self.frameid_root = os.path.expanduser(frameid_root)
        self.return_img_id = return_img_id
        self.transform = transform
        self.len_clip = len_clip
        self.is_train = is_train

        self.pidsource2idx = {
            'goods': [],
            'photo': [],
        }
        self.train_list = []
        for i, line in enumerate(open(self.ann_file, 'r')):
            query, itemid_list, pid_list = line.strip().split('\t')
            self.train_list.append((query, itemid_list, pid_list))
            if itemid_list != 'None':
                self.pidsource2idx['goods'].append(i)
            if pid_list != 'None':
                self.pidsource2idx['photo'].append(i)

        self.train_num = len(self.train_list)
        print("train file: {}".format(ann_file))
        print("total image num: {}".format(self.train_num))

    def get_text_path(self, path, pid_source):
        if pid_source == 'goods':
            txt_path = os.path.join(self.goods_text_root, path)
        elif pid_source == 'photo':
            txt_path = os.path.join(self.photo_text_root, path)

        return txt_path

    def get_img_path(self, path, pid_source):
        if pid_source == 'goods':
            img_full_path = os.path.join(self.goods_image_root, path)
        elif pid_source == 'photo':
            img_full_path = os.path.join(self.photo_image_root, path)

        return img_full_path

    def __getitem__(self, index):
        while True:
            idx, pid_source = index
            query_text, itemid_list, pid_list = self.train_list[idx]
            query_text = query_text.rstrip()

            if pid_source == 'goods':
                pid = random.choice(itemid_list.split(","))
            elif pid_source == 'photo':
                pid = random.choice(pid_list.split(","))

            title = ''
            doc_txt_path = os.path.join(str(int(pid) % 50000), pid + '.txt')
            doc_txt_path = self.get_text_path(doc_txt_path, pid_source)
            if os.path.exists(doc_txt_path):
                lines = open(doc_txt_path, 'r').readlines()
                if pid_source == 'goods':
                    # itemid, category, brand, entity, title, desc
                    if len(lines) == 6:
                        title = lines[4].strip()
                        if self.is_train:
                            title = title_augmentation(title)
                elif pid_source == 'photo':
                    tmp = []
                    # caption, title, text
                    if self.is_train:
                        if len(lines) > 1 and lines[1] != 'null' and random.uniform(0, 1) > 0.2:
                            tmp.append(lines[1])
                        if len(lines) > 2 and lines[2] != 'null' and lines[2] != lines[1] and random.uniform(0,
                                                                                                             1) > 0.2:
                            tmp.append(lines[2][:40])
                        if len(lines) > 0 and lines[0] != 'null' and random.uniform(0, 1) > 0.2:
                            tmp.append(lines[0])
                    else:
                        if len(lines) > 1 and lines[1] != 'null':
                            tmp.append(lines[1])
                        if len(lines) > 2 and lines[2] != 'null' and lines[2] != lines[1]:
                            tmp.append(lines[2][:40])
                        if len(lines) > 0 and lines[0] != 'null':
                            tmp.append(lines[0])
                    title = '|'.join(tmp)[:80]
                elif pid_source == 'live':
                    pass
            # images
            if pid_source == 'goods':
                img_paths = [os.path.join(str(int(pid) % 50000), pid + ".jpg")]
                clip_length = 1
            else:
                line = open(os.path.join(self.frameid_root, str(int(pid) % 50000), pid + ".txt")).readline()
                img_paths = line.strip().split(",")
                clip_length = self.len_clip

            L = len(img_paths) / clip_length
            if self.is_train:
                seed = [int(random.uniform(i * L, (i + 1) * L)) for i in range(clip_length)]
            else:
                seed = [int((2 * i + 1) * L / 2) for i in range(clip_length)]
            imgs = []
            for idx in seed:
                img_path = img_paths[idx]
                img_full_path = self.get_img_path(img_path, pid_source)
                img = image_loader(img_full_path)
                if img is None:
                    continue
                imgs.append(img)
            if len(imgs) == 0:
                # index = (index + 1) % self.train_num
                # print("load data again, index: {}, path: {}".format(index, img_full_path))
                idxlist = self.pidsource2idx[pid_source]
                idx = random.sample(idxlist, 1)[0]
                index = (idx, pid_source)
                continue
            while len(imgs) < clip_length:
                imgs.append(imgs[-1])
            break

        if self.transform is not None:
            imgs = self.transform(imgs)

        if not self.return_img_id:
            return query_text, title, imgs
        else:
            return query_text, title, imgs, pid, pid_source

    def __len__(self):
        return self.train_num


class CrossDomainDistributedSampler(DistributedSampler):
    def __init__(self, dataset, sample_info_file, batch_size, num_replicas=None, rank=None, shuffle=True, seed=0,
                 drop_last=False):
        super(CrossDomainDistributedSampler, self).__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.batch_size = batch_size

        self.query2idlist = {}
        self.total_size = 0
        with open(sample_info_file) as fr:
            lines = fr.readlines()
            for i, line in tqdm(enumerate(lines)):
                self.total_size += 1
                values = line.strip().split("\t")
                query, itemid_list, pid_list = values
                self.query2idlist[query] = [i, itemid_list, pid_list]

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        random.seed(self.seed + self.epoch)
        indices = get_indices(self.query2idlist, self.batch_size, self.num_replicas)
        self.num_samples = int(len(indices) / self.num_replicas)
        # subsample
        indices = indices[self.rank * self.num_samples: (self.rank + 1) * self.num_samples]
        print("indices: {}, num_samples: {}, total_size: {}".format(len(indices), self.num_samples, self.total_size))

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class CollateFn(object):
    def __init__(self):
        self.tokenizer1 = BertTokenizer.from_pretrained('hfl/rbt3')
        self.tokenizer2 = BertTokenizer.from_pretrained('hfl/rbt6')

    def __call__(self, data):
        batch_data = list(zip(*data))
        # query_text, title_text, imgs
        batch_data[0] = self.tokenizer1(list(batch_data[0]), return_tensors='pt', padding=True, truncation=True,
                                        max_length=15)
        batch_data[1] = self.tokenizer2(list(batch_data[1]), return_tensors='pt', padding=True, truncation=True,
                                        max_length=60)
        batch_data[2] = torch.cat(batch_data[2], 0)
        return batch_data
