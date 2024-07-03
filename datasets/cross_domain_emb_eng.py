import os
import random
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


def is_chinese(string):
    """
    检查整个字符串是否包含中文
    :param string: 需要检查的字符串
    :return: bool
    """
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


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


def get_indices(cate_group, n):
    indices_list = []
    for cate_list in cate_group:
        random.shuffle(cate_list)
        for i in range(0, len(cate_list), n):
            indices_list.append(cate_list[i: i + n])

    random.shuffle(indices_list)
    indices = []
    for tmp in indices_list:
        indices.extend(tmp)
    return indices


def get_indices_for_hard_sample(type2datalist, level, n):
    indices_list = []
    for key in type2datalist:  # type->level->cluster
        cluster_dict = type2datalist[key][level]
        cluster_keys = list(cluster_dict.keys())
        random.shuffle(cluster_keys)
        for cluster in cluster_keys:
            index_list = type2datalist[key][level][cluster]
            batch_num = int(len(index_list) / n)
            for i in range(batch_num):
                indices_list.append(index_list[i * n: (i + 1) * n])

    random.shuffle(indices_list)
    indices = []
    for tmp in indices_list:
        indices.extend(tmp)
    return indices


class CrossDomainEmbDatasetV1(data.Dataset):

    def __init__(self, ann_file, len_clip,
                 goods_image_root, goods_text_root,
                 photo_image_root, photo_text_root,
                 live_image_root, live_text_root,
                 shuffle=False, transform=None, return_img_id=False, is_train=True):
        self.ann_file = os.path.expanduser(ann_file)
        self.goods_image_root = os.path.expanduser(goods_image_root)
        self.goods_text_root = os.path.expanduser(goods_text_root)
        self.photo_image_root = os.path.expanduser(photo_image_root)
        self.photo_text_root = os.path.expanduser(photo_text_root)
        self.live_image_root = os.path.expanduser(live_image_root)
        self.live_text_root = os.path.expanduser(live_text_root)
        self.return_img_id = return_img_id
        self.transform = transform
        self.len_clip = len_clip
        self.is_train = is_train

        self.train_list = []
        for line in open(self.ann_file, 'r'):
            pid, pid_source, query_text, img_path_list, doc_txt_path = line.strip().split('\t')
            self.train_list.append((pid, pid_source, query_text, img_path_list, doc_txt_path))
        self.train_num = len(self.train_list)
        print("train file: {}".format(ann_file))
        print("total image num: {}".format(self.train_num))

        self.type2datalist = {}
        with open(self.ann_file) as fr:
            lines = fr.readlines()
            for i, line in tqdm(enumerate(lines)):
                values = line.strip().split("\t")
                pid, pid_source, query_text, img_path_list, doc_txt_path = values
                if pid_source not in self.type2datalist:
                    self.type2datalist[pid_source] = []
                self.type2datalist[pid_source].append(i)

    def get_text_path(self, path, pid_source):
        if pid_source == 'goods':
            txt_path = os.path.join(self.goods_text_root, path.replace(".txt", "_eng.txt"))
        elif pid_source == 'photo':
            txt_path = os.path.join(self.photo_text_root, path.replace(".txt", "_eng.txt"))
        elif pid_source == 'live':
            txt_path = os.path.join(self.live_text_root, path)
        return txt_path

    def get_img_path(self, path, pid_source):
        if pid_source == 'goods':
            img_full_path = os.path.join(self.goods_image_root, path)
        elif pid_source == 'photo':
            img_full_path = os.path.join(self.photo_image_root, path)
        elif pid_source == 'live':
            img_full_path = os.path.join(self.live_image_root, path)
        return img_full_path

    def __getitem__(self, index):
        while True:
            pid, pid_source, query_text, img_path_list, doc_txt_path = self.train_list[index]
            title = ''
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
                        if len(lines) > 1 and lines[1] != 'null' and random.uniform(0, 1) > 0.2:  # 0.5
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

            img_paths = img_path_list.split(",")
            if pid_source == 'goods':
                clip_length = 1
            else:
                clip_length = self.len_clip
            L = len(img_paths) / clip_length
            if self.is_train:
                seed = [int(random.uniform(i * L, (i + 1) * L)) for i in range(clip_length)]
            else:
                seed = [int((2 * i + 1) * L / 2) for i in range(clip_length)]
                # TODO +封面
                if pid_source != 'goods':
                    pass
                    # seed = [0] + seed[:-1]
                    # random_element = random.choice(seed)
                    # seed.remove(random_element)
                    # seed = [0] + seed
                    # seed = [0] + seed
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
                datalist = self.type2datalist[pid_source]
                index = random.sample(datalist, 1)[0]
                continue
            while len(imgs) < clip_length:
                imgs.append(imgs[-1])
            break

        if self.transform is not None:
            imgs = self.transform(imgs)

        title = title[:70]
        if is_chinese(title):
            title = ''

        if not self.return_img_id:
            return query_text, title, imgs
        else:
            return query_text, title, imgs, pid

    def __len__(self):
        return self.train_num


class CrossDomainDistributedSamplerV1(DistributedSampler):
    def __init__(self, dataset, sample_info_file, batch_size, num_replicas=None, rank=None, shuffle=True, seed=0,
                 drop_last=False):
        super(CrossDomainDistributedSamplerV1, self).__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

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

        self.type2datalist = {}
        with open(sample_info_file) as fr:
            lines = fr.readlines()
            for i, line in tqdm(enumerate(lines)):
                values = line.strip().split("\t")
                pid, pid_source, query_text, img_path_list, doc_txt_path = values
                if pid_source not in self.type2datalist:
                    self.type2datalist[pid_source] = []
                self.type2datalist[pid_source].append(i)
        tmp_dict = {}
        self.total_size = 0
        for key, values in self.type2datalist.items():
            num = int(len(values) / self.batch_size / self.num_replicas)
            values = values[: num * self.batch_size * self.num_replicas]
            tmp_dict[key] = values
            self.total_size += len(values)
        self.type2datalist = tmp_dict

        self.num_samples = int(self.total_size / self.num_replicas)
        print("self.num_samples: {}, self.total_size: {}".format(self.num_samples, self.total_size))
        for key, values in self.type2datalist.items():
            print("type: {}, num: {}".format(key, len(values)))
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        random.seed(self.seed + self.epoch)
        indices = get_indices(list(self.type2datalist.values()), self.batch_size)
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank * self.num_samples: (self.rank + 1) * self.num_samples]
        assert len(indices) == self.num_samples

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


class CrossDomainDistributedSampler_HardSampleV1(DistributedSampler):
    def __init__(self, dataset, sample_info_file, batch_size, sample_info_file_2, num_replicas=None, rank=None,
                 shuffle=True, seed=0, drop_last=False):
        super(CrossDomainDistributedSampler_HardSampleV1, self).__init__(dataset, num_replicas, rank, shuffle, seed)

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

        self.query2cluster_info = {}
        print("load clustering info...")
        with open(sample_info_file_2) as fr:
            lines = fr.readlines()
            for line in tqdm(lines):
                values = line.strip().split("\t")
                self.query2cluster_info[values[0]] = values[1:]

        self.type2datalist = {}
        self.index2cluster_info = {}
        not_find_cluster = 0
        with open(sample_info_file) as fr:
            lines = fr.readlines()
            for i, line in tqdm(enumerate(lines)):
                values = line.strip().split("\t")
                pid, pid_source, query_text, img_path_list, doc_txt_path = values
                if pid_source not in self.type2datalist:
                    self.type2datalist[pid_source] = []
                self.type2datalist[pid_source].append(i)

                if query_text not in self.query2cluster_info:
                    not_find_cluster += 1
                    self.index2cluster_info[i] = ['1', '1', '1']
                else:
                    self.index2cluster_info[i] = self.query2cluster_info[query_text]
        print("not_find_cluster: ", not_find_cluster)

        tmp_dict = {}
        self.total_size = 0
        for key, values in self.type2datalist.items():
            num = int(len(values) / self.batch_size / self.num_replicas)
            values = values[: num * self.batch_size * self.num_replicas]
            tmp_dict[key] = values
            self.total_size += len(values)
        self.type2datalist = tmp_dict

        tmp_dict = {}
        for key, values in self.type2datalist.items():
            if key not in tmp_dict:
                tmp_dict[key] = {}
            for index in values:
                cluster_info = self.index2cluster_info[index]  # level0, level1, level2
                for level, cluster in enumerate(cluster_info):
                    if level not in tmp_dict[key]:
                        tmp_dict[key][level] = {}
                    if cluster not in tmp_dict[key][level]:
                        tmp_dict[key][level][cluster] = []
                    tmp_dict[key][level][cluster].append(index)
        self.type2datalist = tmp_dict

        self.num_samples = int(self.total_size / self.num_replicas)
        print("self.num_samples: {}, self.total_size: {}".format(self.num_samples, self.total_size))
        for key, level_cluster_dict in self.type2datalist.items():
            for level in level_cluster_dict.keys():
                print("key: {}, level: {}, cluster: {}".format(key, level, len(list(level_cluster_dict[level].keys()))))
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        random.seed(self.seed + self.epoch)
        rnd = random.uniform(0, 1)
        print('random value: {}'.format(rnd))
        if rnd <= 1.0:
            indices = get_indices_for_hard_sample(self.type2datalist, 0, self.batch_size)  # level 0
        # elif rnd <= 1.0:
        #    indices = get_indices_for_hard_sample(self.type2datalist, 1, self.batch_size) # level 1
        # elif rnd <= 1.0:
        #    indices = get_indices_for_hard_sample(self.type2datalist, 2, self.batch_size) # level 2

        print('indices: {}, self.total_size: {}'.format(len(indices), self.total_size))
        # assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank * self.num_samples: (self.rank + 1) * self.num_samples]
        print('indices: {}, self.total_size: {}'.format(len(indices), self.total_size))

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


if __name__ == '__main__':
    pass
