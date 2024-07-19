import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.torchvideotransforms import video_transforms, volume_transforms
from datasets.cross_domain_emb_v2 import CrossDomainDistributedSampler, CrossDomainEmbDataset
from datasets.cross_domain_emb import CrossDomainEmbDatasetV1
from utils.utils import CollateFn

def get_transforms(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = video_transforms.Compose([
        video_transforms.RandomResizedCrop(args.image_size, scale=(0.64, 1.0)),
        video_transforms.RandomHorizontalFlip(),
        volume_transforms.ClipToTensor2(),
        normalize
    ])

    val_transform = video_transforms.Compose([
        video_transforms.Resize((args.image_size, args.image_size)),
        video_transforms.CenterCrop(args.image_size),
        volume_transforms.ClipToTensor2(),
        normalize
    ])
    
    return train_transform, val_transform

def create_dataset(args, transform, is_train):
    if is_train:
        dataset = CrossDomainEmbDataset(
            args.train_file,
            args.clip_length,
            args.goods_img_root,
            args.goods_text_root,
            args.photo_img_root,
            args.photo_text_root,
            args.live_img_root,
            args.live_text_root,
            args.frameid_root,
            shuffle=False,
            transform=transform,
            return_img_id=True,
            cls_file=args.cls_file,
        )
    else:
        dataset = CrossDomainEmbDatasetV1(
            args.test_file,
            args.clip_length,
            args.goods_img_root,
            args.goods_text_root,
            args.photo_img_root,
            args.photo_text_root,
            args.live_img_root,
            args.live_text_root,
            shuffle=False,
            transform=transform,
            return_img_id=True,
            is_train=False,
        )
    return dataset

def create_sampler(dataset, args, is_train):
    if is_train:
        return CrossDomainDistributedSampler(dataset, args.train_file, args.batch_size)
    else:
        return torch.utils.data.distributed.DistributedSampler(dataset)

def create_loader(dataset, sampler, args, is_train):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=CollateFn(),
    )
