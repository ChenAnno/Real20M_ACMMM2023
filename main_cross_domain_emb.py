import argparse
import os
import time
import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

import models
from losses import *
from datasets import *
from utils.amp import MaxClipGradScaler
from utils.utils import *
from evaluation import *

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Search Recall Training')
parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
parser.add_argument('--train_file', dest='train_file', type=str, metavar='PATH',
                    help='train file')
parser.add_argument('--test_file', dest='test_file', type=str, metavar='PATH',
                    help='test file')
parser.add_argument('--sample_info_file', dest='sample_info_file', type=str, metavar='PATH',
                    help='sample_info_file')
parser.add_argument('--output_dir', dest='output_dir', type=str, metavar='PATH',
                    help='output dir')
parser.add_argument('--train_file_part_num', default=1, type=int, metavar='N',
                    help='train_file_part_num')
parser.add_argument('--goods_img_root', dest='goods_img_root', type=str, metavar='PATH',
                    help='goods image root dir name')
parser.add_argument('--goods_text_root', dest='goods_text_root', type=str, metavar='PATH',
                    help='goods text root dir name')
parser.add_argument('--photo_img_root', dest='photo_img_root', type=str, metavar='PATH',
                    help='photo image root dir name')
parser.add_argument('--photo_text_root', dest='photo_text_root', type=str, metavar='PATH',
                    help='photo text root dir name')
parser.add_argument('--live_img_root', dest='live_img_root', type=str, metavar='PATH',
                    help='live image root dir name')
parser.add_argument('--live_text_root', dest='live_text_root', type=str, metavar='PATH',
                    help='live text root dir name')
parser.add_argument('--frameid_root', dest='frameid_root', type=str, metavar='PATH',
                    help='frameid root dir name')
parser.add_argument('--image_size', default=224, type=int, metavar='N',
                    help='image_size')
parser.add_argument('--embedding_size', default=128, type=int, metavar='N',
                    help='embedding_size')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--decay_epochs', default=30, type=int, metavar='N',
                    help='number of epochs to decay lr')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the '
                         'batch size of single GPU on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--global_step', default=0, type=int, metavar='N',
                    help='global step')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-s', '--save_freq', default=1000, type=int,
                    metavar='N', help='save model frequency (default: 10000)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--checkpoints', default='./outputs/checkpoints/checkpoints', type=str, metavar='PATH',
                    help='path to checkpoints (default: none)')
parser.add_argument('--pretrained', default='./pretrained/resnet50_vd_10w.pth', type=str, metavar='PATH',
                    help='path to pre-trained model')
parser.add_argument('--mixed_precision_training', action='store_true', help='mixed precision training')
parser.add_argument('--using_other_transforms', action='store_true', help='using_other_transforms')
parser.add_argument('--finetune', action='store_true', help='finetune')
parser.add_argument('--evaluate', action='store_true', help='evaluate')
parser.add_argument('--evaluate_live2goods', action='store_true', help='evaluate live2goods')
parser.add_argument('--evaluate_query2goods_query', action='store_true', help='evaluate query2goods')
parser.add_argument('--evaluate_query2goods_goods', action='store_true', help='evaluate query2goods')
parser.add_argument('--evaluate_goods2caption', action='store_true', help='evaluate goods2caption')
parser.add_argument('--is_xbm', action='store_true', help='is_xbm')
parser.add_argument('--clip_length', default=0, type=int, metavar='N', help='clip length')
parser.add_argument('--cls_num', default=1000, type=int, metavar='N',
                    help='the number of classifier')
parser.add_argument('--cls_file', dest='cls_file', type=str, metavar='PATH',
                    help='classifier file')

lr = 0
global_step = 0


def main():
    global global_step
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True

    try:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        dist_url = "tcp://{}:{}".format(os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])
    except KeyError:
        world_size = 1
        rank = 0
        dist_url = "tcp://127.0.0.1:12584"
    args.world_size = world_size
    args.rank = rank
    args.dist_url = dist_url
    print("=> world size:", world_size)
    print("=> rank:", rank)
    print("=> dist_url:", dist_url)

    dist.init_process_group(backend='nccl', init_method=dist_url, rank=rank, world_size=world_size)
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)

    if local_rank == 0:
        print("=> args:", args)

    if not os.path.exists(args.checkpoints) and rank is 0:
        os.makedirs(args.checkpoints)
    else:
        time.sleep(2)
    if local_rank == 0:
        args.tb_writer = SummaryWriter(os.path.join(args.checkpoints, time.strftime('%Y-%m-%d-%H-%M-%S')))
    else:
        args.tb_writer = None

    # create model
    doc_text_model = models.__dict__['RoBERTa_CLIP'](
        fp16=args.mixed_precision_training)
    doc_image_model = models.__dict__['Vitb16'](
        fp16=args.mixed_precision_training)

    doc_fusion_model = models.__dict__['DocAttentionFusionModel'](
        input_dims=[doc_text_model.output_size, doc_text_model.output_size, doc_image_model.output_size],
        emb_dim=args.embedding_size)
    print("=> backbone output size: {} {}".format(
        doc_text_model.output_size, doc_image_model.output_size))

    text_generation_model = models.__dict__['GPT2'](
        embedding_size=args.embedding_size,
        fp16=args.mixed_precision_training)

    # create loss
    text_relevance_loss_1 = TextRelevanceLoss(args.batch_size, args.embedding_size)
    text_relevance_loss_2 = TextRelevanceLoss(args.batch_size, args.embedding_size)
    text_relevance_loss_3 = TextRelevanceLoss(args.batch_size, args.embedding_size)
    ITC_ITM_criterion = ITC_ITM_loss(batch_size=args.batch_size, emb_dim=args.embedding_size)
    text_generate_criterion = TextGenerationLoss()

    # load pretrained model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.pretrained))
        loc = 'cuda:{}'.format(local_rank)
        checkpoint = torch.load(args.pretrained, map_location=loc)

    else:
        print("=> creating model from scratch ...")

    doc_text_model.cuda(local_rank)
    doc_text_model = torch.nn.parallel.DistributedDataParallel(
        module=doc_text_model, device_ids=[local_rank],
        broadcast_buffers=False,
        find_unused_parameters=True
    )
    doc_text_model.train()

    doc_image_model.cuda(local_rank)
    doc_image_model = torch.nn.parallel.DistributedDataParallel(
        module=doc_image_model, device_ids=[local_rank])
    if args.finetune:
        doc_image_model.train()
    else:
        doc_image_model.eval()

    doc_fusion_model.cuda(local_rank)
    doc_fusion_model = torch.nn.parallel.DistributedDataParallel(
        module=doc_fusion_model, device_ids=[local_rank])
    doc_fusion_model.train()

    text_generation_model.cuda(local_rank)
    text_generation_model = torch.nn.parallel.DistributedDataParallel(
        module=text_generation_model, device_ids=[local_rank])
    text_generation_model.train()

    text_lr_factor = 0.01
    image_lr_factor = 1e-3
    fusion_lr_factor = 1
    video_fusion_factor = 1e-2  # 0.01 * 0.01
    text_generation_factor = 1e-3
    print("=> text/image lr factor: {}/{}".format(text_lr_factor, image_lr_factor))

    text_optimizer = torch.optim.Adam(
        params=[{'params': doc_text_model.parameters()}],
        lr=args.lr * text_lr_factor)
    image_optimizer = torch.optim.Adam(
        params=[{'params': doc_image_model.parameters()}],
        lr=args.lr * image_lr_factor)
    fusion_optimizer = torch.optim.Adam(
        params=[{'params': doc_fusion_model.module.attention_pooling.parameters()}],
        lr=args.lr * fusion_lr_factor)
    video_fusion_optimizer = torch.optim.Adam(
        doc_fusion_model.module.video_attention_pooling.parameters(),
        lr=args.lr * video_fusion_factor)
    text_generation_optimizer = torch.optim.Adam(
        params=[
            {'params': text_generation_model.parameters()}, ],
        lr=args.lr * text_generation_factor)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(local_rank)
            checkpoint = torch.load(args.resume, map_location=loc)
            doc_image_model.load_state_dict(checkpoint['state_dict'])
            doc_text_model.load_state_dict(checkpoint['doc_text_model'])
            doc_fusion_model.load_state_dict(checkpoint['doc_fusion_model'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    if args.train_file_part_num == 1:
        train_loader, train_sampler = get_loaders(args)

    grad_amp = MaxClipGradScaler(args.batch_size, 128 * args.batch_size,
                                 growth_interval=100) if args.mixed_precision_training else None

    if args.global_step != 0:
        global_step = args.global_step

    if args.evaluate:
        evaluate(train_loader, [doc_text_model, doc_image_model, doc_fusion_model], args)
        exit(0)

    for epoch in range(args.start_epoch, args.epochs):
        if args.train_file_part_num != 1:
            train_loader, train_sampler = get_loaders(args, epoch)
        train_sampler.set_epoch(epoch)

        adjust_learning_rate(text_optimizer, epoch, text_lr_factor, args)
        adjust_learning_rate(image_optimizer, epoch, image_lr_factor, args)
        adjust_learning_rate(fusion_optimizer, epoch, fusion_lr_factor, args)
        adjust_learning_rate(video_fusion_optimizer, epoch, video_fusion_factor, args)
        adjust_learning_rate(text_generation_optimizer, epoch, text_generation_factor, args)

        print("is_xbm: ", args.is_xbm)
        # train for one epoch
        train(train_loader,
              [doc_text_model, doc_image_model, doc_fusion_model, text_generation_model],
              [text_relevance_loss_1, text_relevance_loss_2, text_relevance_loss_3, ITC_ITM_criterion,
               text_generate_criterion],
              [text_optimizer, image_optimizer, fusion_optimizer, video_fusion_optimizer, text_generation_optimizer],
              grad_amp, epoch, args)

        if rank == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'state_dict': doc_image_model.state_dict(),
                'doc_text_model': doc_text_model.state_dict(),
                'doc_fusion_model': doc_fusion_model.state_dict(),
                'text_optimizer': text_optimizer.state_dict(),
                'image_optimizer': image_optimizer.state_dict(),
                'fusion_optimizer': fusion_optimizer.state_dict(),
                'video_fusion_optimizer': video_fusion_optimizer.state_dict(),
                'text_generation_optimizer': text_generation_optimizer.state_dict()
            }, args.checkpoints)

    dist.destroy_process_group()


def get_loaders(args, epoch=0):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.using_other_transforms:
        print("=> using other transforms")
        train_transform = video_transforms.Compose([
            video_transforms.RandomResizedCrop(args.image_size, scale=(0.64, 1.0)),
            video_transforms.RandomHorizontalFlip(),
            volume_transforms.ClipToTensor2(),
            normalize,
        ])
    else:
        train_transform = video_transforms.Compose([
            video_transforms.RandomResizedCrop(args.image_size, scale=(0.64, 1.0)),
            video_transforms.RandomHorizontalFlip(),
            volume_transforms.ClipToTensor2(),
            normalize,
        ])
    val_transform = video_transforms.Compose([
        video_transforms.Resize((args.image_size, args.image_size)),
        video_transforms.CenterCrop(args.image_size),
        volume_transforms.ClipToTensor2(),
        normalize,
    ])

    if args.train_file_part_num == 1:
        train_file = args.train_file
    else:
        pass

    if args.evaluate:
        test_dataset = CrossDomainEmbDatasetV1(
            args.test_file,
            args.clip_length,
            args.goods_img_root,
            args.goods_text_root,
            args.photo_img_root,
            args.photo_text_root,
            args.live_img_root,
            args.live_text_root,
            shuffle=False,
            transform=val_transform,
            return_img_id=True,
            is_train=False,
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=(test_sampler is None),
            num_workers=args.workers,
            pin_memory=True,
            sampler=test_sampler,
            drop_last=False,
            collate_fn=CollateFn(),
        )
        return test_loader, test_sampler

    train_dataset = CrossDomainEmbDataset(train_file, args.clip_length,
                                          args.goods_img_root, args.goods_text_root,
                                          args.photo_img_root, args.photo_text_root,
                                          args.live_img_root, args.live_text_root,
                                          args.frameid_root,
                                          shuffle=False, transform=train_transform, return_img_id=True,
                                          cls_file=args.cls_file)
    train_sampler = CrossDomainDistributedSampler(train_dataset, train_file, args.batch_size)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, collate_fn=CollateFn())
    return train_loader, train_sampler


def train(train_loader, model_list, loss_list, optimizer_list, grad_amp, epoch, args):
    global lr
    global global_step
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    avg_loss = AverageMeter('Loss', ':.4e')
    avg_loss_t = AverageMeter('Loss_t', ':.4e')
    avg_loss_v = AverageMeter('Loss_v', ':.4e')
    avg_loss_cls = AverageMeter('Loss_cls', ':.4e')
    avg_loss_norm = AverageMeter('Loss_norm', ':.4e')
    avg_loss_itc_itm = AverageMeter('Loss_ITC_ITM', ':.4e')
    avg_loss_text_generate = AverageMeter('Loss_text_generate', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, avg_loss, avg_loss_t, avg_loss_v, avg_loss_cls, avg_loss_norm, avg_loss_itc_itm,
         avg_loss_text_generate],
        prefix="Epoch: [{}]".format(epoch))

    [doc_text_model, doc_image_model, doc_fusion_model, text_generation_model] = model_list
    [text_relevance_loss_1, text_relevance_loss_2, text_relevance_loss_3, ITC_ITM_criterion,
     text_generate_criterion] = loss_list
    [text_optimizer, image_optimizer, fusion_optimizer, video_fusion_optimizer,
     text_generation_optimizer] = optimizer_list

    end = time.time()
    for i, (text_input, title_input, images, pids, pid_sources) in enumerate(train_loader):
        data_time.update(time.time() - end)
        global_step += 1

        for k in text_input:
            text_input[k] = text_input[k].cuda(args.local_rank)
        for k in title_input:
            title_input[k] = title_input[k].cuda(args.local_rank)
        images = images.cuda(args.local_rank)

        # compute output
        text_emb = doc_text_model(text_input)
        title_emb = doc_text_model(title_input)
        images_emb = doc_image_model(images)
        images_emb_d = images_emb.detach()
        images_emb_d.requires_grad = args.finetune
        query_emb = text_emb
        fusion_emb, t_emb, v_emb = doc_fusion_model([title_emb, images_emb_d],
                                                    {'tb_writer': args.tb_writer, 'global_step': global_step,
                                                     'local_rank': args.local_rank})

        text_generation_outputs = text_generation_model({
            'input_ids': text_input['input_ids'],
            'token_type_ids': text_input['token_type_ids'],
            'attention_mask': text_input['attention_mask'],
            'encoder_hidden_states': torch.stack([v_emb], dim=1),  # 选v_emb
            'labels': text_input['input_ids']
        })

        # text relevance
        is_xbm = args.is_xbm
        loss = text_relevance_loss_1(query_emb, fusion_emb,
                                     {'tb_writer': args.tb_writer, 'global_step': global_step,
                                      'local_rank': args.local_rank, 'prefix': 'multimodal'},
                                     is_xbm=is_xbm)
        loss_t = text_relevance_loss_2(query_emb, t_emb,
                                       {'tb_writer': args.tb_writer, 'global_step': global_step,
                                        'local_rank': args.local_rank, 'prefix': 'title'})
        loss_v = text_relevance_loss_3(query_emb, v_emb,
                                       {'tb_writer': args.tb_writer, 'global_step': global_step,
                                        'local_rank': args.local_rank, 'prefix': 'vision'})

        loss_norm = 10.0 * (fusion_emb - v_emb).square().mean()

        loss_itc_itm = ITC_ITM_criterion(v_emb, t_emb, fusion_emb)

        loss_text_generate = text_generation_outputs.loss

        # TODO the total loss function
        loss_total = loss + loss_t + loss_v + loss_norm + loss_text_generate

        # compute gradient and do SGD step
        if args.mixed_precision_training:
            grad_amp.scale(loss_total).backward()
            if args.finetune:
                images_emb_grad = images_emb_d.grad
                images_emb.backward(images_emb_grad)
                grad_amp.unscale_(image_optimizer)
                clip_grad_norm_(doc_image_model.parameters(), max_norm=5, norm_type=2)
                grad_amp.step(image_optimizer)
            grad_amp.unscale_(text_optimizer)
            grad_amp.unscale_(fusion_optimizer)
            grad_amp.unscale_(video_fusion_optimizer)
            grad_amp.unscale_(text_generation_optimizer)
            # without query text model
            clip_grad_norm_(doc_text_model.parameters(), max_norm=5, norm_type=2)
            grad_amp.step(text_optimizer)
            grad_amp.step(fusion_optimizer)
            grad_amp.step(video_fusion_optimizer)
            grad_amp.step(text_generation_optimizer)
            grad_amp.update()
        else:
            loss_total.backward()
            if args.finetune:
                images_emb_grad = images_emb_d.grad
                images_emb.backward(images_emb_grad)
                clip_grad_norm_(doc_image_model.parameters(), max_norm=5, norm_type=2)
                image_optimizer.step()
            clip_grad_norm_(doc_text_model.parameters(), max_norm=5, norm_type=2)
            text_optimizer.step()
            fusion_optimizer.step()
            video_fusion_optimizer.step()
            text_generation_optimizer.step()
        text_optimizer.zero_grad()
        fusion_optimizer.zero_grad()
        video_fusion_optimizer.zero_grad()
        text_generation_optimizer.zero_grad()
        if args.finetune:
            image_optimizer.zero_grad()

        # measure accuracy and record loss
        avg_loss.update(loss.item(), 1)
        avg_loss_t.update(loss_t.item(), 1)
        avg_loss_v.update(loss_v.item(), 1)
        avg_loss_norm.update(loss_norm.item(), 1)
        avg_loss_itc_itm.update(loss_itc_itm.item(), 1)
        avg_loss_text_generate.update(loss_text_generate.item(), 1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.local_rank == 0:
            cur_time = time.strftime('%Y-%m-%d %H:%M:%S')
            print('%s\tLR: %.6f\tGlobalStep: %8d' % (cur_time, lr, global_step), flush=True)
            progress.display(i)

        # Record logs in tensorboard
        if args.local_rank == 0:
            args.tb_writer.add_scalar('epoch', epoch, global_step=global_step)
            args.tb_writer.add_scalar('global_step', global_step, global_step=global_step)
            args.tb_writer.add_scalar('lr', lr, global_step=global_step)
            args.tb_writer.add_scalar('losses/loss', loss.item(), global_step=global_step)
            args.tb_writer.add_scalar('losses/loss_t', loss_t.item(), global_step=global_step)
            args.tb_writer.add_scalar('losses/loss_v', loss_v.item(), global_step=global_step)
            args.tb_writer.add_scalar('losses/loss_norm', loss_norm.item(), global_step=global_step)
            args.tb_writer.add_scalar('losses/loss_itc_itm', loss_itc_itm.item(), global_step=global_step)
            args.tb_writer.add_scalar('losses/loss_text_generate', loss_text_generate.item(), global_step=global_step)

        if (i + 1) % args.save_freq == 0:
            if args.rank == 0:
                save_checkpoint({
                    'epoch': '{}_{}'.format(epoch + 1, i + 1),
                    'global_step': global_step,
                    'state_dict': doc_image_model.state_dict(),
                    'doc_text_model': doc_text_model.state_dict(),
                    'doc_fusion_model': doc_fusion_model.state_dict(),
                    'text_generation_model': text_generation_model.state_dict(),
                    'text_optimizer': text_optimizer.state_dict(),
                    'image_optimizer': image_optimizer.state_dict(),
                    'fusion_optimizer': fusion_optimizer.state_dict(),
                    'video_fusion_optimizer': video_fusion_optimizer.state_dict(),
                    'text_generation_optimizer': text_generation_optimizer.state_dict(),
                }, args.checkpoints)


if __name__ == "__main__":
    main()
