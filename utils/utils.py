import torch
import os
from transformers import AutoTokenizer


def save_checkpoint(state, checkpoints):
    if not os.path.exists(checkpoints):
        os.makedirs(checkpoints)
    filename = os.path.join(checkpoints, 'checkpoint_{}.pth.tar'.format(state['epoch']))
    torch.save(state, filename)


def distribute_model(model, local_rank, train_mode=True, broadcast_buffers=True):
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        module=model, device_ids=[local_rank], broadcast_buffers=broadcast_buffers,
        find_unused_parameters=True
    )
    if train_mode:
        model.train()
    else:
        model.eval()
    return model


def compute_relevance_loss(loss_fn, query_emb, target_emb, prefix, args, GLOBAL_STEP, is_xbm=False):
    return loss_fn(
        query_emb,
        target_emb,
        {
            "tb_writer": args.tb_writer,
            "global_step": GLOBAL_STEP,
            "local_rank": args.local_rank,
            "prefix": prefix,
        },
        is_xbm=is_xbm,
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, lr_factor, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    global LR
    LR = args.lr * (0.1 ** (epoch // args.decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = LR * lr_factor


def intersect(list1, list2):
    return list(set(list1) & set(list2))


class CollateFn(object):
    def __init__(self):
        self.tokenizer1 = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.tokenizer2 = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    def __call__(self, data):
        batch_data = list(zip(*data))
        # print(list(batch_data[0]))
        # query_text, title_text, imgs
        batch_data[0] = self.tokenizer1(list(batch_data[0]), return_tensors='pt', padding=True, truncation=True,
                                        max_length=20)  # 20, 40
        batch_data[1] = self.tokenizer2(list(batch_data[1]), return_tensors='pt', padding=True, truncation=True,
                                        max_length=90)
        batch_data[2] = torch.cat(batch_data[2], 0)
        return batch_data


class CollateFnEng(object):
    def __init__(self):
        self.tokenizer1 = AutoTokenizer.from_pretrained("patrickjohncyh/fashion-clip")
        self.tokenizer2 = AutoTokenizer.from_pretrained("patrickjohncyh/fashion-clip")

    def __call__(self, data):
        batch_data = list(zip(*data))
        # print(list(batch_data[0]))
        # query_text, title_text, imgs
        batch_data[0] = self.tokenizer1(list(batch_data[0]), return_tensors='pt', padding=True, truncation=True,
                                        max_length=20)  # 20, 40
        batch_data[1] = self.tokenizer2(list(batch_data[1]), return_tensors='pt', padding=True, truncation=True,
                                        max_length=90)
        batch_data[2] = torch.cat(batch_data[2], 0)
        return batch_data


class CollateFnQuery(object):
    def __init__(self):
        self.tokenizer1 = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    def __call__(self, data):
        batch_data = list(zip(*data))
        batch_data[0] = self.tokenizer1(list(batch_data[0]), return_tensors='pt', padding=True, truncation=True,
                                        max_length=20)
        return batch_data


class CollateFnGoods(object):
    def __init__(self):
        self.tokenizer1 = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    def __call__(self, data):
        batch_data = list(zip(*data))
        batch_data[1] = torch.cat(batch_data[1], 0)
        batch_data[2] = self.tokenizer1(list(batch_data[2]), return_tensors='pt', padding=True, truncation=True,
                                        max_length=90)
        return batch_data

# class CollateFn(object):
#     def __init__(self):
#         self.tokenizer1 = BertTokenizer.from_pretrained('hfl/rbt3', cache_dir='../pretrained/hfl-rbt3',
#                                                         local_files_only=True)
#         self.tokenizer2 = BertTokenizer.from_pretrained('hfl/rbt6', cache_dir='../pretrained/hfl-rbt6',
#                                                         local_files_only=True)
#         # self.tokenizer1 = BertTokenizer.from_pretrained('../pretrained/hfl-rbt3/')
#         # self.tokenizer2 = BertTokenizer.from_pretrained('../pretrained/hfl-rbt6/')
#
#     def __call__(self, data):
#         batch_data = list(zip(*data))
#         # print(list(batch_data[0]))
#         # query_text, title_text, imgs
#         batch_data[0] = self.tokenizer1(list(batch_data[0]), return_tensors='pt', padding=True, truncation=True,
#                                         max_length=40)  # 20
#         batch_data[1] = self.tokenizer2(list(batch_data[1]), return_tensors='pt', padding=True, truncation=True,
#                                         max_length=90)
#         batch_data[2] = torch.cat(batch_data[2], 0)
#         return batch_data
