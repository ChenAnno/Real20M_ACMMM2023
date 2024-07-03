import torch
import torch.nn as nn
import numpy as np


class TextRelevanceLoss(nn.Module):
    def __init__(self, batch_size, emb_dim=512):
        super(TextRelevanceLoss, self).__init__()

        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale = 20.0
        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()
        self.loss_1 = nn.CrossEntropyLoss()
        self.loss_2 = nn.CrossEntropyLoss()
        self.ground_truth = torch.arange(batch_size).cuda()

        self.K = int(batch_size * 10)
        self.query_feats_bank = torch.zeros(self.K, emb_dim).cuda()
        self.doc_feats_bank = torch.zeros(self.K, emb_dim).cuda()
        self.ptr = 0
        self.is_full = False

    def enqueue_dequeue(self, query_feat, doc_feat):
        q_size = query_feat.shape[0]
        if self.ptr + q_size > self.K:
            self.ptr = q_size
            self.is_full = True

        tmp_query = self.query_feats_bank[0: q_size]
        tmp_doc = self.doc_feats_bank[0: q_size]
        self.query_feats_bank[self.ptr: self.ptr + q_size] = tmp_query
        self.doc_feats_bank[self.ptr: self.ptr + q_size] = tmp_doc
        self.query_feats_bank[0: q_size] = query_feat
        self.doc_feats_bank[0: q_size] = doc_feat
        self.ptr += q_size

    def get(self):
        if self.is_full:
            return self.query_feats_bank, self.doc_feats_bank
        else:
            return self.query_feats_bank[:self.ptr], self.doc_feats_bank[:self.ptr]

    def forward(self, image_emb, text_emb, tb_tools, is_xbm=False):
        logit_scale = self.logit_scale
        logits_per_image = logit_scale * image_emb @ text_emb.t()
        logits_per_text = logit_scale * text_emb @ image_emb.t()
        loss_img = self.loss_img(logits_per_image, self.ground_truth)
        loss_txt = self.loss_txt(logits_per_text, self.ground_truth)

        if is_xbm:
            self.enqueue_dequeue(image_emb.detach(), text_emb.detach())
            query_bank, doc_bank = self.get()

            logits_1 = logit_scale * image_emb @ doc_bank.t()  # N * K
            loss_1 = self.loss_1(logits_1, self.ground_truth)

            logits_2 = logit_scale * text_emb @ query_bank.t()
            loss_2 = self.loss_2(logits_2, self.ground_truth)

        if is_xbm:
            total_loss = (loss_img + loss_txt + loss_1 + loss_2) / 4
        else:
            total_loss = (loss_img + loss_txt) / 2

        if tb_tools['local_rank'] == 0:
            acc1_img, acc5_img = accuracy(logits_per_image, self.ground_truth, topk=(1, 5))
            acc1_txt, acc5_txt = accuracy(logits_per_text, self.ground_truth, topk=(1, 5))
            tb_tools['tb_writer'].add_scalar('{}/acc1_img'.format(tb_tools['prefix']), acc1_img[0],
                                             global_step=tb_tools['global_step'])
            tb_tools['tb_writer'].add_scalar('{}/acc5_img'.format(tb_tools['prefix']), acc5_img[0],
                                             global_step=tb_tools['global_step'])
            tb_tools['tb_writer'].add_scalar('{}/acc1_txt'.format(tb_tools['prefix']), acc1_txt[0],
                                             global_step=tb_tools['global_step'])
            tb_tools['tb_writer'].add_scalar('{}/acc5_txt'.format(tb_tools['prefix']), acc5_txt[0],
                                             global_step=tb_tools['global_step'])

            if is_xbm:
                acc1_img, acc5_img = accuracy(logits_1, self.ground_truth, topk=(1, 5))
                acc1_txt, acc5_txt = accuracy(logits_2, self.ground_truth, topk=(1, 5))
                tb_tools['tb_writer'].add_scalar('{}/acc1_img_with_bank'.format(tb_tools['prefix']), acc1_img[0],
                                                 global_step=tb_tools['global_step'])
                tb_tools['tb_writer'].add_scalar('{}/acc5_img_with_bank'.format(tb_tools['prefix']), acc5_img[0],
                                                 global_step=tb_tools['global_step'])
                tb_tools['tb_writer'].add_scalar('{}/acc1_txt_with_bank'.format(tb_tools['prefix']), acc1_txt[0],
                                                 global_step=tb_tools['global_step'])
                tb_tools['tb_writer'].add_scalar('{}/acc5_txt_with_bank'.format(tb_tools['prefix']), acc5_txt[0],
                                                 global_step=tb_tools['global_step'])

        return total_loss


class CyCLIPLoss(nn.Module):

    def __init__(self, batch_size, emb_dim=512):
        super(CyCLIPLoss, self).__init__()

        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.batch_size = batch_size
        self.logit_scale = 20.0
        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()
        self.ground_truth = torch.arange(batch_size).cuda()

    def forward(self, image_emb, text_emb, tb_tools, is_xbm=False):
        logit_scale = self.logit_scale
        logits_image_per_text = logit_scale * image_emb @ text_emb.t()
        logits_text_per_image = logit_scale * text_emb @ image_emb.t()
        logits_image_per_image = logit_scale * image_emb @ image_emb.t()
        logits_text_per_text = logit_scale * text_emb @ text_emb.t()

        # contrastive_loss
        loss_img = self.loss_img(logits_image_per_text, self.ground_truth)
        loss_txt = self.loss_txt(logits_text_per_image, self.ground_truth)
        contrastive_loss = (loss_img + loss_txt) / 2.0
        # crossmodal_cyclic_loss
        crossmodal_cyclic_loss = (logits_text_per_image - logits_image_per_text).square().mean() / (
                    logit_scale * logit_scale) * self.batch_size
        # inmodal_cyclic_loss
        inmodal_cyclic_loss = (logits_image_per_image - logits_text_per_text).square().mean() / (
                    logit_scale * logit_scale) * self.batch_size
        # total_loss
        cylambda1 = cylambda2 = 0.25
        cyclic_loss = cylambda1 * inmodal_cyclic_loss + cylambda2 * crossmodal_cyclic_loss
        # print("contrastive_loss: {}, cyclic_loss: {}".format(contrastive_loss, cyclic_loss))
        total_loss = contrastive_loss + cyclic_loss

        if tb_tools['local_rank'] == 0:
            acc1_img, acc5_img = accuracy(logits_image_per_text, self.ground_truth, topk=(1, 5))
            acc1_txt, acc5_txt = accuracy(logits_text_per_image, self.ground_truth, topk=(1, 5))
            tb_tools['tb_writer'].add_scalar('{}/acc1_img'.format(tb_tools['prefix']), acc1_img[0],
                                             global_step=tb_tools['global_step'])
            tb_tools['tb_writer'].add_scalar('{}/acc5_img'.format(tb_tools['prefix']), acc5_img[0],
                                             global_step=tb_tools['global_step'])
            tb_tools['tb_writer'].add_scalar('{}/acc1_txt'.format(tb_tools['prefix']), acc1_txt[0],
                                             global_step=tb_tools['global_step'])
            tb_tools['tb_writer'].add_scalar('{}/acc5_txt'.format(tb_tools['prefix']), acc5_txt[0],
                                             global_step=tb_tools['global_step'])

        return total_loss


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
