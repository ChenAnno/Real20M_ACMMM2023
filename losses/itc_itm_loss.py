from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.base_model import SharedQueueMixin, MomentumDistilationMixin


class ITC_ITM_loss(nn.Module):
    def __init__(self, batch_size, emb_dim=512):
        super(ITC_ITM_loss, self).__init__()

        self.logit_scale = 20.0
        self.criterion_img = nn.CrossEntropyLoss()
        self.criterion_txt = nn.CrossEntropyLoss()
        self.ground_truth = torch.arange(batch_size).cuda()
        self.batch_size = batch_size
        # self.itm_head = nn.Linear(emb_dim, 2)

    def forward(self, visual_emb, text_emb, fusion_emb):
        sim_i2t = visual_emb @ text_emb.t() * self.logit_scale
        sim_t2i = text_emb @ visual_emb.t() * self.logit_scale

        """ITC loss"""
        loss_i = self.criterion_img(sim_i2t, self.ground_truth)
        loss_t = self.criterion_txt(sim_t2i, self.ground_truth)
        loss_itc = (loss_i + loss_t) / 2
        """ITC loss"""

        # """ITM loss"""
        # itm_logits = itm_head(fusion_emb)
        # itm_labels = torch.ones(self.batch_size, dtype=torch.long).cuda()
        # loss_itm = F.cross_entropy(itm_logits, itm_labels)
        # """ITM loss"""

        return loss_itc


class ITC_ITM_loss_momentum(nn.Module, SharedQueueMixin, MomentumDistilationMixin):
    def __init__(
            self,
            emb_dim=512,
            visual_encoder=None,
            text_encoder=None,
            fusion_model=None,
            queue_size=65536,
            temp=0.07,
            alpha=0.4,
            momentum=0.995,
    ):
        super(ITC_ITM_loss_momentum, self).__init__()
        self.emb_dim = emb_dim

        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.fusion_model = fusion_model
        self.visual_encoder_m = deepcopy(self.visual_encoder)
        self.text_encoder_m = deepcopy(self.text_encoder)
        self.fusion_model_m = deepcopy(self.fusion_model)
        self.model_pairs = [
            [self.visual_encoder, self.visual_encoder_m],
            [self.text_encoder, self.text_encoder_m],
            [self.fusion_model, self.fusion_model_m]
        ]
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(self.emb_dim, queue_size).cuda())
        self.register_buffer("text_queue", torch.randn(self.emb_dim, queue_size).cuda())

        self.temp = nn.Parameter(temp * torch.ones([]).cuda())
        self.alpha = alpha
        self.momentum = momentum

    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        return min(1, (epoch * num_iters_per_epoch + iters) / (2 * num_iters_per_epoch))

    def forward(self, v_emb, t_emb, visual_encoder_new, text_encoder_new, fusion_model_new, images, text):
        with torch.no_grad():
            model_pairs = [
                [visual_encoder_new, self.visual_encoder_m],
                [text_encoder_new, self.text_encoder_m],
                [fusion_model_new, self.fusion_model_m]
            ]
            for model_pair in model_pairs:
                for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                    param_m.data = param_m.data * self.momentum + param.data * (1.0 - self.momentum)
            images_emb_m = self.visual_encoder_m(images)
            text_emb_m = self.text_encoder_m(text)
            fusion_emb_m, t_emb_m, v_emb_m = self.fusion_model_m([text_emb_m, images_emb_m])
            image_feat_all = torch.cat([v_emb_m.t(), self.image_queue.clone().detach()], dim=1)
            text_feat_all = torch.cat([t_emb_m.t(), self.text_queue.clone().detach()], dim=1)
            sim_i2t_m = v_emb_m @ text_feat_all / self.temp
            sim_t2i_m = t_emb_m @ image_feat_all / self.temp
            sim_targets = torch.zeros(sim_i2t_m.size()).to(images.device)
            sim_targets.fill_diagonal_(1)
            sim_i2t_targets = (self.alpha * F.softmax(sim_i2t_m, dim=1) + (1 - self.alpha) * sim_targets)
            sim_t2i_targets = (self.alpha * F.softmax(sim_t2i_m, dim=1) + (1 - self.alpha) * sim_targets)

        sim_i2t = v_emb @ text_feat_all / self.temp
        sim_t2i = t_emb @ image_feat_all / self.temp
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
        loss_itc = (loss_i2t + loss_t2i) / 2

        return loss_itc


if __name__ == "__main__":
    b = 16
    emb_size = 512
    loss_itc_itm = ITC_ITM_loss(batch_size=b).cuda()

    v_emb, t_emb, f_emb = torch.randn(b, emb_size).cuda(), torch.randn(b, emb_size).cuda(), \
                          torch.randn(b, emb_size).cuda()
    v_emb = F.normalize(v_emb, dim=-1)
    t_emb = F.normalize(t_emb, dim=-1)
    f_emb = F.normalize(f_emb, dim=-1)

    loss_itc, loss_itm = loss_itc_itm(v_emb, t_emb, f_emb)

    print(loss_itc, loss_itm)
