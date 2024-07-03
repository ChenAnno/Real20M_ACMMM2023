import logging
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F


class MomentumDistilationMixin:
    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(
                    model_pair[0].parameters(), model_pair[1].parameters()
            ):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(
                    model_pair[0].parameters(), model_pair[1].parameters()
            ):
                param_m.data = param_m.data * self.momentum + param.data * (
                        1.0 - self.momentum
                )


class SharedQueueMixin:
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idxs=None):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr: ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr: ptr + batch_size] = text_feats.T

        if idxs is not None:
            idxs = concat_all_gather(idxs)
            self.idx_queue[:, ptr: ptr + batch_size] = idxs.T

        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr


class SharedQueueMixinLive:
    @torch.no_grad()
    def _dequeue_and_enqueue(self, live_feat, text_feat, idxs=None):
        # gather keys before updating queue
        live_feats = concat_all_gather(live_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = live_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.live_queue[:, ptr: ptr + batch_size] = live_feats.T
        self.text_queue[:, ptr: ptr + batch_size] = text_feats.T

        if idxs is not None:
            idxs = concat_all_gather(idxs)
            self.idx_queue[:, ptr: ptr + batch_size] = idxs.T

        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
