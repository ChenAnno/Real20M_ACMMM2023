import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = [
    'BaseDocAttentionFusionModel',
]


class VideoAttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(VideoAttentionPooling, self).__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, inputs):
        # (B, T, H) -> (B, T, 1)
        energy = self.projection(inputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, T, H) * (B, T, 1) -> (B, H)
        outputs = (inputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs


class AttentionPooling(nn.Module):
    def __init__(self, emb_dim, emb_num):
        super(AttentionPooling, self).__init__()
        self.emb_dim = emb_dim
        self.emb_num = emb_num
        self.projection = nn.Linear(emb_dim * emb_num, emb_num)

    def forward(self, inputs, tb_tools, is_train=True):
        # (B, T, H) -> (B, T)
        energy = self.projection(inputs.view(inputs.shape[0], -1))
        weights = F.softmax(energy, dim=1)
        # weights[:, 0] = 0.55
        # weights[:, 1] = 0.45
        # print(weights)
        # (B, T, H) * (B, T, 1) -> (B, H)
        outputs = (inputs * weights.unsqueeze(-1)).sum(dim=1)

        if tb_tools is not None and tb_tools['local_rank'] == 0:
            with torch.no_grad():
                mean_weights = torch.mean(weights, dim=0, keepdim=False)
                tb_tools['tb_writer'].add_scalar('attention_weights/title', mean_weights[0],
                                                 global_step=tb_tools['global_step'])
                tb_tools['tb_writer'].add_scalar('attention_weights/image', mean_weights[1],
                                                 global_step=tb_tools['global_step'])

        return outputs


class BaseDocAttentionFusionModel(nn.Module):
    def __init__(self, input_dims, emb_dim=512):
        super(BaseDocAttentionFusionModel, self).__init__()
        # doc_text_dim, query_text_dim, doc_image_dim

        self.attention_pooling = AttentionPooling(emb_dim=emb_dim, emb_num=2)
        self.video_attention_pooling = VideoAttentionPooling(hidden_dim=input_dims[2])
        self.fc_t = nn.Linear(input_dims[0], emb_dim)
        self.fc_v = nn.Linear(input_dims[2], emb_dim)

    def forward(self, emb_list, tb_tools=None, is_train=True, is_fusion=True):
        # text_emb, images_emb
        video_emb = emb_list[1]
        # video_emb = video_emb.view(text_emb.shape[0], -1, video_emb.shape[1])  # B, T, H
        if not is_fusion:
            # video_emb = video_emb.view(1, -1, video_emb.shape[1])  # 1, T, H
            video_emb = video_emb.view(video_emb.shape[0], -1, video_emb.shape[1])
        else:
            video_emb = video_emb.view(emb_list[0].shape[0], -1, video_emb.shape[1])  # B, T, H

        video_emb = self.video_attention_pooling(video_emb)
        video_emb = F.normalize(self.fc_v(video_emb))

        if not is_fusion:
            return video_emb

        text_emb = emb_list[0]
        text_emb = F.normalize(self.fc_t(text_emb))

        embs = torch.stack([text_emb, video_emb], 1)
        emb = F.normalize(self.attention_pooling(embs, tb_tools, is_train))  # [batch, 128]

        return emb, text_emb, video_emb


if __name__ == "__main__":
    # video_attn = VideoAttentionPooling(hidden_dim=768)
    # input = torch.randn([2, 7, 257, 768])
    # out = video_attn(input)
    #
    # print(out.shape)

    pass
