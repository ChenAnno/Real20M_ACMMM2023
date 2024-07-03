import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.xclip import MultiFrameIntegrationTransformer

__all__ = [
    'QueryAttentionFusionModel',
    'DocAttentionFusionModel',
    'VideoAttentionPooling',
    'VideoAttentionFusionModel',
    'AttentionPooling'
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

        return outputs


class QueryAttentionFusionModel(nn.Module):

    def __init__(self, input_dims, emb_dim):
        super(QueryAttentionFusionModel, self).__init__()
        self.fc0 = nn.Linear(input_dims[0], emb_dim)

    def forward(self, emb_list):
        emb0 = F.normalize(self.fc0(emb_list[0]))
        emb = emb0
        return emb


class Combiner(nn.Module):
    def __init__(self, emb_dim=512, projection_dim=512 * 4, hidden_dim=512 * 8):
        super(Combiner, self).__init__()
        self.text_projection_layer = nn.Linear(emb_dim, projection_dim)
        self.image_projection_layer = nn.Linear(emb_dim, projection_dim)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, emb_dim)

        self.dropout3 = nn.Dropout(0.5)
        self.dynamic_scalar = nn.Sequential(nn.Linear(projection_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, image_features, text_features):
        text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        image_projected_features = self.dropout2(F.relu(self.image_projection_layer(image_features)))

        raw_combined_features = torch.cat((text_projected_features, image_projected_features), -1)
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))
        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        output = self.output_layer(combined_features) + dynamic_scalar * text_features + (
                1 - dynamic_scalar) * image_features
        return F.normalize(output, dim=-1)


class DocAttentionFusionModel(nn.Module):
    def __init__(self, input_dims, emb_dim=512):
        super(DocAttentionFusionModel, self).__init__()
        # doc_text_dim, query_text_dim, doc_image_dim

        # 换成combiner
        # self.attention_pooling = Combiner(emb_dim=emb_dim, projection_dim=2 * emb_dim, hidden_dim=4 * emb_dim)
        self.attention_pooling = AttentionPooling(emb_dim=512, emb_num=2)  # emb_dim改成512

        # 换成X-CLIP的帧融合
        self.video_attention_pooling = MultiFrameIntegrationTransformer()
        # TODO
        # self.video_attention_pooling = VideoAttentionPooling(hidden_dim=input_dims[2])

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
        video_emb = F.normalize(video_emb)

        if not is_fusion:
            return video_emb

        text_emb = emb_list[0]
        text_emb = F.normalize(text_emb)

        embs = torch.stack([text_emb, video_emb], 1)
        emb = F.normalize(self.attention_pooling(embs, tb_tools, is_train))  # [batch, 128]
        # emb = self.attention_pooling(video_emb, text_emb)
        # TODO
        # emb = F.normalize(text_emb + video_emb)

        return emb, text_emb, video_emb


class VideoAttentionFusionModel(nn.Module):
    def __init__(self, input_dims, emb_dim=512):
        super(VideoAttentionFusionModel, self).__init__()
        # doc_text_dim, query_text_dim, doc_image_dim
        self.video_attention_pooling = MultiFrameIntegrationTransformer()
        # self.video_attention_pooling = VideoAttentionPooling(hidden_dim=input_dims[2])

    def forward(self, emb_list, tb_tools=None, is_train=True, is_fusion=True):
        # text_emb, images_emb
        text_emb = emb_list[0]
        video_emb = emb_list[1]

        video_emb = video_emb.view(text_emb.shape[0], -1, video_emb.shape[1])  # B, T, H
        video_emb = self.video_attention_pooling(video_emb)
        video_emb = F.normalize(video_emb)
        text_emb = F.normalize(text_emb)

        return text_emb, video_emb


if __name__ == "__main__":
    # video_attn = VideoAttentionPooling(hidden_dim=768)
    # input = torch.randn([2, 7, 257, 768])
    # out = video_attn(input)
    #
    # print(out.shape)

    pass
