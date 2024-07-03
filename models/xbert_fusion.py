import torch
from torch import nn
import torch.nn.functional as F
from cn_clip.clip.model import LayerNorm

from models.xbert import BertConfig, BertForMaskedLM
from models.attention_fusion_model import VideoAttentionPooling
from models.xclip import MultiFrameIntegrationTransformer

__all__ = [
    'BertXFusionModel',
]


class BertXFusionModel(nn.Module):
    def __init__(self, emb_dim=512, fp16=True):
        super(BertXFusionModel, self).__init__()
        self.fp16 = fp16
        self.out_dim = emb_dim
        self.hidden_dim = 768

        self.prompts_visual_ln = LayerNorm(self.hidden_dim)
        self.fc_fusion = nn.Linear(self.hidden_dim, emb_dim)

        bert_config = BertConfig(
            architectures="BertForMaskedLM",
            attention_probs_dropout_prob=0.1,
            hidden_act='gelu',
            hidden_dropout_prob=0.1,
            hidden_size=768,
            initializer_range=0.02,
            intermediate_size=3072,
            layer_norm_eps=1e-12,
            max_position_embeddings=512,
            num_attention_heads=12,
            num_hidden_layers=12,
            pad_token_id=0,
            type_vocab_size=2,
            vocab_size=21128,  # 30522
            fusion_layer=6,
            encoder_width=768,
        )
        self.attention_pooling = BertForMaskedLM.from_pretrained("hfl/chinese-roberta-wwm-ext", config=bert_config)
        # self.video_attention_pooling = VideoAttentionPooling(hidden_dim=emb_dim)
        self.video_attention_pooling = MultiFrameIntegrationTransformer()

    def forward(self, emb_list, is_fusion=True):
        # text_emb, images_emb
        video_emb = emb_list[1]
        if not is_fusion:
            video_emb = video_emb.view(video_emb.shape[0], -1, video_emb.shape[1])
            video_emb = self.video_attention_pooling(video_emb)
            video_emb = F.normalize(video_emb)
            return video_emb
        else:
            # print(emb_list[0].shape, emb_list[1].shape, emb_list[2].shape, emb_list[3].shape)

            with torch.cuda.amp.autocast(self.fp16):
                text_emb = emb_list[0]  # [batch, n, 768]
                text_emb = F.normalize(text_emb)

                text_emb_global = emb_list[2]
                text_emb_global = F.normalize(text_emb_global)  # [batch, 512]

                video_emb_global = emb_list[3]
                video_emb_global = video_emb_global.view(emb_list[0].shape[0], -1, video_emb_global.shape[1])
                video_emb_global = self.video_attention_pooling(video_emb_global)
                video_emb_global = F.normalize(video_emb_global)  # [batch, 512]

                video_emb = self.prompts_visual_ln(video_emb)  # [b*5, 196, 768]
                if video_emb.shape[0] != text_emb.shape[0]:
                    video_emb = video_emb.view(text_emb.shape[0], 5, -1, self.hidden_dim).mean(dim=1, keepdim=False)
                else:
                    video_emb = video_emb.view(text_emb.shape[0], 1, -1, self.hidden_dim).mean(dim=1, keepdim=False)

                fusion_emb = self.attention_pooling.bert(
                    encoder_embeds=text_emb,
                    encoder_hidden_states=video_emb,  # [b, 196, 768]
                    return_dict=True,
                    mode="fusion")
                fusion_emb = fusion_emb.last_hidden_state[:, 0, :]  # [batch, 768]
                fusion_emb = self.fc_fusion(fusion_emb)  # [batch, 512]

                if self.fp16:
                    return fusion_emb.float(), text_emb_global.float(), video_emb_global.float()

                return fusion_emb, text_emb_global, video_emb_global


if __name__ == '__main__':
    pass
