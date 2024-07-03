import torch
import torch.nn as nn


class TextGenerationLoss(nn.Module):
    def __init__(self):
        super(TextGenerationLoss, self).__init__()

        # self.text_generation_model = text_generation_model

    def forward(self, text_generation_model, text_input, fusion_emb):
        """
        compute text construction loss
        :param text_generation_model:
        :param text_input: input_ids, token_type_ids, attention_mask
        :param fusion_emb: [b, emb_dim]
        :return: loss
        """
        loss_text_generate = text_generation_model({
            'input_ids': text_input['input_ids'],
            'token_type_ids': text_input['token_type_ids'],
            'attention_mask': text_input['attention_mask'],
            'encoder_hidden_states': torch.stack([fusion_emb], dim=1),
            'labels': text_input['input_ids']
        }).loss

        return loss_text_generate
