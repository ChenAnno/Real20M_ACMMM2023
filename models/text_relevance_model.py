import torch
import torch.nn as nn
from transformers import BertModel



__all__ = [
    'TextRelevanceModel',
    'Bert',
    'Roberta',
]


class TextRelevanceModel(nn.Module):

    def __init__(self, embedding_size=512, fp16=False):
        super(TextRelevanceModel, self).__init__()

        self.fp16 = fp16
        self.embedding_size = embedding_size

        self.bert_model = BertModel.from_pretrained(
            'bert-base-chinese',
            cache_dir='./pretrained/bert-base-chinese',
            output_hidden_states=False)

        self.output_size = self.bert_model.config.pooler_fc_size
        if self.embedding_size > 0:
            self.fc = nn.Linear(self.output_size, self.embedding_size)
            self.output_size = self.embedding_size

    def forward(self, bert_input):
        with torch.cuda.amp.autocast(self.fp16):
            bert_output = self.bert_model(**bert_input)

        bert_emb = bert_output[1].float() if self.fp16 else bert_output[1]

        if self.embedding_size > 0:
            bert_emb = self.fc(bert_emb)

        return bert_emb


class Bert(nn.Module):

    def __init__(self, embedding_size=512, fp16=False):
        super(Bert, self).__init__()

        self.fp16 = fp16
        self.embedding_size = embedding_size

        self.bert_model = BertModel.from_pretrained(
            'bert-base-chinese',
            cache_dir='./pretrained/bert-base-chinese',
            output_hidden_states=False)

        self.output_size = self.bert_model.config.pooler_fc_size
        if self.embedding_size > 0:
            self.fc = nn.Linear(self.output_size, self.embedding_size)
            self.output_size = self.embedding_size

    def forward(self, bert_input):
        with torch.cuda.amp.autocast(self.fp16):
            bert_output = self.bert_model(**bert_input)

        bert_emb = bert_output[1].float() if self.fp16 else bert_output[1]

        if self.embedding_size > 0:
            bert_emb = self.fc(bert_emb)

        return bert_emb


class Roberta(nn.Module):

    def __init__(self, model_type='rbt6', embedding_size=128, fp16=False):
        super(Roberta, self).__init__()

        self.fp16 = fp16
        self.embedding_size = embedding_size

        if model_type == 'rbt3':
            self.bert_model = BertModel.from_pretrained(
                'hfl/rbt3',
                cache_dir='./pretrained/hfl-rbt3',
                local_files_only=True,
                output_hidden_states=False)
        elif model_type == 'rbt6':
            self.bert_model = BertModel.from_pretrained(
                'hfl/rbt6',
                cache_dir='./pretrained/hfl-rbt6',
                local_files_only=True,
                output_hidden_states=False)

        self.output_size = self.bert_model.config.pooler_fc_size
        if self.embedding_size > 0:
            self.fc = nn.Linear(self.output_size, self.embedding_size)
            self.output_size = self.embedding_size

    def forward(self, bert_input):
        with torch.cuda.amp.autocast(self.fp16):
            bert_output = self.bert_model(**bert_input)

        bert_emb = bert_output[1].float() if self.fp16 else bert_output[1]

        if self.embedding_size > 0:
            bert_emb = self.fc(bert_emb)

        return bert_emb
