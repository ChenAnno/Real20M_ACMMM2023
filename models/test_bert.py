import torch
from transformers import BertModel
from transformers import BertTokenizer


bert_model = BertModel.from_pretrained(
    'hfl/rbt3',
    cache_dir='./pretrained/hfl-rbt3',
    local_files_only=True,
    output_hidden_states=False)

tokenizer = BertTokenizer.from_pretrained('hfl/rbt3', cache_dir='./pretrained/hfl-rbt3', local_files_only=True)

text_1 = "Who was Jim Henson ?"
text_2 = "Jim Henson was a puppeteer"

# Tokenized input with special tokens around it (for BERT: [CLS] at the beginning and [SEP] at the end)
inputs = tokenizer([text_1, text_2], return_tensors='pt', padding=True, truncation=True, max_length=40)
print(inputs)
