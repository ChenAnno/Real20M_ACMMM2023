import torch
import torch.nn as nn
from transformers import GPT2Config
from transformers import GPT2LMHeadModel
from transformers import BertLMHeadModel
from transformers import AutoTokenizer

__all__ = [
    'GPT2',
]


class GPT2(nn.Module):
    def __init__(self, embedding_size=128, fp16=False,
                 bos_token_id=101, eos_token_id=102, n_head=8, n_layer=4, n_positions=32, vocab_size=21128):
        super(GPT2, self).__init__()

        self.fp16 = fp16
        self.config = GPT2Config(
            architectures=[
                "GPT2LMHeadModel"
            ],
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            n_ctx=n_positions,
            n_embd=embedding_size,
            n_head=n_head,
            n_layer=n_layer,
            n_positions=n_positions,
            output_past=True,
            add_cross_attention=True,
            task_specific_params={
                "text-generation": {
                    "do_sample": True,
                    "max_length": n_positions
                }
            },
            tokenizer_class="BertTokenizer",
            use_cache=True,
            vocab_size=vocab_size
        )
        self.gpt2_model = GPT2LMHeadModel(self.config)

    def forward(self, gpt2_input):
        # print(gpt2_input['input_ids'].shape)
        # print(gpt2_input['token_type_ids'].shape)
        # print(gpt2_input['attention_mask'].shape)
        # print(gpt2_input['encoder_hidden_states'].shape)
        # print(gpt2_input['labels'].shape)

        with torch.cuda.amp.autocast(self.fp16):
            gpt2_output = self.gpt2_model(**gpt2_input)
        return gpt2_output


if __name__ == '__main__':
    model = GPT2().cuda()
    print(model.config)

    # input_ids = torch.LongTensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]).cuda()
    # token_type_ids = torch.LongTensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]).cuda()
    # attention_mask = torch.LongTensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]).cuda()
    # inputs_embeds = torch.rand(2, 1, 128).cuda()
    # labels = torch.LongTensor([[0, 0, 1, 1, 2, 2, 3, 3, 3, 3], [0, 0, 1, 1, 2, 2, 3, 3, 3, 0]]).cuda()
    # gpt2_input = {
    #     'input_ids': input_ids,
    #     'token_type_ids': token_type_ids,
    #     'attention_mask': attention_mask,
    #     'encoder_hidden_states': inputs_embeds,
    #     'labels': labels
    # }
    # print(model(gpt2_input).loss)

    # text generation
    tokenizer1 = AutoTokenizer.from_pretrained('hfl/rbt6')
    generated_ids = [101]
    output_ids = [101]
    v_emb = torch.randn([1, 128]).cuda()
    for i in range(32):  # n_position
        generation_outputs = model({
            'input_ids': torch.LongTensor([generated_ids]).cuda(),
            'encoder_hidden_states': torch.stack([v_emb], dim=1),
        })
        logits = generation_outputs.logits[:, i, :]
        predicted_id = logits.argmax(-1)
        generated_ids.append(predicted_id.item())
        output_ids.append(predicted_id)
        if predicted_id.item() == 102:
            break
    final_ids = output_ids[1:-1]
    final_text = tokenizer1.decode([x.item() for x in final_ids])
    print(final_text)
