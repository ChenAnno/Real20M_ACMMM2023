import torch
import torch.nn as nn
import cn_clip.clip as clip
from cn_clip.clip import load_from_name
from transformers import AutoTokenizer

__all__ = [
    'Vitb16',
    'RoBERTa_CLIP',
]


class ImageCLIP(nn.Module):
    def __init__(self, clip_model_name, fp16=True, download_root=None):
        super(ImageCLIP, self).__init__()
        clip_model, _ = load_from_name(clip_model_name, device="cpu", download_root=download_root)

        self.visual_model = clip_model.visual.float()
        self.visual_model.set_grad_checkpointing()
        self.fp16 = fp16
        self.output_size = self.visual_model.output_dim

    def forward(self, images, mode="global", mask_ratio=0.0):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.visual_model(images, mode=mode, mask_ratio=mask_ratio)
            if self.fp16:
                if isinstance(x, tuple):
                    x = (x[0].float(), x[1].float())
                else:
                    x = x.float()
            return x

# class ImageCLIP(nn.Module):
#     def __init__(self, clip_model_name, fp16=True, download_root=None):
#         super(ImageCLIP, self).__init__()
#         clip_model, _ = load_from_name(clip_model_name, device="cpu", download_root=download_root)
#         self.visual_model = clip_model.visual.float()
#         self.visual_model.set_grad_checkpointing()
#         self.fp16 = fp16
#         self.output_size = self.visual_model.output_dim
#         self.vision_projection = clip_model.visual.proj
#
#     def forward(self, images, mode="global", mask_ratio=0.0):  # TODO 开mask
#         with torch.cuda.amp.autocast(self.fp16):
#             x_seq = self.visual_model(images, mode='local', mask_ratio=mask_ratio)
#             x = x_seq[:, 0, :]
#             x = x @ self.vision_projection
#             if self.fp16:
#                 x, x_seq = x.float(), x_seq.float()
#             if mode == "global":
#                 return x
#             elif mode == "both":
#                 return x, x_seq[:, 1:, :]
#             else:
#                 return x_seq[:, 1:, :]


class TextCLIP(nn.Module):
    def __init__(self, clip_model_name, fp16=True, download_root=None):
        super(TextCLIP, self).__init__()
        self.clip_model, _ = load_from_name(clip_model_name, device="cpu", download_root=download_root)
        self.text_model = self.clip_model.bert.float()
        # self.text_model.set_grad_checkpointing()
        self.fp16 = fp16
        self.output_size = self.clip_model.visual.output_dim
        self.text_projection = self.clip_model.text_projection

    def forward(self, text_inputs, mode="global"):
        with torch.cuda.amp.autocast(self.fp16):
            # x = self.text_model(text_inputs["input_ids"])[0][:, 0, :]  # @ self.clip_model.text_projection
            x_seq = self.text_model(text_inputs["input_ids"])[0]  # [b, n, 768]
            x = x_seq[:, 0, :]
            x = x @ self.text_projection
            if self.fp16:
                x = x.float()
                x_seq = x_seq.float()
            if mode == "global":
                return x
            elif mode == "both":
                return x, x_seq[:, 1:, :]
            else:
                return x_seq[:, 1:, :]


def Vitb16(fp16):
    # clip_model = CLIPModel(clip_model_name="ViT-B-16", fp16=fp16,
    #                        download_root='/mnt/vision_retrieval/chenyanzhe/goods_se_train_code_base/pretrained')
    # for param in clip_model.parameters():
    #     param.requires_grad = False
    # for param in clip_model.visual.parameters():
    #     param.requires_grad = True
    clip_model = ImageCLIP(clip_model_name="ViT-B-16", fp16=fp16,
                           download_root='/mnt/vision_retrieval/chenyanzhe/goods_se_train_code_base/pretrained')
    return clip_model


def RoBERTa_CLIP(fp16):
    clip_model = TextCLIP(clip_model_name="ViT-B-16", fp16=fp16,
                          download_root='/mnt/vision_retrieval/chenyanzhe/goods_se_train_code_base/pretrained')
    return clip_model


if __name__ == '__main__':
    # model = vitb(True)
    # print(model.clip_model.visual)

    # clip_model = load_from_name("ViT-B-16", device="cuda",
    #                             download_root='/mnt/vision_retrieval/chenyanzhe/goods_se_train_code_base/pretrained')

    # clip_model = load_from_name("ViT-B-16", device="cpu",
    #                             download_root='/mnt/vision_retrieval/chenyanzhe/goods_se_train_code_base/pretrained')
    # visual_model = clip_model[0].visual.float().cuda()

    # visual_model = Vitb16(False)
    # visual_model = visual_model.cuda()
    # images = torch.randn([4, 3, 224, 224]).cuda()
    # out = visual_model(images)
    # print(out.shape)

    # caption = ["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]
    # tokenizer1 = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    # text = tokenizer1(
    #     caption,
    #     padding='max_length',
    #     truncation=True,
    #     max_length=52,
    #     return_tensors="pt",
    # )
    # text.input_ids = text.input_ids.cuda()
    # text.token_type_ids = text.token_type_ids.cuda()
    # text.attention_mask = text.attention_mask.cuda()

    images = torch.randn([4, 3, 224, 224]).cuda()

    model = Vitb16(False).cuda()

    # text_output = model(text_inputs=text, mode="text")
    image_output = model(images=images, mode="global")

    # print(text_output.shape)
    print(image_output.shape)

    # text_model = RoBERTa_CLIP(False).cuda()
    # output = text_model(text)
    # output2 = text_model(text2)
    #
    # print("Text input:", text)
    # print("Tokenizer:", text2)
    # print("Output:", output.shape)
    # print("Output2:", output2.shape)
