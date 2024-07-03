import os
from tqdm import tqdm
import torch.nn.functional as F

from datasets import *
from utils.utils import *


def evaluate(test_loader, model_list, args):
    [doc_text_model, doc_image_model, doc_fusion_model] = model_list
    doc_text_model = doc_text_model.eval()
    doc_image_model = doc_image_model.eval()
    doc_fusion_model = doc_fusion_model.eval()

    fw = open(os.path.join(args.output_dir, str(args.local_rank)), "w")

    for i, (text_input, title_input, images, pids) in enumerate(tqdm(test_loader)):
        for k in text_input:
            text_input[k] = text_input[k].cuda(args.local_rank)
        for k in title_input:
            title_input[k] = title_input[k].cuda(args.local_rank)
        images = images.cuda(args.local_rank)

        title_emb = doc_text_model(title_input)
        images_emb = doc_image_model(images)
        images_emb_d = images_emb.detach()
        # images_emb_d.requires_grad = args.finetune
        doc_emb, t_emb, v_emb = doc_fusion_model(
            [title_emb, images_emb_d], is_train=False
        )
        output = F.normalize(doc_emb).detach()
        output = output.cpu().numpy().astype("float32").tolist()
        for pid, feat in zip(pids, output):
            fw.write("\t".join([pid, ",".join([str(x) for x in feat])]) + "\n")
    fw.flush()
    fw.close()
