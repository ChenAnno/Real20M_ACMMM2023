import os
import sys
import re
import glob
import json
import time
import base64
import numpy as np
from tqdm import tqdm

photo_img_root = '/mnt/longvideo/chenkaibing/data/product_video_cls/images'
photo_text_root = '/mnt/longvideo/chenkaibing/data/product_video_cls/texts'
goods_text_root = '/mnt/longvideo/chenkaibing/data/product_search_data_zhannei/texts'
goods_img_root = '/mnt/longvideo/chenkaibing/data/product_search_data_zhannei/images'

def vis_ann(vis_dict, html_file, query2gt):
    html_file_fp = open(html_file, 'w')
    html_file_fp.write('<html>\n<body>\n')
    html_file_fp.write('<meta charset="utf-8">\n')

    for i, (query, items) in tqdm(enumerate(vis_dict.items())):
        html_file_fp.write('<p>\n')
        html_file_fp.write('<tr><td height="5" width="224" alig="center">Query item_id: %s</td></tr>\n' % (query))
        text_path = os.path.join(photo_text_root, str(int(query) % 50000), query + ".txt")
        if not os.path.exists(text_path):
            continue
        with open(text_path) as fr:
            lines = fr.readlines()
            html_file_fp.write('<tr><td height="5" width="1000" alig="left">caption: %s</td></tr>\n<br>' % (lines[0].strip()))
            html_file_fp.write('<tr><td height="5" width="1000" alig="left">title: %s</td></tr>\n<br>' % (lines[1].strip()))
            html_file_fp.write('<tr><td height="5" width="1000" alig="left">text: %s</td></tr>\n<br>' % (lines[2].strip()))

        html_file_fp.write('<table border="0" align="center">\n')
        html_file_fp.write('<tr><td height="10" colspan="100"><hr></td></tr>\n')
        html_file_fp.write('<tr>\n')

        all_list = glob.glob(os.path.join(photo_img_root, str(int(query) % 50000), query + "*.jpg"))
        tmp = []
        if len(all_list) > 1:
            cover = None
            for f in all_list:
                if len(f.split("/")[-1].split(".")[0].split("_")) == 1:
                    cover = f
                else:
                    tmp.append(f)
            tmp.sort(key=lambda x : int(x.split("/")[-1].split(".")[0].split("_")[1]))
            tmp = [cover] + tmp
        else:
            tmp = cover
        clip_length = 5
        L = len(tmp) / clip_length
        seed = [int((2*i+1)*L/2) for i in range(clip_length)]
        seed = [0] + seed
        imgs = []
        for idx in seed:
            imgs.append(tmp[idx])

        gt_list = query2gt[query]

        for img_path in imgs:
            try:
              html_file_fp.write(
                """
                <td bgcolor=%s align='center'>
                    <img width="224" height="224" src="data:image/jpeg;base64, %s">
                    <br> pid: %s
                    <br> gt_num: %s
                </td>
                """ % ("white", base64.b64encode(open(img_path, 'rb').read()).decode(), img_path.split("/")[-1].split(".")[0], len(gt_list))
              )
            except:
                print(query)
                continue

        for gt_pid in gt_list:
            img_path = os.path.join(goods_img_root, str(int(gt_pid) % 50000), gt_pid + ".jpg")
            text_path = os.path.join(goods_text_root, str(int(gt_pid) % 50000), gt_pid + ".txt")
            if not os.path.exists(text_path):
                continue
            text = open(text_path).readlines()[4].strip()
            try:
              html_file_fp.write(
                """
                <td bgcolor=%s align='center'>
                    <img width="224" height="224" src="data:image/jpeg;base64, %s">
                    <br> pid: %s
                    <br> gt_num: %s
                    <br> text: %s
                </td>
                """ % ("green", base64.b64encode(open(img_path, 'rb').read()).decode(), img_path.split("/")[-1].split(".")[0], len(gt_list), text)
              )
            except:
                print(query)
                continue
        html_file_fp.write('</tr>\n')
        html_file_fp.write('<tr><td height="10" colspan="100"><hr></td></tr>\n')
        html_file_fp.write('</table>\n')
        html_file_fp.write('</p>\n')

        html_file_fp.write('<table border="0" align="center">\n')
        html_file_fp.write('<tr><td height="10" colspan="100"><hr></td></tr>\n')
        html_file_fp.write('<tr>\n')

        topk = 20
        for pid, cos_sim in zip(items[0][:topk], items[1][:topk]):
            img_path = os.path.join(goods_img_root, str(int(pid) % 50000), pid + ".jpg")
            text_path = os.path.join(goods_text_root, str(int(pid) % 50000), pid + ".txt")
            if not os.path.exists(text_path):
                continue
            try:
                text = open(text_path).readlines()[4].strip()
            except Exception as e:
                print(e)
                continue
            color = 'white'
            if pid in gt_list:
                color = 'red'
            try:
              html_file_fp.write(
                """
                <td bgcolor=%s align='center'>
                    <img width="224" height="224" src="data:image/jpeg;base64, %s">
                    <br> text: %s
                    <br> pid: %s
                    <br> image sim: %s
                </td>
                """ % (color, base64.b64encode(open(img_path, 'rb').read()).decode(), text, pid, cos_sim)
              )
            except:
                print(pid)
                continue
        html_file_fp.write('</tr>\n')
        html_file_fp.write('<tr><td height="10" colspan="100"><hr></td></tr>\n')
        html_file_fp.write('</table>\n')
        html_file_fp.write('</p>\n')
    html_file_fp.write('</body>\n</html>')


def main(idx_path, dis_path, query_path, doc_feat_path, gt_path, output_path):
    I = np.load(idx_path)
    D = np.load(dis_path)
    print("I shape: ", I.shape)
    print("D shape: ", D.shape)

    query2gt = {}
    with open(gt_path) as fr:
        lines = fr.readlines()
        for line in tqdm(lines):
            values = line.strip().split("\t")
            queryid = values[0].split("/")[-1].split("_")[0]
            for doc in values[1].split(","):
                docid = doc.split("/")[-1].split(".")[0]
                if queryid not in query2gt:
                    query2gt[queryid] = []
                query2gt[queryid].append(docid)

    all_doc_pid = set()
    doc_id2pid = {}
    with open(doc_feat_path) as fr:
        lines = fr.readlines()
        for i, line in enumerate(tqdm(lines)):
            pid = line.strip().split("\t")[0]
            doc_id2pid[i] = pid
            all_doc_pid.add(pid)
    print("all doc pid: ", len(all_doc_pid))

    query_id2pid = {}
    with open(query_path) as fr:
        lines = fr.readlines()
        for i, line in enumerate(tqdm(lines)):
            pid = line.strip().split("\t")[0]
            query_id2pid[i] = pid

    vis_dict = {}
    count = 0
    for idx, query_pid in tqdm(query_id2pid.items()):

        pid_list = []
        dist_list = []
        pid_set = set()
        for i, dist in zip(I[idx], D[idx]):
            pid = doc_id2pid[i]
            cos_sim = (2 - dist) / 2.0
            #if pid in pid_set or cos_sim < 0.85 or pid == query_pid:
            #    continue
            pid_set.add(pid)
            pid_list.append(pid)
            dist_list.append(cos_sim)

        if len(pid_list) > 0:
            count += 1
        else:
            continue
        vis_dict[query_pid] = [pid_list, dist_list]

    print(len(list(vis_dict.keys())))
    print("count: ", count)
    vis_ann(vis_dict, output_path, query2gt)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('usage: python tools.py func', file=sys.stderr)
