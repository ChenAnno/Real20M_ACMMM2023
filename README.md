# **Real20M: A Large-scale E-commerce Dataset for Cross-domain Retrieval**

This is the source code of our ACM MM 2023 paper "[Real20M: A Large-scale E-commerce Dataset for Cross-domain Retrieval](https://hexiangteng.github.io/papers/ACM%20MM%202023%20Real20M.pdf)".

![image](imgs/framework.png)

## Installation

```bash
conda env create -f environment.yml
source activate Real
```

## Dataset

![image](imgs/dataset.png)

We release the entire section of the Real400K dataset and conducted experimental comparisons for [our paper](https://hexiangteng.github.io/papers/ACM%20MM%202023%20Real20M.pdf) on this dataset. To facilitate downloading of this dataset, we provide a **Baidu Netdisk download link**. Please note:

1. Regarding dataset download, please sign the [Release Agreement](Release_Agreement.pdf) and send it to [Yanzhe Chen](chenyanzhe@stu.pku.edu.cn). By sending the application, you are agreeing and acknowledging that you have read and understand the [notice](notice.pdf). We will reply with the file and the corresponding guidelines right after we receive your request!

2. The dataset is large in scale and requires approximately **136G** of storage consumption.

3. The organization format of this dataset is as follows, please pay attention to the correspondence between goods images, video frames, and their related text.

   ```unicode
   Dataset/
   ├─ Real20M|Real400K/
   │  ├─ query/
   │  ├─ goods/
   │  │  ├─ images
   │  │  ├─ text
   │  ├─ video/
   │  │  ├─ images
   │  │  ├─ text
   ├─ train_file/
   ├─ test_file/
   ├─ checkpoints/
   ```

## Quick Start

- **Dataset**: Contains the data, split files and ckpts.
- **datasets**: Contains loading files for the dataset.
- **evaluate**: Consisting of evaluation scripts for rapid retrieval of massive samples, ≈ 20 minutes to run on a V100.
- **losses**: Includes the loss functions used in this project as well as other commonly used loss functions.
- **models**: Includes the models used in this project as well as related models that may be compared.
- **utils**: Comprises of utility functions that support various tasks within the project.
- ```evaluation.py```: Evaluation code, due to the large dataset size, features are stored by writing to files.
- ```main_cross_domain_emb.py```: Entry for training and testing, with the logic in ```main()``` as follows.
  - Basic settings
  - Initialize models
  - Optionally resume from a checkpoint
  - Data loading
  - TESTING (Exit after running the test code if args.evaluate==True)
  - Initialize losses, optimizers, and grad_amp
  - TRAINING LOOP
  - Terminate the DDP process

Please note:

1. Please note to **complete the path** at the beginning of the following script files.
2. The training code is built on PyTorch with DistributedDataParallel (**DDP**).
3. We pretrain the framework on 2 nodes, each with 8 V100 GPUs (10 epochs in about two days).

```bash
# Train the query-guided cross-domain retrieval framework.
sh train.sh

# Evaluate on the Video2goods task.
sh video2goods_evaluate.sh

# Evaluate on the Goods2video task.
sh goods2video_evaluate.sh
```

## Model Wights

Due to the restriction imposed by Kuaishou on code sharing, which prevents us from making the pre-training framework code public.

However, we will open-source the model weights and provide a link to access them in the **Baidu Netdisk download link**.

Please download and put the checkpoints under: `outputs/checkpoints/`, `pretrain.pth.tar` is the pre-trained model, while `checkpoint.pth.tar` is the model that achieves the SOTA results.

## Citation

If you find our work helps, please cite our paper.

```bibtex
@inproceedings{chen2023real20m,
  title={Real20M: A large-scale e-commerce dataset for cross-domain retrieval},
  author={Chen, Yanzhe and Zhong, Huasong and He, Xiangteng and Peng, Yuxin and Cheng, Lele},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={4939--4948},
  year={2023}
}
```

## Contact

This repo is maintained by [Yanzhe Chen](https://github.com/ChenAnno). Questions and discussions are welcome via `chenyanzhe@stu.pku.edu.cn`.

## Acknowledgements

Our codes reference the following projects. Many thanks to the authors!

- [ALBEF](https://github.com/salesforce/ALBEF)
- [ALPRO](https://github.com/salesforce/ALPRO)
- [CLIP](https://github.com/openai/CLIP)
- [CLIP-Chinese](https://github.com/yangjianxin1/CLIP-Chinese)
- [OpenFashionCLIP](https://github.com/aimagelab/open-fashion-clip)
- [X-CLIP](https://github.com/microsoft/VideoX/tree/master/X-CLIP)
