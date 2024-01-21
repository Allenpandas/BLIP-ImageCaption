## BLIP-ImageCaption

本仓库节选了BLIP论文中 Image Caption预测部分的代码，在源代码的基础上进行优化，原论文内容及实现参考下列资料：

**Github 地址：** https://github.com/Allenpandas/BLIP-ImageCaption

**论文链接：** https://arxiv.org/abs/2201.12086

## 1.环境准备

**步骤0：** 从 [官方网站 ](https://docs.conda.io/en/latest/miniconda.html)下载并安装 Miniconda。

**步骤1：** 创建并激活一个 conda 环境，本仓库建议使用 python3.8、PyTorch1.10.0【原文推荐的环境为PyTorch 1.10.x】、cuda11.3 版本，推荐使用 `环境名-python版本-cuda版本` 的方式进行 conda 环境的命名。

```shell
conda create --name BLIPIC-py3.8-cu11.3 python=3.8 -y
conda activate BLIPIC-py3.8-cu11.3
```

**步骤2（可忽略）：** 若安装 GPU 版本的 PyTorch，参考 [Pytorch官网 ](https://pytorch.org/get-started/previous-versions/) ，使用以下脚本。

```shell
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

## 2.安装流程

**步骤1：** 克隆本仓库。

```shell
git clone https://github.com/Allenpandas/BLIP-ImageCaption.git
cd BLIP-ImageCaption/
```

**步骤2：** 安装所需要的依赖包。

```shell
pip install -r requirements.txt
```

## 3.推理预测

**步骤1：** 下载 [预训练的模型](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth)，将模型放置在根目录下。

**步骤2：** 执行推理脚本 `python image_caption.py --image_dir /path/to/image_directory --model_dir /path/to/pre-train_model _directory --output_json_dir /path/to/output_json_directory`，其中：

- `--image_dir` 是待处理的图像目录，默认为： `./demo` 目录；
- `--model_dir` 是ImageCaption预训练模型的地址，默认为： `./model_base_caption_capfilt_large.pth`；
- `--outout_json_dir` 是输出结果的目录，默认为： `./output` 目录。

**示例脚本：**

```shell
python image_caption.py \
--image_dir './demo' \
--model_dir './model_base_caption_capfilt_large.pth' \
--output_json_dir './output'
```

**输入图像：** `./demo` 目录下的 `000000.png` （上图）和 `000074.png` （下图）。

<p><img src="./demo/000000.png" ></p>

<p><img src="./demo/000074.png"></p>

**输出示例：** `./output/output.json`

```json
[
  {
    "id": 1,
    "filename": "000000.png",
    "caption": "a man walking down a sidewalk next to a blue building"
  },
  {
    "id": 2,
    "filename": "000074.png",
    "caption": "a group of people walking down a sidewalk"
  }
]
```

## Acknowledgement

The implementation of BLIP relies on resources from [ALBEF](https://github.com/salesforce/ALBEF), [Huggingface Transformers](https://github.com/huggingface/transformers), and [timm](https://github.com/rwightman/pytorch-image-models/tree/master/timm). We thank the original authors for their open-sourcing.
