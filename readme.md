<h1 align="center">
    General Template for Image Classification
</h1>

<div align="center">
  <img src="https://img.shields.io/badge/Cuda-support-green.svg" alt="Cuda 支持度">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python 版本">
  <img src="https://img.shields.io/badge/License-MIT-red.svg" alt="License">
</div>

<p align="center">
  一个灵活且模块化的图像分类项目，支持多种backbone
  <br>
  <b>Author: </b>japhone
  <br>
  <b>email: </b>japhonehou@gmail.com
</p>

---

## 目录

- [目录](#目录)
- [特性](#特性)
- [项目结构](#项目结构)
- [安装](#安装)
- [使用方法](#使用方法)

---

## 特性

- **多种骨干网络支持**  
  项目支持 Swin、ViT、VGG、ResNet 和 ConvNeXt 等多种模型，方便比较不同架构的性能。

- **模块化设计**  
  数据加载、模型构建、训练循环等各模块均已封装，便于扩展和维护。

- **灵活的配置**  
  可通过配置文件或命令行参数统一控制超参数、优化器、调度器等设置。

- **自动日志记录与可视化**  
  自动记录训练和验证指标（如准确率、召回率、精确率、F1 分数），并生成指标曲线图。

---

## 项目结构

以下是项目目录的结构：
- **readme.md**：项目的说明文件，包含项目概述、安装、使用方法等信息。
- **requirements.txt**：记录项目需要安装的 Python 包，方便用户快速配置环境。
- **train.py**：训练脚本，负责模型的训练、验证及日志记录。
- **test.py**: 测试脚本，负责模型的评估。
- **annotations/**: 存放数据集标签标注文件
- **configs/**：存放项目的配置文件，比如超参数等。
- **datasets/**：存放数据集，通常将训练、验证、测试数据分开放置，每个子目录下按照类别存放图片。
- **utils/**：包含一些工具函数，例如模型构建、数据预处理、随机种子设置等。
- **pretrained_weights/**：存放预训练模型或训练过程中保存的模型权重。

---

## 安装

1. **克隆项目仓库（跳过，还未上传到github，直接开始第二步）**  

   ```bash
   git clone https://github.com/23japhone/shoes_classification_v2.git
   cd shoes_classification_v2
   ```

2. **创建并激活虚拟环境（推荐）**

    ```bash
    # 使用 conda
    conda create -n classify python=3.10
    conda activate classify

    # 或使用 virtualenv
    python -m venv venv
    source venv/bin/activate   # Linux/Mac
    ```

3. **安装电脑所需配置的pytorch环境，若在linux下载`cuda`版本为`12.4`的，请输入以下命令**

    ```bash
    pip install torch torchvision torchaudio
    ```

4. **安装依赖文件**

    ```bash
    pip install -r requirements.txt
    ```

## 使用方法
项目主要包含`train.py` 和 `test.py` 两个脚本，下面是一些常见用法示例。

1. 训练脚本`train.py`:
- 若不使用GPU或者只使用一个GPU，请注释掉`os.environ["CUDA_VISIBLE_DEVICES"]`
- 若使用CPU

```bash
# 使用swin模型
python train.py --model swin --device cpu --use_gpus false

# 使用VGG模型
python train.py --model vgg --device cpu --use_gpus false

# 使用vit模型
python train.py --model vit --device cpu --use_gpus false

# 使用resnet模型
python train.py --model resnet --device cpu --use_gpus false

# 使用convnext模型
python train.py --model convnext --device cpu --use_gpus false
```

- 若使用一个GPU

```bash
# 使用swin模型
python train.py --model swin --use_gpus false

# 使用VGG模型
python train.py --model vgg --use_gpus false

# 使用vit模型
python train.py --model vit --use_gpus false

# 使用resnet模型
python train.py --model resnet --use_gpus false

# 使用convnext模型
python train.py --model convnext --use_gpus false
```

- 若使用多个GPU

```bash
# 检查需要使用的GPU序号
nvidias-smi

# 设置可用GPU的环境变量，e.g.
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

# 使用swin模型
python train.py --model swin

# 使用VGG模型
python train.py --model vgg

# 使用vit模型
python train.py --model vit

# 使用resnet模型
python train.py --model resnet

# 使用convnext模型
python train.py --model convnext
```
