from torchvision.models import (
    swin_s,
    Swin_S_Weights,
    vgg16,
    VGG16_Weights,
    resnet18,
    ResNet18_Weights,
    vit_b_16,
    ViT_B_16_Weights,
    convnext_small,
    ConvNeXt_Small_Weights,

    SwinTransformer,
    VGG,
    ResNet,
    VisionTransformer,
    ConvNeXt
)
import torch.nn as nn
import torch
from argparse import Namespace
import os
from typing import Dict, Tuple


def getModel(opt: Namespace,
             mode: str,
             num_classes: int) -> Tuple[SwinTransformer | VGG | ResNet | VisionTransformer | ConvNeXt, Dict]:
    # 加载模型
    if opt.model == "swin":
        if opt.resume.lower() == "true":
            model = swin_s()
        else:
            model = swin_s(weights=Swin_S_Weights.DEFAULT)
    elif opt.model == "vgg":
        if opt.resume.lower() == "true":
            model = vgg16()
        else:
            model = vgg16(weights=VGG16_Weights.DEFAULT)
    elif opt.model == "resnet":
        if opt.resume.lower() == "true":
            model = resnet18()
        else:
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
    elif opt.model == "vit":
        if opt.resume.lower() == "true":
            model = vit_b_16()
        else:   
            model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    elif opt.model == "convnext":
        if opt.resume.lower() == "true":
            model = convnext_small()
        else:
            model = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
    else:
        raise ValueError("The parameter --model must be in ['swin', 'vgg', 'resnet', 'vit', 'convnext']")

    if hasattr(model, 'head'):
        # 针对部分模型存在 head 属性
        in_features = model.head.in_features
        # 新的头
        new_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, num_classes)
        )
        # 替换头
        model.head = new_head
    elif hasattr(model, 'classifier'):
        # 最后一层通常在 classifier 中
        in_features = model.classifier[-1].in_features
        new_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, num_classes)
        )
        model.classifier[-1] = new_layer
    elif hasattr(model, 'fc'):
        # 针对 ResNet、DenseNet 等模型
        in_features = model.fc.in_features
        new_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, num_classes)
        )
        model.fc = new_layer
    elif hasattr(model, 'heads'):
        # 针对 ViT 等模型
        in_features = model.heads[0].in_features
        new_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, num_classes)
        )
        model.heads[0] = new_head
    else:
        raise AttributeError("模型结构不支持自动替换分类层，请检查模型定义。")

    # 模型信息
    model_info = {}
    
    # 如果需要checkpoints
    if opt.resume.lower() == "true":
        # 如果是训练模式
        if mode == "train":
            # 检查点路径
            checkpoint_path = os.path.join(opt.log_dir, opt.model, "last.pth")
            # 断点
            checkpoint = torch.load(checkpoint_path, map_location=opt.device)
            # 加载权重
            model.load_state_dict(checkpoint["model_state_dict"])
            # 验证最大准确率
            model_info["valid_acc"] = checkpoint["valid_acc"]
            # 优化器权重
            model_info["optimizer"] = checkpoint['optimizer']
            # epoch
            model_info["epoch"] = checkpoint["epoch"]
            # lr
            model_info["lr_schedule"] = checkpoint['lr_schedule']
        # 如果是测试模式
        elif mode == "test":
            # 检查点路径
            checkpoint_path = os.path.join(opt.log_dir, opt.model, "best.pth")
            # 断点
            checkpoint = torch.load(checkpoint_path, map_location=opt.device)
            # 加载权重
            model.load_state_dict(checkpoint["model_state_dict"])
            # 验证最大准确率
            model_info["valid_acc"] = checkpoint["valid_acc"]
        # 如果是预测模式
        elif mode == "predict":
            # 断点
            checkpoint = torch.load(opt.weight_path, map_location=opt.device)
            # 加载权重
            model.load_state_dict(checkpoint["model_state_dict"])
            # 索引对应类别
            model_info["idx_to_class"] = checkpoint["idx_to_class"]
        else:
            raise ValueError("The parameter --mode must be in ['train', 'test', 'predict']")
    return model, model_info
