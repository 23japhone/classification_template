import os
from torch.utils.data import Dataset
from typing import Tuple
import torch
from torchvision.transforms import v2
from PIL import Image
import json


class myDataset(Dataset):
    def __init__(self, 
                 dataset_dir: str, 
                 dataset_mode: str, 
                 transform: v2.Compose) -> None:
        """
        将自己的数据集转换为torch的Dataset格式

        Args:
            dataset_dir: 数据集根目录
            dataset_mode: 数据集的属性，比如train、val、test
            transform: 数据预处理的变换
        """

        self.dataset_dir = os.path.join(dataset_dir, dataset_mode)
        self.transform = transform
        self.samples = []
        # 读取数据集并进行缓存
        self.readDataset()
    
    def readDataset(self) -> None:
        """
        读取数据集并进行缓存
        """
        
        # 打开json文件
        with open("./annotation/label.json", 'r') as f:
            class_to_idx = json.load(f)
        classes = list(class_to_idx.keys())
        
        # 收集图片路径和标签
        for cls_name in classes:
            cls_dir = os.path.join(self.dataset_dir, cls_name)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    path = os.path.join(cls_dir, fname)
                    self.samples.append((path, class_to_idx[cls_name]))

    def __len__(self) -> int:
        """
        读取数据集的长度
        """
        return len(self.samples)

    def __getitem__(self, 
                    index: int) -> Tuple[torch.Tensor, int]:
        """
        读取单个数据的图片和标签
        Args:
            index: 数据集的索引
        Returns:
            image: Tensor类型的图片
            label: 图片的标签
        """
        image_path, label = self.samples[index]
        image = Image.open(image_path)

        # 如果图像是调色板模式，则转换为 RGBA
        if image.mode == 'P':
            image = image.convert("RGBA")
        
        image = image.convert("RGB")

        image = self.transform(image)
        return image, label
    