import os
import time
from torchvision.models import (
    SwinTransformer,
    VGG,
    ResNet,
    VisionTransformer,
    ConvNeXt
)
import argparse
import torch
from torch import Tensor
from argparse import Namespace
from torchvision.transforms import InterpolationMode, v2
from utils import (
    setRandomSeed,
    getModel
)
import json
import torch.nn as nn
from PIL import Image


# 预测模型类
class predModel(object):

    # 类初始化
    def __init__(self,
                 opt: Namespace) -> None:
        # 命令行参数
        self.opt = opt
        # 设备
        self.device: torch.device = torch.device(opt.device)

        # 超参数配置
        with open("./configs/hyper_param.json", 'r') as f:
            self.hyper_param = json.load(f)
        
        # 数据预处理
        self.original_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize((self.hyper_param["resize"], self.hyper_param["resize"]), interpolation=InterpolationMode.BICUBIC, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 获取预测集
        self.pred_dataset = self.GetDataset()

        # 预测信息
        self.meta = {}
    
    def transformImage(self, 
                       image_path: str) -> Tensor:
        # 打开并预处理图片
        image = Image.open(image_path)

        # 如果图像是调色板模式，则转换为 RGBA
        if image.mode == 'P':
            image = image.convert("RGBA")
        
        image = image.convert("RGB")

        image = self.original_transform(image)
        # 增加 batch 维度
        image = image.unsqueeze(0)
        return image

    # 获取预测集
    def GetDataset(self) -> list:
        # 预测集
        pred_dataset = []

        # 如果为单张图片
        if os.path.isfile(self.opt.img_path):
            filename = os.path.basename(self.opt.img_path)
            # 加入数据集
            pred_dataset.append((filename, self.transformImage(self.opt.img_path)))
        # 如果是文件夹
        elif os.path.isdir(self.opt.img_path):
            # 如果是目录，则预测目录下所有支持的图片
            supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
            # 遍历所有名字
            for filename in os.listdir(self.opt.img_path):
                # 找到属于图片的
                if filename.lower().endswith(supported_extensions):
                    file_path = os.path.join(self.opt.img_path, filename)
                    # 加入数据集
                    pred_dataset.append((filename, self.transformImage(file_path)))
        # 如果没输入
        elif not self.opt.img_path:
            raise ValueError("--img_path必须得传值")
        else:
            raise ValueError("--img_path可能不是文件或者目录")
        return pred_dataset

    @torch.no_grad()
    def predLoop(self,
                 model: SwinTransformer | VGG | ResNet | VisionTransformer | ConvNeXt,
                 image: Tensor) -> int:
        # 验证模式
        model.eval()

        # 加入到设备
        image = Tensor(image).clone().to(self.device)
        # 预测标签
        pred_label = model(image)

        # 判断设备类型
        if self.opt.device == "cpu":
            # 获取预测类别
            pred_class = torch.max(pred_label.detach(), 1)[1]
        # 有gpu
        else:
            # 获取预测类别
            pred_class = torch.max(pred_label.cpu().detach(), 1)[1]
        
        # 变为数字
        predicted_idx = pred_class.item()
        return predicted_idx

    def predFit(self) -> None:
        # 得到模型
        model, model_info = getModel(self.opt, "predict", self.hyper_param["num_classes"])
        # 加入设备
        model = model.to(self.device)
        print(f"{self.opt.model}模型已经加载完毕!")
        
        # 索引对类别字典
        idx_to_class = model_info["idx_to_class"]

        # 使用多gpu
        if torch.cuda.device_count() > 1:
            gpus_len = torch.cuda.device_count()
            print(f"现在使用{gpus_len}块GPUs!")
            device_ids = []
            for i in range(gpus_len):
                device_ids.append(i)
            model = nn.DataParallel(model, device_ids=device_ids)
        elif torch.cuda.device_count() == 1:
            print(f"现在使用1块GPU!")
        else:
            print("现在使用CPU!")

        for filename, image in self.pred_dataset:
            # 开始执行时间
            t1 = time.perf_counter()

            # 模型预测
            predicted_idx = self.predLoop(model, image)

            # 结束执行时间
            t2 = time.perf_counter()
            # 执行时间
            exec_time = t2 - t1
            # 预测标签
            pred_label = idx_to_class[predicted_idx]

            print(f"图片 [{filename}] 预测的类别为: {pred_label}，预测时间为{exec_time: .2f}s")

            # 加入预测字典里
            self.meta[filename] = pred_label


def main():
    # 设置随机数种子
    setRandomSeed()
    
    # 创建命令行
    parser = argparse.ArgumentParser()
    # 创建参数
    parser.add_argument('--model', type=str, default="swin")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--gpu_id", type=str, default="")

    parser.add_argument("--resume", type=str, default="true")

    parser.add_argument("--img_path", type=str, default="./datasets/test/nike")

    parser.add_argument("--weight_path", type=str, default="./log/swin/best.pth")
    parser.add_argument("--save_dir", type=str, default="./pred_result")

    # 解析命令行
    opt = parser.parse_args()

    if opt.gpu_id and opt.device != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    
    # 实例化模型
    my_pred = predModel(opt)
    # 预测模型
    my_pred.predFit()

    if not os.path.exists(os.path.join(opt.save_dir, opt.model)):
        os.makedirs(os.path.join(opt.save_dir, opt.model))

    # 写入json文件
    pred_metrics_path = os.path.join(opt.save_dir, opt.model, "pred_label.json")
    with open(pred_metrics_path, 'w') as f:
        json.dump(my_pred.meta, f, indent=4)

    print(f"预测结果已保存")


if __name__ == "__main__":
    main()
