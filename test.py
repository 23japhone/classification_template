import os
import time
from torch.utils.data import DataLoader, Dataset
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
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score
)
from torchvision.transforms import InterpolationMode, v2
from utils import (
    myDataset,
    annotation,
    setRandomSeed,
    getModel,
    write_to_excel
)
import json
import torch.nn as nn
import platform
import multiprocessing


# 测试模型类
class TestModel(object):

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
        
        if platform.system() == "Windows":
            self.hyper_param["num_workers"] = 0
            print(f"由于是windows系统，num_workers超参数自动设置为0")
        else:
            # 非Windows系统下可以根据实际需求设置num_workers，这里以CPU核心数为例
            num_workers = multiprocessing.cpu_count()
            if self.hyper_param["num_workers"] > num_workers:
                self.hyper_param["num_workers"] = num_workers // 2
                print(f"由于该设备cpu核心数为{num_workers}个，超过预设超参数，因此设置为{self.hyper_param['num_workers']}个")

        # 获取测试集
        self.test_dataset = self.GetDataset()

        # 测试的度量
        self.meta = {}

    # 获取测试集
    def GetDataset(self) -> Dataset:
        # 如果不存在标签文件
        if not os.path.exists("./annotation/label.json"):
            # 生成标签文件
            annotation(self.opt.dataset_dir)

        # 数据预处理
        original_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize((self.hyper_param["resize"], self.hyper_param["resize"]), interpolation=InterpolationMode.BICUBIC, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 得到测试集
        test_dataset = myDataset(self.opt.dataset_dir, "test", original_transform)
        return test_dataset

    @torch.no_grad()
    def TestLoop(self,
                 model: SwinTransformer | VGG | ResNet | VisionTransformer | ConvNeXt,
                 data_loader: DataLoader) -> None:
        # 验证模式
        model.eval()
        # 有多少个batch
        batch = len(data_loader)
        # 真实的标签
        all_real_labels = []
        # 预测的标签
        all_pred_labels = []
        # 开始执行时间
        t1 = time.perf_counter()
        for idx, (img, label) in enumerate(data_loader):
            # 加入真实标签列表
            all_real_labels.extend(label.detach().numpy().tolist())
            # 加入到设备
            img = Tensor(img).clone().to(self.device)
            # 预测标签
            pred_label = model(img)
            # 判断设备类型
            if self.opt.device == "cpu":
                # 获取预测类别
                pred_class = torch.max(pred_label.detach(), 1)[1]
            # 有gpu
            else:
                # 获取预测类别
                pred_class = torch.max(pred_label.cpu().detach(), 1)[1]
            # 加入预测标签列表
            all_pred_labels.extend(pred_class.numpy().tolist())

            # 打印进度条
            percentage = ((idx + 1) / batch) * 100
            pred_bar = "*" * int(percentage)
            future_bar = "." * int(100 - percentage)

            print(f"\rtest progress: {percentage:^3.0f}%[{pred_bar}->{future_bar}]", end="")
        # 多余空行
        print()

        # 结束执行时间
        t2 = time.perf_counter()
        # 执行时间
        exec_time = t2 - t1

        # 计算准确率
        test_acc = accuracy_score(all_real_labels, all_pred_labels)
        # 计算召回率
        test_recall = recall_score(all_real_labels, all_pred_labels, average='macro')
        # 计算精确率
        test_precision = precision_score(all_real_labels, all_pred_labels, average='macro')
        # 计算f1分数
        test_f1 = f1_score(all_real_labels, all_pred_labels, average='macro')

        if "test_acc" not in self.meta:
            self.meta["test_acc"] = []
        self.meta["test_acc"].append(test_acc)

        if "test_recall" not in self.meta:
            self.meta["test_recall"] = []
        self.meta["test_recall"].append(test_recall)

        if "test_precision" not in self.meta:
            self.meta["test_precision"] = []
        self.meta["test_precision"].append(test_precision)

        if "test_f1" not in self.meta:
            self.meta["test_f1"] = []
        self.meta["test_f1"].append(test_f1)

        # 打印输出信息
        print(f"exec time: {exec_time: .2f}s test acc: {test_acc: .4f} test recall: {test_recall: .4f} test precision: {test_precision: .4f} test f1: {test_f1: .4f}")

    def TestFit(self) -> None:
        # 加载
        test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.hyper_param["batch_size"], 
            shuffle=False, 
            num_workers=self.hyper_param["num_workers"]
        )

        # 得到模型
        model, model_info = getModel(self.opt, "test", self.hyper_param["num_classes"])
        # 加入设备
        model = model.to(self.device)
        print(f"{self.opt.model}模型已经加载完毕!")
        # 打印模型训练信息
        print(f"模型训练的最佳验证集准确率为：{model_info['valid_acc']: .4f}")

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

        # 模型测试
        self.TestLoop(model, test_loader)


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

    parser.add_argument("--log_dir", type=str, default="./log")
    parser.add_argument("--dataset_dir", type=str, default="./datasets")

    # 解析命令行
    opt = parser.parse_args()

    if opt.gpu_id and opt.device != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    
    # 实例化模型
    my_test = TestModel(opt)
    # 测试模型
    my_test.TestFit()

    # 写入excel
    test_metrics_excel_path = os.path.join(opt.log_dir, opt.model, "test_metrics.xlsx")
    write_to_excel(my_test.meta, test_metrics_excel_path)

    print(f"测试指标总览表已保存")


if __name__ == "__main__":
    main()
