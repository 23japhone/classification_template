import os
import time
from torch.utils.data import (
    DataLoader,
    Dataset,
    ConcatDataset
)
from torchvision.models import (
    SwinTransformer,
    VGG,
    ResNet,
    VisionTransformer,
    ConvNeXt
)
from typing import Tuple
import torch.nn as nn
from torch import optim
import argparse
import torch
from torch import Tensor
from argparse import Namespace
from torchvision.transforms import InterpolationMode
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.transforms import v2
from utils import (
    myDataset,
    annotation,
    setRandomSeed,
    getModel,
    draw_picture,
    write_to_excel
)
import json
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score
)
import platform
import multiprocessing


# 训练模型类
class TrainModel(object):

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


        # 训练集和验证集
        self.train_dataset, self.val_dataset = self.GetDataset()

        # 读取类别索引文件
        with open("./annotation/label.json", 'r') as f:
            class_to_idx = json.load(f)
        # 索引类别文件
        self.idx_to_class = {}
        for each_class, each_idx in class_to_idx.items():
            self.idx_to_class[each_idx] = each_class

        # 训练的度量
        self.meta = {}

    # 获取训练集、验证集
    def GetDataset(self) -> Tuple[Dataset, Dataset]:
        # 如果不存在标签文件
        if not os.path.exists("./annotation/label.json"):
            # 生成标签文件
            annotation(self.opt.dataset_dir)

        # 数据预处理
        original_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize((self.hyper_param["resize"], self.hyper_param["resize"]), interpolation=InterpolationMode.BICUBIC,
                      antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 数据增强
        augment_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize((self.hyper_param["resize"], self.hyper_param["resize"]), interpolation=InterpolationMode.BICUBIC,
                      antialias=True),
            v2.RandomRotation(degrees=10),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 原始训练集转换
        sub_train_dataset = myDataset(self.opt.dataset_dir, "train", original_transform)
        # 验证集
        val_dataset = myDataset(self.opt.dataset_dir, "val", original_transform)
        # 数据增强
        augment_dataset = myDataset(self.opt.dataset_dir, "train", augment_transform)
        # 训练集合并
        train_dataset = ConcatDataset([sub_train_dataset, augment_dataset])
        return train_dataset, val_dataset

    # 训练阶段
    def TrainLoop(self,
                  epoch: int,
                  model: SwinTransformer | VGG | ResNet | VisionTransformer | ConvNeXt,
                  data_loader: DataLoader,
                  criterion: nn.CrossEntropyLoss,
                  optimizer: optim.Optimizer,
                  scheduler: ExponentialLR,
                  warmup_epochs: int,
                  total_steps: int,
                  initial_lr: float,
                  final_lr: float) -> None:
        # 训练模式
        model.train()
        # 一次训练的总损失值
        loss_value = 0
        # 有多少个batch
        batch = len(data_loader)
        # 真实的标签
        all_real_labels = []
        # 预测的标签
        all_pred_labels = []
        # 开始执行时间
        t1 = time.perf_counter()
        # 开始遍历
        for idx, (img, label) in enumerate(data_loader):
            # 加入真实标签列表
            all_real_labels.extend(label.detach().numpy().tolist())
            # 梯度清零
            optimizer.zero_grad()
            # 加入到设备
            img = Tensor(img).clone().to(self.device)
            label = Tensor(label).clone().to(self.device)
            # 预测标签
            pred_label = model(img)
            # 判断设备类型
            if self.opt.device == "cpu":
                # 复制预测类别张量
                pred_class = torch.max(pred_label.detach(), 1)[1]
            # 有gpu
            else:
                # 复制预测类别张量
                pred_class = torch.max(pred_label.cpu().detach(), 1)[1]
            # 加入预测标签列表
            all_pred_labels.extend(pred_class.numpy().tolist())

            # 计算损失值
            loss = criterion(pred_label, label)
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()
            # 损失值加入到总损失中
            loss_value += loss.item()

            # 学习率预热
            if epoch < warmup_epochs:
                self.LinearWarmup(optimizer, total_steps, initial_lr, final_lr)

            # 打印进度条
            percentage = ((idx + 1) / batch) * 100
            pred_bar = "*" * int(percentage)
            future_bar = "." * int(100 - percentage)

            print(f"\repoch {epoch + 1}: {percentage:^3.0f}%[{pred_bar}->{future_bar}]", end="")
        # 多余空行
        print()

        # 学习率衰减
        if epoch >= warmup_epochs:
            scheduler.step()

        # 结束执行时间
        t2 = time.perf_counter()
        # 执行时间
        exec_time = t2 - t1

        # 计算平均损失
        train_loss = loss_value / batch
        # 计算准确率
        train_acc = accuracy_score(all_real_labels, all_pred_labels)
        # 计算召回率
        train_recall = recall_score(all_real_labels, all_pred_labels, average='macro')
        # 计算精确率
        train_precision = precision_score(all_real_labels, all_pred_labels, average='macro')
        # 计算F1分数      
        train_f1 = f1_score(all_real_labels, all_pred_labels, average='macro')

        # 打印输出信息
        print(
            f"exec time: {exec_time: .2f}s train loss: {train_loss: .4f} train acc: {train_acc: .4f} train recall: {train_recall: .4f} train precision: {train_precision: .4f} train f1: {train_f1: .4f} cur_lr: {optimizer.param_groups[0]['lr']: .7f}")

        if "train_loss" not in self.meta:
            self.meta["train_loss"] = []
        self.meta["train_loss"].append(train_loss)

        if "train_acc" not in self.meta:
            self.meta["train_acc"] = []
        self.meta["train_acc"].append(train_acc)

        if "train_recall" not in self.meta:
            self.meta["train_recall"] = []
        self.meta["train_recall"].append(train_recall)

        if "train_precision" not in self.meta:
            self.meta["train_precision"] = []
        self.meta["train_precision"].append(train_precision)

        if "train_f1" not in self.meta:
            self.meta["train_f1"] = []
        self.meta["train_f1"].append(train_f1)

        if "lr" not in self.meta:
            self.meta["lr"] = []
        self.meta["lr"].append(optimizer.param_groups[0]["lr"])

    @torch.no_grad()
    def ValidLoop(self,
                  model: SwinTransformer | VGG | ResNet | VisionTransformer | ConvNeXt,
                  data_loader: DataLoader,
                  criterion: nn.CrossEntropyLoss) -> None:
        # 验证模式
        model.eval()
        # 一次测试的总损失值
        loss_value = 0
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
            label = Tensor(label).clone().to(self.device)
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

            # 计算损失值
            loss = criterion(pred_label, label)
            # 损失值加入到总损失中
            loss_value += loss.item()

            # 打印进度条
            percentage = ((idx + 1) / batch) * 100
            pred_bar = "*" * int(percentage)
            future_bar = "." * int(100 - percentage)

            print(f"\rvalid progress: {percentage:^3.0f}%[{pred_bar}->{future_bar}]", end="")
        # 多余空行
        print()

        # 结束执行时间
        t2 = time.perf_counter()
        # 执行时间
        exec_time = t2 - t1

        # 计算平均损失
        valid_loss = loss_value / batch
        # 计算准确率
        valid_acc = accuracy_score(all_real_labels, all_pred_labels)
        # 计算召回率
        valid_recall = recall_score(all_real_labels, all_pred_labels, average='macro')
        # 计算精确率
        valid_precision = precision_score(all_real_labels, all_pred_labels, average='macro')
        # 计算f1分数
        valid_f1 = f1_score(all_real_labels, all_pred_labels, average='macro')

        # 打印输出信息
        print(
            f"exec time: {exec_time: .2f}s valid loss: {valid_loss: .4f} valid acc: {valid_acc: .4f} valid recall: {valid_recall: .4f} valid precision: {valid_precision: .4f} valid f1: {valid_f1: .4f}")

        if "valid_loss" not in self.meta:
            self.meta["valid_loss"] = []
        self.meta["valid_loss"].append(valid_loss)

        if "valid_acc" not in self.meta:
            self.meta["valid_acc"] = []
        self.meta["valid_acc"].append(valid_acc)

        if "valid_recall" not in self.meta:
            self.meta["valid_recall"] = []
        self.meta["valid_recall"].append(valid_recall)

        if "valid_precision" not in self.meta:
            self.meta["valid_precision"] = []
        self.meta["valid_precision"].append(valid_precision)

        if "valid_f1" not in self.meta:
            self.meta["valid_f1"] = []
        self.meta["valid_f1"].append(valid_f1)

    @staticmethod
    def LinearWarmup(optimizer: optim.Optimizer,
                     warmup_steps: int,
                     initial_lr: float,
                     final_lr: float) -> None:
        for param_group in optimizer.param_groups:
            param_group['lr'] += (final_lr - initial_lr) / warmup_steps

    def TrainFit(self) -> None:
        # 加载
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.hyper_param["batch_size"],
            shuffle=True,
            num_workers=self.hyper_param["num_workers"]
        )
        valid_loader = DataLoader(
            self.val_dataset,
            batch_size=self.hyper_param["batch_size"],
            shuffle=False,
            num_workers=self.hyper_param["num_workers"]
        )

        # 得到模型
        model, model_info = getModel(self.opt, "train", self.hyper_param["num_classes"])
        # 加入设备
        model = model.to(self.device)
        print(f"{self.opt.model}模型已经加载完毕!")

        # 设置损失函数
        criterion = nn.CrossEntropyLoss()
        # 设置优化器
        optimizer = optim.Adam(model.parameters(), lr=self.hyper_param["initial_lr"])
        # 设置学习率下降
        scheduler = ExponentialLR(optimizer, gamma=self.hyper_param["gamma"])

        # 如果加入断点
        if self.opt.resume.lower() == "true":
            best_valid_acc = model_info["valid_acc"]
            optimizer.load_state_dict(model_info["optimizer"])
            # 开始的epoch
            start_epoch = model_info["epoch"]
            scheduler.load_state_dict(model_info["lr_schedule"])
            print(f"加载checkpoints完成")
        else:
            # 训练最大准确率
            best_valid_acc = -float("inf")
            # 开始的epoch
            start_epoch = -1
            print(f"从预训练权重开始训练")

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

        # 进行0.1 * epoch的预热
        warmup_epochs = int(self.hyper_param["warm_up_ratio"] * self.hyper_param["epochs"])
        # 总的步数
        total_steps = len(train_loader) * warmup_epochs

        # 权重保存路径
        weights_dir = os.path.join(self.opt.log_dir, self.opt.model)
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
        best_weights_path = os.path.join(weights_dir, "best.pth")
        last_weights_path = os.path.join(weights_dir, "last.pth")

        # 开始执行时间
        t1 = time.perf_counter()

        # 开始迭代
        for epoch in range(start_epoch + 1, self.hyper_param["epochs"]):
            # 训练模型
            self.TrainLoop(
                epoch, model, train_loader, criterion, optimizer, scheduler,
                warmup_epochs, total_steps, self.hyper_param["initial_lr"], self.hyper_param["warmup_lr"]
            )

            # 验证模型
            self.ValidLoop(model, valid_loader, criterion)

            # 当前权重
            cur_state_dict = model.state_dict()
            # 去掉键中的 "module." 前缀
            new_state_dict = {k.replace("module.", ""): v for k, v in cur_state_dict.items()}

            # 保存最后的模型
            checkpoint = {
                "model_state_dict": new_state_dict,
                "valid_acc": self.meta["valid_acc"][-1],
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                'lr_schedule': scheduler.state_dict()
            }
            torch.save(checkpoint, last_weights_path)

            # 如果准确率是最好的保存模型
            if self.meta["valid_acc"][-1] >= best_valid_acc and epoch >= warmup_epochs:
                # 此时最好的准确率为当前准确率
                best_valid_acc = self.meta["valid_acc"][-1]
                # 站点
                checkpoint = {
                    "model_state_dict": new_state_dict,
                    "valid_acc": self.meta["valid_acc"][-1],
                    "idx_to_class": self.idx_to_class
                }
                # 保存模型
                torch.save(checkpoint, best_weights_path)

        # 结束执行时间
        t2 = time.perf_counter()
        # 执行时间
        exec_time = t2 - t1
        print(f"{self.opt.model}模型训练总用时: {exec_time: .2f} s")
        self.meta["exec_time"] = exec_time

        print(f"模型训练完毕，best valid acc: {best_valid_acc: .4f}")

        if self.opt.delete_last.lower() == "true":
            os.remove(last_weights_path)
            print(f"last.pth已删除")


def main() -> None:
    # 设置随机数种子
    setRandomSeed()

    # 创建命令行
    parser = argparse.ArgumentParser()
    # 创建参数
    parser.add_argument('--model', type=str, default="swin")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--gpu_id", type=str, default="")

    parser.add_argument("--resume", type=str, default="false")
    parser.add_argument("--delete_last", type=str, default="true")

    parser.add_argument("--log_dir", type=str, default="./log")
    parser.add_argument("--dataset_dir", type=str, default="./datasets")
    parser.add_argument("--pretrained_path", type=str, default="./pretrained_weights")

    # 解析命令行
    opt = parser.parse_args()

    if opt.gpu_id and opt.device != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

    # 判断预训练权重下载路径
    if opt.pretrained_path:
        # 更改预训练下载位置
        os.environ['TORCH_HOME'] = opt.pretrained_path

    # 实例化模型
    my_train = TrainModel(opt)
    # 训练模型
    my_train.TrainFit()

    # 分离训练和验证指标
    train_metrics = {}
    valid_metrics = {}
    lr_metrics = {}

    for key, value in my_train.meta.items():
        if key.startswith("train_"):
            # 去掉前缀 "train_" 后存入 train_metrics
            new_key = key[len("train_"):]
            train_metrics[new_key] = value
        elif key.startswith("valid_"):
            # 去掉前缀 "valid_" 后存入 valid_metrics
            new_key = key[len("valid_"):]
            valid_metrics[new_key] = value
        elif key == "lr":
            lr_metrics["learning rate"] = value
        else:
            pass

    # 训练指标图路径
    train_metrics_path = os.path.join(opt.log_dir, opt.model, "train_metrics.png")
    # 验证指标图路径
    valid_metrics_path = os.path.join(opt.log_dir, opt.model, "valid_metrics.png")
    # 学习率图路径
    lr_metrics_path = os.path.join(opt.log_dir, opt.model, "lr_metrics.png")

    # 画图
    draw_picture(train_metrics, "training metrics", "epoch", "value", train_metrics_path)
    draw_picture(valid_metrics, "valid metrics", "epoch", "value", valid_metrics_path)
    draw_picture(lr_metrics, "lr metrics", "epoch", "value", lr_metrics_path)

    print(f"训练指标图已保存")

    # 写入excel
    all_metrics_excel_path = os.path.join(opt.log_dir, opt.model, "metrics_output.xlsx")
    write_to_excel(my_train.meta, all_metrics_excel_path)

    print(f"训练指标总览表已保存")


if __name__ == "__main__":
    main()
