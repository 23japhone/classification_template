import json
import os


def annotation(dataset_dir: str) -> None:
    """
    Introduction:
        数据集处理，获取类别及其索引

    Args:
        dataset_dir: 数据集根目录
    """

    # 训练集路径
    train_dir = os.path.join(dataset_dir, "train")

    # 如果没有该目录
    if not os.path.exists("./annotation"):
        # 建立文件夹
        os.makedirs("./annotation")

    # 读出每一类的目录
    class_name_list = []
    for class_name in os.listdir(train_dir):
        class_name_list.append(class_name)
    
    # 类别收集
    classes = sorted(class_name_list)
    # 类别对应索引
    class_to_idx = {}
    for idx, cls_name in enumerate(classes):
        class_to_idx[cls_name] = idx
    
    # 写入json文件
    with open("./annotation/label.json", 'w') as f:
        json.dump(class_to_idx, f, indent=4)
