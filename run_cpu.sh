#!/bin/bash

# 命令为：python train.py --model convnext --device "cpu"

# 如果传入参数，则使用传入的 conda.sh 路径；否则使用默认路径
CONDA_SH_PATH=${1:-'D:\app\anaconda3\etc\profile.d\conda.sh'}

echo "使用的 conda.sh 路径为：$CONDA_SH_PATH"

# 加载 conda 初始化脚本
source "$CONDA_SH_PATH"

# 加载环境
conda activate classify

# 定义模型数组
models=("vgg" "resnet" "vit" "swin" "convnext")

# 遍历每个模型
for model in "${models[@]}"; do
    echo "正在训练模型：$model"

    last_pth_path="./log/$model/last.pth"

    # 第一次执行训练脚本，传入当前模型名称
    python train.py --model "$model" --device "cpu"
    status=$?

    # 检查第一次训练是否成功，若返回状态码非 0 则表示出错
    if [ $status -ne 0 ]; then
        echo "训练模型 $model 时出现错误，尝试恢复训练..."

        if [ ! -f "$last_pth_path" ]; then
            echo "错误：文件 $last_pth_path 不存在，程序结束。"
            exit 1
        fi
        
        python train.py --model "$model" --resume "true" --device "cpu"

        # 更新状态码，保存恢复训练后的返回状态
        status=$?
    fi

    # 检查训练是否成功
    if [ $status -eq 0 ]; then
        echo "训练完成，开始测试模型：$model"
        python test.py --model "$model" --device "cpu"

        # 保存 test.py 的返回状态码
        test_status=$?  

        # 检查 test.py 是否运行成功
        if [ $test_status -ne 0 ]; then
            echo "测试模型 $model 失败，程序结束。"
            exit 1
        fi

    else
        echo "训练模型 $model 恢复训练仍然失败，程序结束。"
        # 退出整个脚本，返回状态码 1（非 0 表示异常退出）
        exit 1  
    fi

    echo "------------------------------------"
done
