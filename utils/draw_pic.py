from matplotlib import pyplot as plt


# 画图
def draw_picture(data: list | dict,
                 title: str,
                 x_label: str,
                 y_label: str,
                 save_path: str) -> None:
    # 设置图片大小15x8，以及高清晰度
    plt.figure(figsize=(15, 8), dpi=80)

    if isinstance(data, dict):
        data_length = len(next(iter(data.values())))
        x = list(range(1, data_length + 1))
        for key, values in data.items():
            # 不传入 color 参数，让 Matplotlib 自动选择颜色
            plt.plot(x, values, lw=3, label=key)
        plt.legend(fontsize=20)
    elif isinstance(data, list):
        data_length = len(data)
        x = list(range(1, data_length + 1))
        plt.plot(x, data, lw=3)
    else:
        raise ValueError("data 必须是 list 或 dict 类型")

    # 绘制标题
    plt.title(title, fontsize=30, pad=20)
    # 设置x轴标签
    plt.xlabel(x_label, fontsize=28, labelpad=8)
    # 设置y轴标签
    plt.ylabel(y_label, fontsize=28, labelpad=18)
    # 主刻度标签字体大小
    plt.tick_params(axis='both', which='major', labelsize=25)
    # 画出格子
    plt.grid(alpha=0.4)
    # 保存图片
    plt.savefig(save_path)
