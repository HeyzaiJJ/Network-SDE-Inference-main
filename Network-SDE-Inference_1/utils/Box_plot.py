import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 实验组
EXP_DIR = r"C:\Users\chenzhijia\Desktop\Network-SDE-Inference-main\Network-SDE-Inference_1\utils\every_indicator_EXP"
# 对照组
CTRL_DIR = r"C:\Users\chenzhijia\Desktop\Network-SDE-Inference\Network-SDE-Inference_0\utils\every_indicator_CTRL"
# 输出
SAVE_PATH = r"C:\Users\chenzhijia\Desktop\Network-SDE-Inference-main\Network-SDE-Inference_1\utils\every_indicator_EXP"

# 获取文件
exp_files = sorted([
    os.path.join(EXP_DIR, f)
    for f in os.listdir(EXP_DIR)
    if f.endswith(".csv")
])

ctrl_files = sorted([
    os.path.join(CTRL_DIR, f)
    for f in os.listdir(CTRL_DIR)
    if f.endswith(".csv")
])

n_groups = len(exp_files)
print(f"发现 {n_groups} 组数据")

# 组名（文件名）
group_names = [
    os.path.splitext(os.path.basename(f))[0]
    for f in exp_files
]

# 画图函数
def draw_metric(metric, ylabel, save_name):

    fig, ax = plt.subplots(figsize=(12, 8))

    data_all = []
    positions = []

    current_pos = 1

    for exp_file, ctrl_file in zip(exp_files, ctrl_files):

        df_exp = pd.read_csv(exp_file)
        df_ctrl = pd.read_csv(ctrl_file)

        exp_values = df_exp[metric].dropna().values
        ctrl_values = df_ctrl[metric].dropna().values

        data_all.append(exp_values)
        data_all.append(ctrl_values)

        positions.append(current_pos)
        positions.append(current_pos + 0.20)

        current_pos += 0.7

    # 去除异常值显示
    bp = ax.boxplot(data_all,positions=positions,widths=0.18,showfliers=False,patch_artist=True)
        # 去除异常值显示

    # 着色
    for i in range(len(bp['boxes'])):

        if i % 2 == 0:
            color = 'blue'  # Experiment
        else:
            color = 'red'  # Control

        # 箱体
        bp['boxes'][i].set(facecolor='white',edgecolor=color,linewidth=2.5)

        # 中位数线
        bp['medians'][i].set(color=color,linewidth=2.5)

        # 上下须
        bp['whiskers'][2 * i].set(color=color,linewidth=2.0)
        bp['whiskers'][2 * i + 1].set(color=color,linewidth=2.0)

        # 顶部和底部横线
        bp['caps'][2 * i].set(color=color,linewidth=2.0)
        bp['caps'][2 * i + 1].set(color=color,linewidth=2.0)


    # 横坐标
    centers = []

    for i in range(0, len(positions), 2):
        center = positions[i] + (positions[i + 1] - positions[i]) / 2
        centers.append(center)

    ax.set_xticks(centers)

    ax.set_xticks(centers)
    ax.set_xticklabels(group_names,rotation=0,fontsize=14,fontweight='bold')

    # 坐标轴
    # ax.set_xlabel("Group", fontsize=18, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=18, fontweight='bold')

    ax.tick_params(axis='both',labelsize=14,width=2)
    ax.grid(True, linestyle="--", alpha=0.5)

    # 图例
    legend_elements = [
        Patch(facecolor="white",edgecolor="blue",linewidth=2,label="Experiment"),
        Patch(facecolor="white",edgecolor="red",linewidth=2,label="Control")
    ]

    ax.legend(handles=legend_elements,loc="upper center",bbox_to_anchor=(0.5, -0.05),ncol=2,frameon=False,fontsize=16)

    plt.tight_layout()
    save_path = os.path.join(SAVE_PATH,save_name)

    plt.savefig(save_path,dpi=600,bbox_inches="tight")
    print(f"保存完成: {save_path}")
    plt.show()

# KL divergence
draw_metric(metric="KL",ylabel="KL divergence",save_name="KL_divergence_boxplot.png")

# Lyapunov
draw_metric(metric="Lyapunov",ylabel="Lyapunov exponent",save_name="Lyapunov_boxplot.png")
