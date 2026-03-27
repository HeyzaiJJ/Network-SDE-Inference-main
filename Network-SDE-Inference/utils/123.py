import os
# 关键：开启CUDA同步调试，定位具体错误行
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from matplotlib import pyplot as plt
import sys

# 添加工具库路径（根据您的实际路径调整）
sys.path.append("../Network-SDE-Inference/utils")
from NeuGNN_model import *
from Interaction_func import *

# ====================== 新增：定义图片保存路径并确保路径存在 ======================
SAVE_DIR = r"C:\Users\chenzhijia\Desktop\Network-SDE-Inference-main\Network-SDE-Inference\utils"
os.makedirs(SAVE_DIR, exist_ok=True)  # 若路径不存在则创建，存在则不报错

# ====================== 核心参数配置（适配您的需求） ======================
# 自动判断GPU并配置设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")
USE_CUDA = torch.cuda.is_available()

Num_nodes = 100  # 数据节点数：100
Dimension = 1  # 数据特征维度：1（[200,100]→[200,100,1]）
ndim = Dimension * 1  # 特征维度保持1维（与模型参数名统一）
delt_t = 0.01  # 时间步长
msg_dim = 1  # 消息维度适配1维特征
hidden = 100  # 模型隐藏层维度
aggr = 'add'  # 聚合方式
model = 'Lorenz'  # 模型标识
batch_size = 1  # batch参数改为1
total_epochs = 50  # 训练轮数（可调整）
batch_per_epoch = 156  # 每轮batch数（可调整）
init_lr = 1e-3  # 初始学习率

# ====================== 1. 读取并处理您的原始数据 [200,100] ======================
# 读取您的时序数据（请替换为您的实际文件路径）
Timeseries = pd.read_csv('../utils/100_bcNGSdwran20220607_141207A3108HRfMRIBOLDs004a1001.csv', encoding='utf-8',
                         header=None)
# print(f"原始数据形状: {Timeseries.shape}")  # 应输出 (200, 100)

# 转换为 [时间步长 × 节点数 × 特征维度] 格式
timeseries = Timeseries.values.reshape((-1, Num_nodes, Dimension)).astype(np.float32)  # 强制float32
# print(f"重塑后数据形状: {timeseries.shape}")  # 应输出 (200, 100, 1)

# 计算数值差分（保持原始逻辑，适配1维特征）
timeseries_t0 = timeseries[:-2, :, :]  # t时刻数据
timeseries_t1 = timeseries[1:-1, :, :]  # t+1时刻数据
dX = (timeseries_t1 - timeseries_t0)
dXdt = (timeseries_t1 - timeseries_t0) / delt_t

# 拼接原始数据和导数（适配1维特征）
data = np.concatenate((timeseries[:-2, :, :], dXdt), axis=2)
# print(f"拼接后data形状: {data.shape}")  # 应输出 (198, 100, 2)

# ====================== 2. 构建X（输入）和y（标签） ======================
# 构建t时刻→t+1时刻的预测对
goal_data = data[1:-1, :, 0:Dimension]  # t+1时刻特征
mapping_data = data[0:-2, :, 0:Dimension]  # t时刻特征
# print(f"mapping_data形状: {mapping_data.shape}")  # 应输出 (196, 100, 1)
# print(f"goal_data形状: {goal_data.shape}")  # 应输出 (196, 100, 1)

# 转换为PyTorch张量并移到指定设备（强制float32）
X = torch.as_tensor(mapping_data, dtype=torch.float32).to(DEVICE)
y = torch.as_tensor(goal_data, dtype=torch.float32).to(DEVICE)

# %%
# ====================== 3. 分割训练集/测试集（时序友好分割） ======================
from sklearn.model_selection import train_test_split

# 先转回CPU分割（sklearn不支持GPU张量），分割后再移回GPU
X_np = X.cpu().numpy()
y_np = y.cpu().numpy()

X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, shuffle=False, test_size=0.2)

# 转换为float32张量并移到GPU/CPU（强制类型+设备）
X_train = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
X_test = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32, device=DEVICE)

# print(f"X_train形状: {X_train.shape}")  # 应输出 (156, 100, 1)
# print(f"y_train形状: {y_train.shape}")  # 应输出 (156, 100, 1)
# print(f"X_train设备: {X_train.device}")  # 应输出 cuda:0（若有GPU）

# ====================== 4. 读取100×100邻接矩阵并处理 edge_index ======================
# 读取您处理后的100×100邻接矩阵文件（核心修复：强制裁剪为100×100）
Adj = pd.read_csv('../utils/100x100_correlation_matrix.csv', encoding='utf-8', header=None)
# 核心修复1：强制裁剪为100×100，去掉多余的行/列
Adj = Adj.iloc[:100, :100]  # 只保留前100行、前100列
Adj = Adj.fillna(0)  # 填充空值
Adj = Adj.astype(np.float32)  # 统一类型
# print(f"邻接矩阵形状: {Adj.shape}")  # 必须输出 (100, 100)
assert Adj.shape == (100, 100), "邻接矩阵必须是100×100！"  # 断言校验，避免维度错误


# 构建edge_index（source→target）- 强化索引校验，确保索引≤99
def get_edge_index(Adj):
    num_nodes = Adj.shape[0]
    Adj_np = Adj.values
    # 只保留非0的边，并转换为整数索引
    edge_index = np.where(Adj_np != 0)  # 仅取邻接矩阵中非0的边
    edge_index = np.array(edge_index, dtype=np.int64)  # 强制转为int64（GPU要求）
    # 核心修复2：强制裁剪索引到0~99（100个节点的最大索引）
    edge_index = np.clip(edge_index, 0, 99)  # 直接固定最大索引为99，避免越界
    # 转换为GPU张量
    edge_index = torch.from_numpy(edge_index).to(DEVICE, dtype=torch.int64)
    return edge_index


edge_index = get_edge_index(Adj)  # 初始为target→source
index = [1, 0]
edge_index = edge_index[index]  # 转换为source→target
# 验证edge_index合法性（核心！必须≤99）
# print(f"edge_index形状: {edge_index.shape}")  # 应输出 (2, E)
# print(f"edge_index最大值: {torch.max(edge_index).item()}")  # 必须≤99
assert torch.max(edge_index).item() <= 99, "edge_index索引必须≤99！"
# print(f"edge_index设备: {edge_index.device}")  # 应输出 cuda:0（若有GPU）

# ====================== 5. 构建DataLoader（batch_size=1） ======================
from torch_geometric.data import Data, DataLoader


# 定义数据转换函数（修复batch处理+设备/类型统一）
def create_data_list(X, y, edge_index, device):
    data_list = []
    for i in range(len(y)):
        data = Data(
            x=X[i].to(device, dtype=torch.float32),  # 强制float32
            edge_index=edge_index.clone().to(device),  # 克隆边索引，避免共享内存
            y=y[i].to(device, dtype=torch.float32),
            batch=torch.zeros(X[i].shape[0], dtype=torch.int64).to(device)  # 手动指定batch
        )
        data_list.append(data)
    return data_list


# 训练集DataLoader
train_data_list = create_data_list(X_train, y_train, edge_index, DEVICE)
trainloader = DataLoader(
    train_data_list,
    batch_size=batch_size,  # batch改为1
    shuffle=True,
    pin_memory=False  # 关闭pin_memory，避免GPU内存冲突
)

# 测试集DataLoader
test_data_list = create_data_list(X_test, y_test, edge_index, DEVICE)
testloader = DataLoader(
    test_data_list,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=False
)

# ====================== 6. 初始化模型并移到GPU ======================
ogn = SDIunweighted(
    model=model,
    n_f=Dimension,
    msg_dim=msg_dim,
    ndim=ndim,
    delt_t=delt_t,
    hidden=hidden,
    edge_index=edge_index,
    aggr=aggr
).to(DEVICE)
# 验证模型参数设备
# print(f"模型设备: {next(ogn.parameters()).device}")  # 应输出 cuda:0（若有GPU）
# print("模型初始化完成")

# ====================== 7. 定义优化器和学习率调度器 ======================
from torch.optim.lr_scheduler import OneCycleLR

opt = torch.optim.Adam(ogn.parameters(), lr=init_lr, weight_decay=1e-8)
sched = OneCycleLR(opt, max_lr=init_lr,
                   steps_per_epoch=batch_per_epoch,
                   epochs=total_epochs, final_div_factor=1e5)


# ====================== 8. 定义特征提取函数（保留原始核心） ======================
def get_messages(ogn, newtestloader):
    def get_message_info(tmp):
        ogn.to("cpu")
        tmp = tmp.to("cpu")

        s1 = tmp.x[tmp.edge_index[0]]  # source
        s1 = s1[:, 0]
        s2 = tmp.x[tmp.edge_index[1]]  # target
        s2 = s2[:, 0]
        tmp_cat = torch.cat([s2, s1])  # xi,xj
        tmp_cat = tmp_cat.reshape(2, -1).t()  # [E, 2]

        m12 = ogn.msg_fnc(tmp_cat)
        all_messages = torch.cat((tmp_cat, m12), dim=1)

        columns = ['x1', 'x2', 'e1']
        df = pd.DataFrame(
            data=all_messages.cpu().detach().numpy(),
            columns=columns
        )

        ogn.to(DEVICE)
        return df

    msg_info = []
    for i, g in enumerate(newtestloader):
        msg_info.append(get_message_info(g))
    msg_info = pd.concat(msg_info)
    return msg_info


def get_selfDynamics(ogn, newtestloader):
    def get_selfDynamics_info(tmp):
        ogn.to("cpu")
        tmp = tmp.to("cpu")

        tmp_x = tmp.x[tmp.edge_index[1]]
        self_dyn_x = ogn.node_fnc_x(tmp_x)
        self_dyn_all = torch.cat((tmp_x, self_dyn_x), dim=1)
        columns = ['x', 's1']

        df = pd.DataFrame(
            data=self_dyn_all.cpu().detach().numpy(),
            columns=columns
        )

        ogn.to(DEVICE)
        return df

    selfDyn_info = []
    for i, g in enumerate(newtestloader):
        selfDyn_info.append(get_selfDynamics_info(g))
    selfDyn_info = pd.concat(selfDyn_info)
    return selfDyn_info


def get_diffusion(ogn, newtestloader):
    def get_diffusion_info(tmp):
        ogn.to("cpu")
        tmp = tmp.to("cpu")

        tmp_x = tmp.x[tmp.edge_index[1]]
        self_diff_x = ogn.stochastic_x(tmp_x)
        self_diff_all = torch.cat((tmp_x, self_diff_x), dim=1)
        columns = ['x', 'd1']

        df = pd.DataFrame(
            data=self_diff_all.cpu().detach().numpy(),
            columns=columns
        )

        ogn.to(DEVICE)
        return df

    selfDiffusion_info = []
    for i, g in enumerate(newtestloader):
        selfDiffusion_info.append(get_diffusion_info(g))
    selfDiffusion_info = pd.concat(selfDiffusion_info)
    return selfDiffusion_info


# %%
# ====================== 9. 构建测试集用于特征记录 ======================
import numpy as onp

onp.random.seed(0)
test_idxes = onp.random.randint(0, len(X_test), min(1000, len(X_test)))

# 构建新的testloader用于特征提取
new_test_data_list = [test_data_list[i] for i in test_idxes]
newtestloader = DataLoader(
    new_test_data_list,
    batch_size=len(test_idxes),
    shuffle=False,
    pin_memory=False
)

# ====================== 10. 训练模型（原始逻辑+GPU异常捕获） ======================
from tqdm import tqdm

# 初始化记录列表
messages_over_time = []
selfDyn_over_time = []
diffusion_over_time = []
recorded_models = []
epoch = 0

# 开始训练
for epoch in tqdm(range(epoch, total_epochs), desc="Training"):
    ogn.train()
    total_loss = 0.0
    i = 0
    j = 0
    num_items = 0
    valid_loss = 0
    valid_num_items = 0

    # 训练阶段（添加异常捕获）
    while i < batch_per_epoch:
        for ginput in trainloader:
            if i >= batch_per_epoch:
                break
            try:
                opt.zero_grad()
                ginput = ginput.to(DEVICE)

                loss = ogn.loss(ginput)
                loss.backward()
                opt.step()
                sched.step()

                total_loss += loss.item()
                i += 1
                num_items += int(ginput.batch[-1] + 1)
            except Exception as e:
                print(f"训练批次{i}出错: {e}")
                print(f"ginput.x形状: {ginput.x.shape}, 设备: {ginput.x.device}")
                print(f"ginput.edge_index形状: {ginput.edge_index.shape}, 最大值: {torch.max(ginput.edge_index).item()}")
                raise e

    # 验证阶段（添加异常捕获）
    ogn.eval()
    with torch.no_grad():  # 验证阶段关闭梯度，减少GPU内存占用
        while j < batch_per_epoch:
            for ginput in testloader:
                if j >= batch_per_epoch:
                    break
                try:
                    ginput = ginput.to(DEVICE)

                    loss = ogn.loss(ginput)
                    valid_loss += loss.item()
                    valid_num_items += int(ginput.batch[-1] + 1)
                    j += 1
                except Exception as e:
                    print(f"验证批次{j}出错: {e}")
                    raise e

    # 计算损失
    cur_loss = total_loss / num_items if num_items > 0 else 0
    cur_valid_loss = valid_loss / valid_num_items if valid_num_items > 0 else 0
    print(f"Epoch {epoch + 1}/{total_epochs} | Train Loss: {cur_loss:.6f} | Valid Loss: {cur_valid_loss:.6f}")

    # 记录特征
    cur_msgs = get_messages(ogn, newtestloader)
    cur_selfdyn = get_selfDynamics(ogn, newtestloader)
    cur_diff = get_diffusion(ogn, newtestloader)
    cur_msgs['epoch'] = epoch
    cur_msgs['loss'] = cur_loss
    messages_over_time.append(cur_msgs)
    selfDyn_over_time.append(cur_selfdyn)
    diffusion_over_time.append(cur_diff)

    # 保存模型状态
    ogn.to("cpu")
    from copy import deepcopy

    recorded_models.append(deepcopy(ogn.state_dict()))
    ogn.to(DEVICE)

# ====================== 11. 模型保存（原始逻辑） ======================
# 核心修改：模型文件名适配100×100邻接矩阵
# torch.save(ogn.cpu().state_dict(), os.path.join(SAVE_DIR, 'Lorenz_model_N100_dim1_100x100adj.pth'))
# print("模型训练完成并保存！")
# ogn.to(DEVICE)

# %%
# ====================== 12. Calculate the inference error（修复Normal-Tensor减法错误） ======================
print("\n========== Calculate the inference error ==========")
ogn.eval()
inference_errors = []
predictions = []
true_values = []

# 遍历测试集计算推理误差（核心修复：提取Normal分布的均值张量）
with torch.no_grad():
    for ginput in testloader:
        ginput = ginput.to(DEVICE)
        # 模型返回元组：(Normal分布对象, xUpdate张量)
        model_output = ogn(ginput.x, ginput.edge_index)
        # 核心修复：提取Normal分布的均值（张量类型）作为预测值
        pred = model_output[0].mean  # 替换原pred = model_output[0]

        # 仅计算MSE误差（原始核心）
        mse_error = torch.mean((pred - ginput.y) ** 2).item()
        inference_errors.append(mse_error)
        # 保存预测值和真实值
        predictions.append(pred.detach().cpu().numpy())
        true_values.append(ginput.y.cpu().numpy())

# 合并数据（原始逻辑）
predictions = np.array(predictions)
true_values = np.array(true_values)

# 仅计算平均MSE（原始逻辑）
mean_mse = np.mean(inference_errors)
print(f"测试集平均MSE误差: {mean_mse:.6f}")

# 基础可视化（原始核心折线图）
plt.figure(figsize=(8, 4))
node_idx = 0
max_time_steps = min(50, len(true_values))
time_steps = np.arange(max_time_steps)
plt.plot(time_steps, true_values[:50, node_idx, 0], color='blue', label='True value')
plt.plot(time_steps, predictions[:50, node_idx, 0], color='red', linestyle='--', label='Predicted value')
plt.xlabel('Time step')
plt.ylabel('Value')
plt.title('Prediction vs True Value')
plt.legend(loc='best')
# 保存折线图到指定路径
plt.savefig(os.path.join(SAVE_DIR, "prediction_vs_true.png"), dpi=300, bbox_inches='tight')
plt.show()

# ====================== 13. Learn the stochastic differential equations（纯原始代码逻辑） ======================
print("\n========== Learn the stochastic differential equations ==========")
# 构建SDE分析用的数据集（原始逻辑）
sde_loader = DataLoader(
    test_data_list,
    batch_size=len(test_data_list),
    shuffle=False,
    pin_memory=False
)

# 提取模型的动力学参数（原始核心逻辑）
with torch.no_grad():
    for ginput in sde_loader:
        ginput = ginput.to(DEVICE)
        ogn.to("cpu")
        ginput = ginput.to("cpu")

        # 1. 计算漂移项（自动力学 + 交互项）- 原始逻辑
        self_dynamics = ogn.node_fnc_x(ginput.x).detach().numpy()
        s1 = ginput.x[ginput.edge_index[0]]
        s2 = ginput.x[ginput.edge_index[1]]
        tmp_cat = torch.cat([s2, s1], dim=1)
        interaction = ogn.msg_fnc(tmp_cat).detach().numpy()

        # 聚合交互项（原始逻辑）
        node_interaction = np.zeros_like(self_dynamics)
        for i, target_idx in enumerate(ginput.edge_index[1].numpy()):
            node_interaction[target_idx] += interaction[i]

        # 总漂移项（原始定义）
        drift = self_dynamics + node_interaction

        # 2. 计算扩散项（原始逻辑）
        diffusion = ogn.stochastic_x(ginput.x).detach().numpy()

        ogn.to(DEVICE)

# 仅打印基础SDE参数（原始逻辑）
print(f"漂移项形状: {drift.shape}")
print(f"扩散项形状: {diffusion.shape}")

# %%
# ====================== 14. Check the accuracy（纯原始代码逻辑） ======================
print("\n========== Check the accuracy ==========")
# 仅展平数据（原始逻辑）
y_true_flat = true_values.flatten()
y_pred_flat = predictions.flatten()

# 仅基础可视化（原始核心散点图）
plt.figure(figsize=(8, 6))
plt.scatter(y_true_flat[:1000], y_pred_flat[:1000], alpha=0.5, s=10)
plt.plot([np.min(y_true_flat), np.max(y_true_flat)],
         [np.min(y_true_flat), np.max(y_true_flat)],
         'r--', linewidth=2, label='Perfect Fit')  # 添加理想拟合线
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.title('Accuracy Check')
plt.legend(loc='best')
# 保存散点图到指定路径
plt.savefig(os.path.join(SAVE_DIR, "accuracy_scatter.png"), dpi=300, bbox_inches='tight')
plt.show()

# 仅打印基础精度信息（原始逻辑）
print(f"预测值范围: [{np.min(y_pred_flat):.4f}, {np.max(y_pred_flat):.4f}]")
print(f"真实值范围: [{np.min(y_true_flat):.4f}, {np.max(y_true_flat):.4f}]")

# 额外提示：打印图片保存路径
print(f"\n图片已保存到：{SAVE_DIR}")
print(f"- 预测值vs真实值折线图：prediction_vs_true.png")
print(f"- 精度散点图：accuracy_scatter.png")