import os

# CUDA同步调试
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from copy import deepcopy
from matplotlib import pyplot as plt

# 工具库路径
sys.path.append("../Network-SDE-Inference_1/utils")

from NeuGNN_model import *
from Interaction_func import *
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence


# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")
USE_CUDA = torch.cuda.is_available()

# 参数
Num_nodes = 100
Dimension = 1
ndim = Dimension * 1
delt_t = 0.01
msg_dim = 1
hidden = 100
aggr = 'add'
model = 'Lorenz'
batch_size = 1
total_epochs = 50
init_lr = 5e-4

# 路径
TIME_DIR = r"C:\Users\chenzhijia\Desktop\Network-SDE-Inference-main\Network-SDE-Inference_1\utils\new_data\100_time_series_csv\None"
ADJ_DIR = r"C:\Users\chenzhijia\Desktop\Network-SDE-Inference-main\Network-SDE-Inference_1\utils\new_data\100x100_correlation_matrix\None"
RESULT_DIR = r"C:\Users\chenzhijia\Desktop\Network-SDE-Inference-main\Network-SDE-Inference_1\utils\batch_results_d"
os.makedirs(RESULT_DIR, exist_ok=True)

# 获取全部CSV文件
def get_all_csv_files(root_dir):
    csv_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".csv"):
                full_path = os.path.join(root, file)
                csv_files.append(full_path)
    return csv_files

# edge_index
def get_edge_index(Adj):
    Adj_np = Adj.values.copy()
    # 稀疏化
    threshold = 0.7
    Adj_np[np.abs(Adj_np) < threshold] = 0
    # density = np.sum(Adj_np != 0) / (Adj_np.shape[0] ** 2)
    # print(f"图连接密度: {density:.4f}")
    edge_index = np.where(Adj_np != 0)
    edge_index = np.array(edge_index, dtype=np.int64)
    edge_index = np.clip(edge_index, 0, 99)
    edge_index = torch.from_numpy(edge_index).to(DEVICE,dtype=torch.int64)
    return edge_index

# DataList
def create_data_list(X, y, edge_index, device):
    data_list = []
    for i in range(len(y)):
        data = Data(
            x=X[i].to(device, dtype=torch.float32),
            edge_index=edge_index.clone().to(device),
            y=y[i].to(device, dtype=torch.float32),
            batch=torch.zeros(X[i].shape[0],dtype=torch.int64).to(device))
        data_list.append(data)
    return data_list

# 获取全部Timeseries文件
timeseries_files = get_all_csv_files(TIME_DIR)
print(f"发现 {len(timeseries_files)} 个Timeseries文件")

# 结果保存
all_results = []

# 批量训练
for ts_file in timeseries_files:
    try:
        # 获取对应邻接矩阵
        relative_path = os.path.relpath(ts_file,TIME_DIR)
        adj_file = os.path.join(ADJ_DIR,relative_path)
        file_name = os.path.basename(ts_file)
        print(f"邻接矩阵:\n{adj_file}")
        if not os.path.exists(adj_file):
            print("未找到对应邻接矩阵")
            continue

        # 读取Timeseries
        Timeseries = pd.read_csv(ts_file,encoding='utf-8',header=None)
        timeseries = Timeseries.values.reshape((-1, Num_nodes, Dimension)).astype(np.float32)

        # 差分
        timeseries_t0 = timeseries[:-2, :, :]
        timeseries_t1 = timeseries[1:-1, :, :]
        dX = (timeseries_t1 - timeseries_t0)
        dXdt = dX / delt_t

        data = np.concatenate((timeseries[:-2, :, :], dXdt),axis=2)
        goal_data = data[1:-1, :, 0:Dimension]
        mapping_data = data[0:-2, :, 0:Dimension]

        # tensor
        X = torch.as_tensor(mapping_data,dtype=torch.float32).to(DEVICE)
        y = torch.as_tensor(goal_data,dtype=torch.float32).to(DEVICE)

        # train test split
        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy()

        X_train, X_test, y_train, y_test = train_test_split(X_np,y_np,shuffle=False,test_size=0.2)
        X_train = torch.tensor(X_train,dtype=torch.float32,device=DEVICE)
        y_train = torch.tensor(y_train,dtype=torch.float32,device=DEVICE)
        X_test = torch.tensor(X_test,dtype=torch.float32,device=DEVICE)
        y_test = torch.tensor(y_test,dtype=torch.float32,device=DEVICE)


        # 读取邻接矩阵
        Adj = pd.read_csv(adj_file,encoding='utf-8',header=None)
        Adj = Adj.values.astype(np.float32)

        # 强制裁剪
        assert Adj.shape == (100, 100), "邻接矩阵必须100×100"
        # 限制范围
        Adj = np.clip(Adj, -1.0, 1.0)
        # 去除自连接
        np.fill_diagonal(Adj, 0)
        Adj = pd.DataFrame(Adj)


        # edge_index
        edge_index = get_edge_index(Adj)
        index = [1, 0]
        edge_index = edge_index[index]

        assert torch.max(edge_index).item() <= 99

        # DataLoader
        train_data_list = create_data_list(X_train,y_train,edge_index,DEVICE)
        test_data_list = create_data_list(X_test,y_test,edge_index,DEVICE)

        trainloader = DataLoader(train_data_list,batch_size=batch_size,shuffle=True,pin_memory=False)
        testloader = DataLoader(test_data_list,batch_size=batch_size,shuffle=False,pin_memory=False)

        # 动态batch_per_epoch
        batch_per_epoch = len(trainloader)
        print(f"batch_per_epoch: {batch_per_epoch}")

        # 初始化模型
        ogn = SDIunweighted(model=model,n_f=Dimension,msg_dim=msg_dim,ndim=ndim,delt_t=delt_t,hidden=hidden,edge_index=edge_index,aggr=aggr).to(DEVICE)
        opt = torch.optim.Adam(ogn.parameters(),lr=init_lr,weight_decay=1e-5)
        sched = OneCycleLR(opt,max_lr=init_lr,steps_per_epoch=batch_per_epoch,epochs=total_epochs,final_div_factor=1e5)

        # 开始训练
        for epoch in tqdm(range(total_epochs),desc=f"Training {file_name}"):

            ogn.train()
            total_loss = 0.0
            num_items = 0

            for ginput in trainloader:
                ginput = ginput.to(DEVICE)
                opt.zero_grad()

                loss = ogn.loss(ginput)
                loss.backward()
                opt.step()
                sched.step()
                total_loss += loss.item()
                num_items += 1


            cur_loss = total_loss / num_items
            print(
                f"Epoch {epoch+1}/{total_epochs} "

                f"| Loss: {cur_loss:.6f}")

        # 测试
        ogn.eval()
        inference_errors = []
        kl_errors = []
        predictions = []
        true_values = []
        lyapunov_values = []

        with torch.no_grad():
            for ginput in testloader:
                ginput = ginput.to(DEVICE)
                model_output = ogn(ginput.x,ginput.edge_index)
                pred_dist = model_output[0]
                pred = pred_dist.mean

                # MSE
                mse_error = torch.mean((pred - ginput.y) ** 2).item()
                inference_errors.append(mse_error)

                # KL
                true_mean = ginput.y
                true_std = torch.std(ginput.y)
                true_std = true_std * torch.ones_like(true_mean)
                true_dist = Normal(true_mean,true_std)

                kl_error = kl_divergence(true_dist,pred_dist).mean().item()
                kl_errors.append(kl_error)

                predictions.append(pred.detach().cpu().numpy())
                true_values.append(ginput.y.cpu().numpy())

                # Lyapunov
                x0 = ginput.x.clone()
                epsilon = 1e-5
                perturbation = epsilon * torch.randn_like(x0)
                x1 = x0 + perturbation

                pred0 = ogn(x0,ginput.edge_index)[0].mean
                pred1 = ogn(x1,ginput.edge_index)[0].mean
                delta0 = torch.norm(x1 - x0)
                delta1 = torch.norm(pred1 - pred0)
                delta1 = torch.clamp(delta1, min=1e-12)

                lyapunov = torch.log(delta1 / delta0)
                lyapunov_values.append(lyapunov.item())


        # 指标
        predictions = np.array(predictions)
        true_values = np.array(true_values)

        mean_mse = np.mean(inference_errors)
        mean_kl = np.mean(kl_errors)
        mean_lyapunov = np.mean(lyapunov_values)

        y_true_flat = true_values.flatten()
        y_pred_flat = predictions.flatten()

        r2 = r2_score(y_true_flat,y_pred_flat)

        # 输出结果
        print(f"MSE: {mean_mse:.6f}")
        print(f"KL divergence: {mean_kl:.6f}")
        print(f"R²: {r2:.6f}")
        print(f"Lyapunov: {mean_lyapunov:.6f}")


        # 保存结果
        all_results.append({"file": file_name,"MSE": mean_mse,"KL": mean_kl,"R2": r2,"Lyapunov": mean_lyapunov})
        # ====================================================

    except Exception as e:
        print(f"文件处理失败: {file_name}")
        print(f"错误信息: {e}")
        continue

# 保存总结果
results_df = pd.DataFrame(all_results)
save_csv_path = os.path.join(RESULT_DIR,"batch_training_results.csv")
results_df.to_csv(save_csv_path,index=False,encoding='utf-8-sig')
print("全部文件处理完成")
print(f"结果已保存到:\n{save_csv_path}")
