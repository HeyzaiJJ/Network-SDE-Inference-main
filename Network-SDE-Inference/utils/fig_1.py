# %%
import numpy as np
import pandas as pd
import os
import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt


import sys

sys.path.append("../Network-SDE-Inference/utils")
import NeuGNN_model
import Self_func
import Interaction_func
from NeuGNN_model import *
from Self_func import *
from Interaction_func import *

# %%
USE_CUDA = False
# %% md
## Import data and topology
# %%
Timeseries = pd.read_csv('../Data/TimeSeries&Topologies/Lorenz_stochastic_in005_200.csv',
                         encoding='utf-8', header=None)
# %%

# %%
Adj = pd.read_csv('../Data/TimeSeries&Topologies/unweighted_adj_20nodes.csv', encoding='utf-8',
                  header=None)
# %%
Adj.shape[0]
# %%
Num_nodes = 20
Dimension = 3
dim = Dimension * 1
# %%
time = Timeseries.values
timeseries = time.reshape((-1, Num_nodes, Dimension))  # checked, correct input data
# %%
delt_t = 0.01
# %% md
## Calculate the numerical difference
# %%
timeseries_t0 = timeseries[:-2, :, :]
timeseries_t1 = timeseries[1:-1, :, :]
dX = (timeseries_t1 - timeseries_t0)
dXdt = (timeseries_t1 - timeseries_t0) / delt_t
# %%
data = np.concatenate((timeseries[:-2, :, :], dXdt), axis=2)


# %% md
## Check the topology and edge list (source to target)
# %%
def get_edge_index(Adj):
    num_nodes = Adj.shape[0]
    Adj = Adj.values
    edge_index = torch.from_numpy(np.array(np.where(Adj)))
    return edge_index


# %%
edge_index = get_edge_index(Adj)  # target to source

index = [1, 0]
edge_index = edge_index[index]  # source to target
# checked, correct input edge_index data
# %% md
## Construct the data, inluding goal and input data
# %%
# data = data.reshape((10,-1,4,6))
# goal_data = np.concatenate((data[1:-1,:,0:Dimension],data[0:-2,:,Dimension:Dimension*2]),axis=2)
goal_data = data[1:-1, :, 0:Dimension]
mapping_data = data[0:-2, :, 0:Dimension]
# %%
# tmp1 = np.concatenate([mapping_data[:, i] for i in range(0, mapping_data.shape[1], 1)])
# tmp2 = np.concatenate([goal_data[:, i] for i in range(0, goal_data.shape[1], 1)])
# X = torch.from_numpy(mapping_data)
# y = torch.from_numpy(goal_data)

X = torch.as_tensor(np.array(mapping_data).astype('float'))
y = torch.as_tensor(np.array(goal_data).astype('float'))
# %%
from sklearn.model_selection import train_test_split

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
X_train = X_train.float()
y_train = y_train.float()
X_test = X_test.float()
y_test = y_test.float()
# %%
import torch
from torch import nn
from torch.functional import F
from torch.optim import Adam
from torch_geometric.nn import MetaLayer, MessagePassing
from torch_geometric.data import Data, DataLoader

# %%
aggr = 'add'
hidden = 100
model = 'Lorenz'
msg_dim = 1
n_f = mapping_data.shape[2]
# %% md
## Instant model
# %%
ogn = SDIunweighted(model, n_f, msg_dim, Dimension, delt_t, hidden=hidden, edge_index=edge_index, aggr=aggr)
# %% md
## Use these "_over_time" to save estimated values
# %%
messages_over_time = []
selfDyn_over_time = []
diffusion_over_time = []
ogn = ogn
# %%
x = X_train[1]
y = y_train[1]
_q = Data(
    x=X_train[1],
    edge_index=edge_index,
    y=y_train[1])
# %%
ogn.loss(_q)
# %%
batch = 64
trainloader = DataLoader(
    [Data(
        Variable(X_train[i]),
        edge_index=edge_index,
        y=Variable(y_train[i])) for i in range(len(y_train))],
    batch_size=batch,
    shuffle=True
)

testloader = DataLoader(
    [Data(
        X_test[i],
        edge_index=edge_index,
        y=y_test[i]) for i in range(len(y_test))],
    batch_size=256,
    shuffle=True
)
# %%
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
# %%
init_lr = 1e-3

opt = torch.optim.Adam(ogn.parameters(), lr=init_lr, weight_decay=1e-8)

# total_epochs = 200
total_epochs = 50

batch_per_epoch = 1000

sched = OneCycleLR(opt, max_lr=init_lr,
                   steps_per_epoch=batch_per_epoch,  # len(trainloader),
                   epochs=total_epochs, final_div_factor=1e5)

batch_per_epoch
# %%
epoch = 0
from tqdm import tqdm
# %%
import numpy as onp

onp.random.seed(0)
test_idxes = onp.random.randint(0, len(X_test), 1000)

# Record messages over test dataset here:
newtestloader = DataLoader(
    [Data(
        X_test[i],
        edge_index=edge_index,
        y=y_test[i]) for i in test_idxes],
    batch_size=len(X_test),
    shuffle=False
)

# newtestloader = DataLoader(
#     [Data(
#         X_train[i],
#         edge_index=edge_index,
#         y=y_train[i]) for i in test_idxes],
#     batch_size=len(X_train),
#     shuffle=False
# )
# %%
test_idxes.shape
# %%
import numpy as onp
import pandas as pd


def get_messages(ogn):
    def get_message_info(tmp):
        ogn.cpu()

        s1 = tmp.x[tmp.edge_index[0]]  # source
        s1 = s1[:, 0]
        # print(s1)
        s2 = tmp.x[tmp.edge_index[1]]  # target
        s2 = s2[:, 0]
        # print(s2)
        tmp = torch.cat([s2, s1])  # tmp --> xi,xj
        tmp = tmp.reshape(2, -1)
        tmp = tmp.t()  # tmp has shape [E, 2 * in_channels]

        m12 = ogn.msg_fnc(tmp)

        all_messages = torch.cat((
            tmp,
            m12), dim=1)
        if dim == 1:
            columns = [elem % (k) for k in range(1, 3) for elem in 'x%d'.split(' ')]
            columns += ['e%d' % (k,) for k in range(msg_dim)]
        if dim == 2:
            columns = [elem % (k) for k in range(1, 3) for elem in 'x%d'.split(' ')]
            columns += ['e%d' % (k,) for k in range(msg_dim)]
        elif dim == 3:
            columns = [elem % (k) for k in range(1, 3) for elem in 'x%d'.split(' ')]
            columns += ['e%d' % (k,) for k in range(msg_dim)]

        return pd.DataFrame(
            data=all_messages.cpu().detach().numpy(),
            columns=columns
        )
        # print(all_messages.shape)
        return pd.DataFrame(all_messages)

    msg_info = []
    for i, g in enumerate(newtestloader):
        msg_info.append(get_message_info(g))

    msg_info = pd.concat(msg_info)
    #     if dim == 1:
    #         msg_info['dx'] = msg_info.x1 - msg_info.x2
    #         msg_info['dy'] = msg_info.y1 - msg_info.y2
    #         msg_info['dz'] = msg_info.z1 - msg_info.z2
    #         msg_info['r'] = np.sqrt(
    #                 (msg_info.dx)**2(msg_info.dy)**2 + (msg_info.dz)**2)
    #     else:
    #         msg_info['dx'] = msg_info.x1 - msg_info.x2
    #         msg_info['dy'] = msg_info.y1 - msg_info.y2
    #         if dim == 2:
    #             msg_info['r'] = np.sqrt(
    #                 (msg_info.dx)**2 + (msg_info.dy)**2
    #             )
    #         elif dim == 3:
    #             msg_info['dz'] = msg_info.z1 - msg_info.z2
    #             msg_info['r'] = np.sqrt(
    #                 (msg_info.dx)**2 + (msg_info.dy)**2 + (msg_info.dz)**2
    #             )

    return msg_info


# %%
def get_selfDynamics(ogn):
    def get_selfDynamics_info(tmp):
        ogn.cpu()

        tmp = tmp.x[tmp.edge_index[1]]
        if dim == 1:
            self_dyn_x = ogn.node_fnc_x(tmp)
            self_dyn_all = torch.cat((tmp, self_dyn_x), dim=1)
            columns = ['x', 's1']

        if dim == 2:
            self_dyn_x = ogn.node_fnc_x(tmp)
            self_dyn_y = ogn.node_fnc_y(tmp)
            self_dyn_all = torch.cat((tmp, self_dyn_x, self_dyn_y), dim=1)
            columns = ['x', 'y', 's1', 's2']
        if dim == 3:
            self_dyn_x = ogn.node_fnc_x(tmp)
            self_dyn_y = ogn.node_fnc_y(tmp)
            self_dyn_z = ogn.node_fnc_z(tmp)
            self_dyn_all = torch.cat((tmp, self_dyn_x, self_dyn_y, self_dyn_z), dim=1)
            columns = ['x', 'y', 'z', 's1', 's2', 's3']

        return pd.DataFrame(
            data=self_dyn_all.cpu().detach().numpy(),
            columns=columns
        )
        return pd.DataFrame(self_dyn_all)

    selfDyn_info = []
    for i, g in enumerate(newtestloader):
        selfDyn_info.append(get_selfDynamics_info(g))

    selfDyn_info = pd.concat(selfDyn_info)
    return selfDyn_info


# %%
def get_diffusion(ogn):
    def get_diffusion_info(tmp):
        ogn.cpu()

        tmp = tmp.x[tmp.edge_index[1]]
        if dim == 1:
            self_diff_x = ogn.stochastic_x(tmp)
            self_diff_all = torch.cat((tmp, self_diff_x), dim=1)
            columns = ['x', 'd1']

        if dim == 2:
            self_diff_x = ogn.stochastic_x(tmp)
            self_diff_y = ogn.stochastic_y(tmp)
            self_diff_all = torch.cat((tmp, self_diff_x, self_diff_y), dim=1)
            columns = ['x', 'y', 'd1', 'd2']
        if dim == 3:
            self_diff_x = ogn.stochastic_x(tmp)
            self_diff_y = ogn.stochastic_y(tmp)
            self_diff_z = ogn.stochastic_z(tmp)
            self_diff_all = torch.cat((tmp, self_diff_x, self_diff_y, self_diff_z), dim=1)
            columns = ['x', 'y', 'z', 'd1', 'd2', 'd3']

        return pd.DataFrame(
            data=self_diff_all.cpu().detach().numpy(),
            columns=columns
        )
        return pd.DataFrame(self_diff_all)

    selfDiffusion_info = []
    for i, g in enumerate(newtestloader):
        selfDiffusion_info.append(get_diffusion_info(g))

    selfDiffusion_info = pd.concat(selfDiffusion_info)
    return selfDiffusion_info


# %%
recorded_models = []
# %% md
## Start training
# %%
for epoch in tqdm(range(epoch, total_epochs)):
    ogn
    total_loss = 0.0
    i = 0
    j = 0
    num_items = 0
    valid_loss = 0
    valid_num_items = 0
    while i < batch_per_epoch:
        for ginput in trainloader:
            if i >= batch_per_epoch:
                break
            opt.zero_grad()
            ginput.x = ginput.x
            ginput.y = ginput.y
            ginput.edge_index = ginput.edge_index
            ginput.batch = ginput.batch
            loss = ogn.loss(ginput)
            loss.backward()
            opt.step()
            sched.step()

            total_loss += loss.item()
            i += 1
            num_items += int(ginput.batch[-1] + 1)

    ogn.eval()
    with torch.no_grad():
        while j < batch_per_epoch:
            for ginput in testloader:
                if j >= batch_per_epoch:
                    break
                ginput.x = ginput.x
                ginput.y = ginput.y
                ginput.edge_index = ginput.edge_index
                ginput.batch = ginput.batch
                loss = ogn.loss(ginput)  # /int(ginput.batch[-1]+1)
                valid_loss += loss.item()
                valid_num_items += int(ginput.batch[-1] + 1)
                j += 1

    cur_loss = total_loss / num_items
    cur_valid_loss = valid_loss / valid_num_items
    print(cur_loss)
    print(cur_valid_loss)
    cur_msgs = get_messages(ogn)
    cur_selfdyn = get_selfDynamics(ogn)
    cur_diff = get_diffusion(ogn)
    cur_msgs['epoch'] = epoch
    cur_msgs['loss'] = cur_loss
    messages_over_time.append(cur_msgs)
    selfDyn_over_time.append(cur_selfdyn)
    diffusion_over_time.append(cur_diff)

    ogn.cpu()
    from copy import deepcopy as copy

    recorded_models.append(ogn.state_dict())
# %%
# torch.save(ogn, 'Lorenz_N20_intensity1_t50_net.pth')
# %%
# ogn = torch.load('Lorenz_N20_intensity1_t150_net.pth')
# %%
diffusion = diffusion_over_time[-1]
diffusion = pd.DataFrame(diffusion)
diffusion_new = pd.DataFrame.drop_duplicates(diffusion, subset=None, keep='first', inplace=False)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

s = diffusion_new['d1'] / np.sqrt(0.01)  # diffusion is sqrt(h)*theta(x0)

# fig = plt.figure(figsize = (10,6))
# ax1 = fig.add_subplot(2,1,1)
# #ax1.scatter(s.index, s.values)
# ax1.plot(s.index, s.values,'.')
# plt.grid()

# ax2 = fig.add_subplot(2,1,2)
# s.hist(bins=30,alpha = 0.5,ax = ax2)
# s.plot(kind = 'kde', secondary_y=True,ax = ax2)
# plt.title('The standard deviation distribusion')
# plt.grid()
# %%
import seaborn as sns
import numpy as np
from numpy.random import randn
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

sns.set_palette('deep', desat=.6)
sns.set_context(rc={'figure.figsize': (8, 5)})
np.random.seed(1425)
# %%
from matplotlib import colors as mcolors

# %%
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
# %% md
## Check the accuracy
# %%
gamma = 1
s = diffusion_new['d1'] / np.sqrt(0.01)
s1 = diffusion['x'].abs() / np.sqrt(gamma)
s2 = diffusion_new['d2'] / np.sqrt(0.01)
tmp1 = (28 - diffusion['z']) / np.sqrt(gamma)
s3 = tmp1.abs() / np.sqrt(gamma)
s4 = diffusion_new['d3'] / np.sqrt(0.01)
s5 = diffusion['y'].abs() / np.sqrt(gamma)
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, figsize=(10, 3))
c1, c2, c3 = sns.color_palette('Set1', 3)

sns.kdeplot(s, shade=True, color=colors['mediumvioletred'], label='dist1', ax=ax1)
# sns.kdeplot(s1, shade=True, color=colors['plum'], label='dist2',ax=ax1)
sns.kdeplot(s2, shade=True, color=colors['seagreen'], label='dist1', ax=ax1)
# sns.kdeplot(s3, shade=True, color=colors['mistyrose'], label='dist2',ax=ax1)
sns.kdeplot(s4, shade=True, color=colors['royalblue'], label='dist1', ax=ax1)
# sns.kdeplot(s5, shade=True, color=colors['skyblue'], label='dist2',ax=ax1)

sns.kdeplot(s1, shade=True, color=colors['plum'], label='dist2', ax=ax1)
sns.kdeplot(s3, shade=True, color=colors['yellowgreen'], label='dist2', ax=ax1)
sns.kdeplot(s5, shade=True, color=colors['skyblue'], label='dist2', ax=ax1)
plt.xlim([0, 50])
plt.ylim([0, 0.11])
plt.savefig('../Results/Figures/Lorenz_N20_intensity1_t200_dif_fig2.pdf')

# plot three dimensions' diffusion intensity density in the first subfig.
# %% md
## Check if the well-trained network can produce the similar trajectories
# %%
ogn.cpu()
ogn.load_state_dict(recorded_models[-1])
# %%
X = torch.as_tensor(np.array(mapping_data).astype('float'))
y = torch.as_tensor(np.array(goal_data).astype('float'))
x_Update = []
y_Update = []
z_Update = []
for i in range(5000):  # X.shape[0]
    _q = Data(
        x=X[i].float(),
        edge_index=edge_index,
        y=y[i].float())
    x_tmp, y_tmp, z_tmp = ogn.average_trajectories(_q)
    X_tmp = torch.cat((x_tmp, y_tmp, z_tmp), 1)
    if i < X.shape[0] - 2:
        X[i + 1] = X_tmp
    else:
        break
    x_Update.append(x_tmp.reshape(1, -1))
    y_Update.append(y_tmp.reshape(1, -1))
    z_Update.append(z_tmp.reshape(1, -1))
x_i = torch.stack(x_Update, dim=0).reshape(-1, Num_nodes)
y_i = torch.stack(y_Update, dim=0).reshape(-1, Num_nodes)
z_i = torch.stack(z_Update, dim=0).reshape(-1, Num_nodes)
# %%
time1 = pd.read_csv('../Data/TimeSeries&Topologies/Lorenz_determ_in005_200.csv', encoding='utf-8',
                    header=None)
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.style': 'normal'})
plt.rcParams.update({'font.family': 'Arial'})
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
x_tra = x_i.detach().numpy()[0:500, 0]
x_real = time1.iloc[:, 0]
fig = plt.figure(figsize=(18, 3.5))
ax1 = fig.add_subplot(1, 3, 1)
t = np.arange(0, x_tra.shape[0])
ax1.plot(t, x_tra, c='steelblue', label="inferred")
ax1.plot(t, x_real[1:x_tra.shape[0] + 1,], c='darkblue', label="real")
plt.ylabel('real vs inferred trajectory of dimension x')
plt.legend()

ax2 = fig.add_subplot(1, 3, 2)
y_tra = y_i.detach().numpy()[0:500, 0]
y_real = time1.iloc[:, 1]
t = np.arange(0, y_tra.shape[0])
ax2.plot(t, y_tra, c='steelblue', label="inferred")
ax2.plot(t, y_real[1:y_tra.shape[0] + 1,], c='darkblue', label="real")
plt.ylabel('real vs inferred trajectory of dimension y')
plt.legend()

ax3 = fig.add_subplot(1, 3, 3)
z_tra = z_i.detach().numpy()[0:500, 0]
z_real = time1.iloc[:, 2]
t = np.arange(0, z_tra.shape[0])
ax3.plot(t, z_tra, c='steelblue', label="inferred")
ax3.plot(t, z_real[1:z_tra.shape[0] + 1,], c='darkblue', label="real")
plt.ylabel('real vs inferred trajectory of dimension z')
plt.legend()
plt.savefig('../Results/Figures/Lorenz_N20_intensity1_t200_trajectory_fig2.pdf')
plt.show()
plt.close()
# %%
best_message = np.argmax([np.std(messages_over_time[-1]['e%d' % (i,)]) for i in range(msg_dim)])
# %%
bestMe = messages_over_time[-1][['e%d' % (best_message), 'x1', 'x2']]
# %%
# coup_value = 0.15*(2-bestMe['x1'].values)/(1+np.exp(-10*(bestMe['x2'].values-1))) # x1 (x_i) is target, x2 (x_j) is source
coup_value = bestMe['x2'].values
# %%
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'font.style': 'normal'})
plt.rcParams.update({'font.family': 'Arial'})
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
best_message = np.argmax([np.std(messages_over_time[-1]['e%d' % (i,)]) for i in range(msg_dim)])
bestMe = messages_over_time[-1][['e%d' % (best_message), 'x1', 'x2']]
temp = bestMe.iloc[:, 0].values
coup_value = coup_value[0:5000]
temp = temp[0:5000]

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(coup_value, temp, s=40, c='steelblue', alpha=0.2)
ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls='--', c='k', label="1:1 line")
parameter = np.polyfit(coup_value, temp, 1)
f = np.poly1d(parameter)
ax.plot(coup_value, f(coup_value), c='#ff9999', lw=1.0)
corr = np.corrcoef(coup_value, temp)[0, 1]
bbox = dict(fc='1', alpha=0.5)
plt.text(0.05, 0.9, '$R^2=%.2f$' % (corr ** 2), transform=ax.transAxes, size=12, bbox=bbox)
plt.title('True vs. Inferred Interaction')
plt.xlabel("True coupling values")
plt.ylabel("Inferred value")
plt.savefig('../Results/Figures/Lorenz_N20_intensity1_t200_interaction_fig2.pdf')
plt.show()
plt.close()

# %%
best_selfDyn = selfDyn_over_time[-1]
best_selfDyn = pd.DataFrame(best_selfDyn)
# %%
gamma = 1
sx = best_selfDyn['s1'].values
sx_true = 10 * best_selfDyn['y'] - (10 + 2 / gamma) * best_selfDyn['x']
sy = best_selfDyn['s2'].values
sy_true = (28 - best_selfDyn['z']) * best_selfDyn['x'] - (1 + 2 / gamma) * best_selfDyn['y']
sz = best_selfDyn['s3'].values
sz_true = best_selfDyn['x'] * best_selfDyn['y'] - (8 / 3 + 4 / gamma) * best_selfDyn['z']

plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.style': 'normal'})
plt.rcParams.update({'font.family': 'Arial'})
sx = best_selfDyn['s1'].values
# sx_true = best_selfDyn['y']-best_selfDyn['x']**3+3*best_selfDyn['x']**2-best_selfDyn['z']+3.24
# sy = best_selfDyn['s2'].values
# sy_true = 1-5*best_selfDyn['x']**2-best_selfDyn['y']
# sz = best_selfDyn['s3'].values
# sz_true = 0.005*(4*(best_selfDyn['x']+1.6)-best_selfDyn['z'])
sx = sx[0:5000]
sx_true = sx_true[0:5000]
sy = sy[0:5000]
sy_true = sy_true[0:5000]
sz = sz[0:5000]
sz_true = sz_true[0:5000]

fig = plt.figure(figsize=(18, 5))
ax1 = fig.add_subplot(1, 3, 1)
plt.title("true self- x dimension vs. inferred", fontsize=10)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
ax1.scatter(sx_true, sx, s=40, c='steelblue', alpha=0.2)
ax1.plot((0, 1), (0, 1), transform=ax1.transAxes, ls='--', c='k', label="1:1 line")
parameterx = np.polyfit(sx_true, sx, 1)
fx = np.poly1d(parameterx)
ax1.plot(sx_true, fx(sx_true), c='palevioletred', lw=1)
corrx = np.corrcoef(sx_true, sx)[0, 1]
bbox = dict(fc='1', alpha=0.5)
plt.text(0.05, 0.9, '$R^2=%.2f$' % (corrx ** 2), transform=ax1.transAxes, size=15, bbox=bbox)
plt.xlabel("True self x")
plt.ylabel("Inferred self x")

ax2 = fig.add_subplot(1, 3, 2)
plt.title("true self- y dimension vs. inferred", fontsize=10)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
ax2.scatter(sy_true, sy, s=40, c='steelblue', alpha=0.2)
ax2.plot((0, 1), (0, 1), transform=ax2.transAxes, ls='--', c='k', label="1:1 line")
parametery = np.polyfit(sy_true, sy, 1)
fy = np.poly1d(parametery)
ax2.plot(sy_true, fy(sy_true), c='palevioletred', lw=1)
corry = np.corrcoef(sy_true, sy)[0, 1]
bbox = dict(fc='1', alpha=0.5)
plt.text(0.05, 0.9, '$R^2=%.2f$' % (corry ** 2), transform=ax2.transAxes, size=15, bbox=bbox)
plt.xlabel("True self y")
plt.ylabel("Inferred self y")

ax3 = fig.add_subplot(1, 3, 3)
plt.title("true self- z dimension vs. inferred", fontsize=10)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
ax3.scatter(sz_true, sz, s=40, c='steelblue', alpha=0.2)
ax3.plot((0, 1), (0, 1), transform=ax3.transAxes, ls='--', c='k', label="1:1 line")
parameterz = np.polyfit(sz_true, sz, 1)
fz = np.poly1d(parameterz)
ax3.plot(sz_true, fz(sz_true), c='palevioletred', lw=1)
corrz = np.corrcoef(sz_true, sz)[0, 1]
bbox = dict(fc='1', alpha=0.5)
plt.text(0.05, 0.9, '$R^2=%.2f$' % (corrz ** 2), transform=ax3.transAxes, size=15, bbox=bbox)
plt.xlabel("True self z")
plt.ylabel("Inferred self z")
plt.savefig('../Results/Figures/Lorenz_N20_intensity1_t200_fig2.pdf')
plt.show()
plt.close()
# %% md
## Learn the stochastic differential equations
# %%
import Self_func
# %%
import Interaction_func
# %%
from Self_func import *
from Interaction_func import *

# %%
"""Construct the elementary matrix with pre-defined library"""
xi = bestMe['x1'].values
xj = bestMe['x2'].values
Matrix = ElementaryFunctions_Matrix(xi, xj)
Matrix = Matrix.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
# %%
from sklearn.preprocessing import normalize
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error

# %%
goal = bestMe['e0'].values.reshape(-1, 1)
goal = pd.DataFrame(data=goal, columns=['e0'])
# %%
X = Matrix.copy()
y = goal.copy()
# %%
"""Normalization process, eliminate order of magnitude differences"""
X_mat = X.values
y_mat = y.values
x_norml1 = []
y_norml1 = []
num = np.shape(X_mat)[1]
num2 = 1
L = np.shape(X_mat)[0]

for i in range(0, num):
    x_norml1.append(sum(abs(X_mat[:, i])))

for i in range(0, num2):
    y_norml1.append(sum(abs(y_mat[:, i])))

X = pd.DataFrame(X)
y = pd.DataFrame(y)

X[X.columns] = normalize(X[X.columns], norm='l1', axis=0) * L
y[y.columns] = normalize(y[y.columns], norm='l1', axis=0) * L

X_col = X.columns
Xin = X.iloc[:, :]
out = np.array(y)
y1 = (out[:, 0])
# %%
reg1 = LassoCV(cv=5, fit_intercept=False, n_jobs=-1, max_iter=1000, normalize=False).fit(Xin, y1)
print(reg1.score(Xin, y1))
print('Best threshold: %.3f' % reg1.alpha_)
# %%
for i in range(len(reg1.coef_)):
    reg1.coef_[i] = reg1.coef_[i] * y_norml1[0] / x_norml1[i]
# %%
coef1 = pd.Series(reg1.coef_, index=X_col)
imp_ = pd.concat([coef1.sort_values(key=abs).head(int(0)),
                  coef1.sort_values(key=abs).tail(int(10))])
imp_no_cons = imp_ + (1e-10)
print("Elementary functions discovered by Phase 1 without constant.")
print(imp_no_cons)
# %%
from math import log


def calculate_aic(n, mse, num_params):
    aic = n * log(mse) + 2 * num_params
    return aic


# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def terms_sort_fit(X_lib, Y_goal, intercept):
    reg = LassoCV(cv=5, fit_intercept=intercept, n_jobs=-1, max_iter=1000, normalize=False).fit(X_lib, Y_goal)
    coef = pd.Series(reg.coef_, index=X_lib.columns)
    if intercept == True:
        coef['constant'] = reg.intercept_
        num_params = len(coef)
    else:
        num_params = len(coef)
    P = X_lib
    Score = reg.score(X_lib, Y_goal)
    yhat = reg.predict(P)
    mse = mean_squared_error(Y_goal, yhat)
    aic = calculate_aic(len(Y_goal), mse, num_params)
    # print('label of function: %.3f' % time)
    sort = coef.sort_values()
    print(coef)
    return Score, mse, aic


# %%
X = pd.DataFrame()
Score_list = np.zeros(shape=(imp_no_cons.shape[0], 1))
MSE_list = np.zeros(shape=(imp_no_cons.shape[0], 1))
AIC_list = np.zeros(shape=(imp_no_cons.shape[0], 1))
y = goal.copy()
for i in range(1, imp_no_cons.shape[0] + 1):
    tmp = Matrix.copy()[imp_no_cons.index[-i]]
    if i == 1:
        X = Matrix.copy()[imp_no_cons.index[-1]].values.reshape(-1, 1)
        X = pd.DataFrame(X)
    else:
        X = pd.concat([X, tmp], axis=1)
    Score, mse, aic = terms_sort_fit(X, y, False)
    Score_list[i - 1] = Score
    MSE_list[i - 1] = mse
    AIC_list[i - 1] = aic
    print(Score, mse, aic, imp_no_cons.index[-i])
    if Score > 0.95:
        break
# %%
# Visualization
Index = np.arange(1, imp_no_cons.shape[0] + 1, 1)
fig = plt.figure(figsize=(10, 10))
fig.add_subplot(3, 1, 1)
l1, = plt.plot(Index, Score_list, marker='o', linestyle='dashed')
plt.xticks(Index)
plt.xlabel('The label of the equation')
plt.ylabel('values-Score')
fig.add_subplot(3, 1, 2)
l2, = plt.plot(Index, MSE_list, marker='+', linestyle='dashed')
plt.xticks(Index)
plt.xlabel('The label of the equation')
plt.ylabel('values-MSE')
fig.add_subplot(3, 1, 3)
l3, = plt.plot(Index, AIC_list, marker='o')
plt.xticks(Index)
plt.xlabel('The label of the equation')
plt.ylabel('values of AIC')
plt.show()
# %%
xi = bestMe['x1'].values
xj = bestMe['x2'].values
Matrix = ElementaryFunctions_Matrix(xi, xj)
Matrix = Matrix.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
goal = bestMe['e0'].values.reshape(-1, 1)
goal = pd.DataFrame(data=goal, columns=['e0'])
X = Matrix.copy()
y = goal.copy()
# %%
Xfind = X['xj'].values.reshape(-1, 1)
yfind = y['e0']
model_linear = LinearRegression(fit_intercept=False)
model_linear.fit(Xfind, yfind)
a = model_linear.coef_
print(a)
# %%
error_int = abs(a - 1) / (abs(a) + 1)
print('The error of interaction part:', error_int)
# %%
TimeSeries = best_selfDyn.iloc[:, 0:3].values
TimeSeries = TimeSeries.reshape(-1, Dimension)
# %%
self_matrix = self_ElementaryFunctions_Matrix(TimeSeries, Dimension, 4, PolynomialIndex=True, TrigonometricIndex=False, \
                                              ExponentialIndex=True, FractionalIndex=False, ActivationIndex=False)
self_matrix = self_matrix.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
# %%
goal = best_selfDyn['s1'].values
goal = pd.DataFrame(data=goal, columns=['s1'])
# %%
X = self_matrix.copy()
y = goal.copy()
# %%
X_mat = X.values
y_mat = y.values
x_norml1 = []
y_norml1 = []
num = np.shape(X_mat)[1]
num2 = 1
L = np.shape(X_mat)[0]

for i in range(0, num):
    x_norml1.append(sum(abs(X_mat[:, i])))

for i in range(0, num2):
    y_norml1.append(sum(abs(y_mat[:, i])))

X = pd.DataFrame(X)
y = pd.DataFrame(y)

X[X.columns] = normalize(X[X.columns], norm='l1', axis=0) * L
y[y.columns] = normalize(y[y.columns], norm='l1', axis=0) * L

X_col = X.columns
Xin = X.iloc[:, :]
out = np.array(y)
y1 = (out[:, 0])
# %%
reg1 = LassoCV(cv=5, fit_intercept=False, n_jobs=-1, max_iter=5000, normalize=False).fit(Xin, y1)
print(reg1.score(Xin, y1))
print('Best threshold: %.3f' % reg1.alpha_)
# %%
for i in range(len(reg1.coef_)):
    reg1.coef_[i] = reg1.coef_[i] * y_norml1[0] / x_norml1[i]
coef1 = pd.Series(reg1.coef_, index=X_col)
# %%
imp_coef1 = pd.concat([coef1.sort_values(key=abs).head(int(0)),
                       coef1.sort_values(key=abs).tail(int(10))])
imp_cons = imp_coef1 + (1e-10)
# imp_cons['constant'] = reg1.intercept_*y_norml1[0]/L
# imp_cons = imp_cons.sort_values(key=abs)
print("Elementary functions discovered by Phase 1 with constant.")
print(imp_cons)
# %%
X = pd.DataFrame()
Score_list = np.zeros(shape=(imp_cons.shape[0], 1))
MSE_list = np.zeros(shape=(imp_cons.shape[0], 1))
AIC_list = np.zeros(shape=(imp_cons.shape[0], 1))
y = goal.copy()
for i in range(1, imp_cons.shape[0] + 1):
    if imp_cons.index[-i] != 'constant':
        tmp = self_matrix.copy()[imp_cons.index[-i]]
        if i == 1:
            X = self_matrix.copy()[imp_cons.index[-i]].values.reshape(-1, 1)
            X = pd.DataFrame(X)
        else:
            X = pd.concat([X, tmp], axis=1)
        Score, mse, aic = terms_sort_fit(X, y, False)
    else:
        Score, mse, aic = terms_sort_fit(X, y, True)
    Score_list[i - 1] = Score
    MSE_list[i - 1] = mse
    AIC_list[i - 1] = aic
    print(Score, mse, aic, imp_cons.index[-i])
    if Score > 0.997:
        break
# %%
# Visualization
Index = np.arange(1, imp_cons.shape[0] + 1, 1)
fig = plt.figure(figsize=(10, 10))
fig.add_subplot(3, 1, 1)
l1, = plt.plot(Index, Score_list, marker='o', linestyle='dashed')
plt.xticks(Index)
plt.xlabel('The label of the equation')
plt.ylabel('values-Score')
fig.add_subplot(3, 1, 2)
l2, = plt.plot(Index, MSE_list, marker='+', linestyle='dashed')
plt.xticks(Index)
plt.xlabel('The label of the equation')
plt.ylabel('values-MSE')
fig.add_subplot(3, 1, 3)
l3, = plt.plot(Index, AIC_list, marker='o')
plt.xticks(Index)
plt.xlabel('The label of the equation')
plt.ylabel('values of AIC')
plt.show()
# %% md
# Calculate the inference error
# %%
TimeSeries = best_selfDyn.iloc[:, 0:3].values
TimeSeries = TimeSeries.reshape(-1, Dimension)
self_matrix = self_ElementaryFunctions_Matrix(TimeSeries, Dimension, 4, PolynomialIndex=True, TrigonometricIndex=False, \
                                              ExponentialIndex=True, FractionalIndex=False, ActivationIndex=False)
self_matrix = self_matrix.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
goal = best_selfDyn['s1'].values
goal = pd.DataFrame(data=goal, columns=['s1'])
X = self_matrix.copy()
y = goal.copy()
# %%
Xfind = pd.concat([X['x1'], X['x2']], axis=1)  # pd.concat([v1,v2],axis=1)
yfind = y['s1']
model_linear = LinearRegression(fit_intercept=False)
model_linear.fit(Xfind, yfind)
a = model_linear.coef_
a
# %%

# %%
model_linear.intercept_
# %%
error_self1 = abs(a[0] - [-12]) / (abs(a[0]) + abs(-12))
# %%
error_int
# %%
error_self2 = abs(a[1] - [10]) / (abs(a[1]) + abs(10))
# %%
Error = (error_self1 + error_self2 + error_int) / 3
# %%
Error
# %%
TimeSeries = best_selfDyn.iloc[:, 0:3].values
TimeSeries = TimeSeries.reshape(-1, Dimension)
self_matrix = self_ElementaryFunctions_Matrix(TimeSeries, Dimension, 4, PolynomialIndex=True, TrigonometricIndex=False, \
                                              ExponentialIndex=True, FractionalIndex=False, ActivationIndex=False)
self_matrix = self_matrix.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
goal = best_selfDyn['s3'].values
goal = pd.DataFrame(data=goal, columns=['s3'])
X = self_matrix.copy()
y = goal.copy()
# %%
Xfind = pd.concat([X['x1x2'], X['x3']], axis=1)  # pd.concat([v1,v2],axis=1)
yfind = y['s3']
model_linear = LinearRegression(fit_intercept=False)
model_linear.fit(Xfind, yfind)
a = model_linear.coef_
a
# %%
