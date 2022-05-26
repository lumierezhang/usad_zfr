import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import statistics
import torch.utils.data as data_utils

from utils import *
from usad_cp import *
from sklearn import preprocessing, metrics
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

#Read data
normal = pd.read_csv("../input/SWaT_Dataset_Normal_v1.csv")  # normal中存储每一行数据，每一行为一个向量
normal = normal.drop(["Timestamp" , "Normal/Attack"], axis = 1)  # 丢弃第一列和最后一列的标签 axis=1，
normal = pd.DataFrame(np.array(normal)[:489612])
# normal = pd.DataFrame(np.array(normal)[:6132])
print(normal.shape)  # （495000行，51列 ）
#
# #Downsampling
# normal = normal.groupby(np.arange(len(normal.index)) // 5).mean()
# print(normal.shape)

# Transform all columns into float64
for i in list(normal):  # i=49（从每一列标签开始循环）
    normal[i] = normal[i].apply(lambda x: str(x).replace(",", "."))
normal = normal.astype(float)

min_max_scaler = preprocessing.MinMaxScaler()  # 将每个特征中的最小值变成0，最大值变成1
x = normal.values  # （495000，51）x，原数据 将每一行数据作为一个向量
x_scaled = min_max_scaler.fit_transform(x)  # 对x数据进行统一处理
normal = pd.DataFrame(x_scaled)  # （495000，51） 设置idex  normal=x_scaled
print(normal.head(2))

#Read data
attack = pd.read_csv("../input/SWaT_Dataset_Attack_v0.csv",sep=";")
labels = [ float(label!= 'Normal' ) for label  in attack["Normal/Attack"].values]
labels = labels[:440652]
# labels = labels[:6132]
attack = attack.drop(["Timestamp", "Normal/Attack" ], axis=1)
attack = pd.DataFrame(np.array(attack)[:440652])
# attack = pd.DataFrame(np.array(attack)[:6132])
print(attack.shape)

# #Downsampling the attack data
# attack = attack.groupby(np.arange(len(attack.index)) // 5).mean()
# print(attack.shape)
#
# #Downsampling the labels
# labels_down=[]
#
# for i in range(len(labels)//5):
#     if labels[5*i:5*(i+1)].count(1.0):
#         labels_down.append(1.0) #Attack
#     else:
#         labels_down.append(0.0) #Normal
#
# #for the last few labels that are not within a full-length window
# if labels[5*(i+1):].count(1.0):
#     labels_down.append(1.0) #Attack
# else:
#     labels_down.append(0.0) #Normal
#
# print(len(labels_down))

# Transform all columns into float64
for i in list(attack):
    attack[i] = attack[i].apply(lambda x: str(x).replace(",", "."))
attack = attack.astype(float)

# 最大最小归一化
x = attack.values
x_scaled = min_max_scaler.transform(x)  # x_scaled=attack变换值
attack = pd.DataFrame(x_scaled)  # 赋idex
print(attack.head(2))

window_size=12
windows_normal=normal.values[np.arange(window_size)[None, :] + np.arange(normal.shape[0]-window_size)[:, None]]  # 数组 float64 变维,三维，作用于行和列（494988=495000-12，12*51）
print(windows_normal.shape)
windows_attack=attack.values[np.arange(window_size)[None, :] + np.arange(attack.shape[0]-window_size)[:, None]]   # 数组 float64（449907=449919-12，12*51）
print(windows_attack.shape)



BATCH_SIZE = 612
N_EPOCHS = 1
hidden_size = 100

w_size = windows_normal.shape[1]*windows_normal.shape[2] # 12*51
z_size = windows_normal.shape[1]*hidden_size  # 1200

windows_normal_train = windows_normal[:int(np.floor(.8 *  windows_normal.shape[0]))]  # （9792，12*51） 数组 float64
windows_normal_val = windows_normal[int(np.floor(.8 *  windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]  # float64（2448，12*51）数组


train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_train).float().view(([windows_normal_train.shape[0], 1, w_size]))
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)  # (20)

val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_val).float().view(([windows_normal_val.shape[0], 1, w_size]))
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)  # (4)

test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_attack).float().view(([windows_attack.shape[0], 1, w_size]))
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)  # (20)

model = UsadModel(w_size, z_size)  # 经过三次线性平展以及一次将小于0的数取0进行编码，解码多一次非线性激活函数
model = to_device(model, device)
print("Enter training")
history = training(N_EPOCHS, model, train_loader, val_loader)  # 训练并计算每次循环的损失函数既优化的过程
print("Leave training")
plot_history(history)

torch.save({
            'encoder': model.encoder.state_dict(),
            'decoder1': model.decoder1.state_dict(),
            'decoder2': model.decoder2.state_dict()
            }, "model.pth")  # 模型的保存

#test
checkpoint = torch.load("model.pth")
model.encoder.load_state_dict(checkpoint['encoder'])   # 分别取出
model.decoder1.load_state_dict(checkpoint['decoder1'])
model.decoder2.load_state_dict(checkpoint['decoder2'])
results = testing(model,test_loader)  # (20)
# print(results)
# print(len(results))

windows_labels=[]
for i in range(len(labels)-window_size):  # （12252-12）
    windows_labels.append(list(np.int_(labels[i:i+window_size])))  # （449907）

y_test = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels ]  # (12240,1)
# print(len(y_test))
y_test = np.array(y_test).reshape((-1,1))
# y_pred = np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
#                               results[-1].flatten().detach().cpu().numpy()])

y_pred = results
y_pred = np.array(y_pred).reshape((-1,1))

threshold1 = ROC(y_test, y_pred)  # true,score
print(threshold1)
# threshold = 0.450 # Decide on your own threshold
dist = np.linalg.norm(y_test - y_pred,axis=1)
score = dist.copy()
score.sort()
cutoff = int(0.999 * len(score))
threshold = score[cutoff]
print(threshold)
y_pred_label = [1.0 if (score > threshold) else 0 for score in y_pred]
prec=precision_score(y_test,y_pred_label,pos_label=1)
recall=recall_score(y_test,y_pred_label,pos_label=1)
f1=f1_score(y_test,y_pred_label,pos_label=1)
print('precision=',prec)
print('recall=',recall)
print('f1=',f1)


# AUC=0.79
# f1_score=0.72
# Epoch [0], val_loss1: 0.1912, val_loss2: 0.1913




