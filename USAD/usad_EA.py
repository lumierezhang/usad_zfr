import torch
import torch.nn as nn
from torch.nn import init
import  os
#import psutil
#import  profile
import sys
import time

# from memory_profiler import profile

import torch.nn.functional as F
from utils import *
device = get_default_device()

# cnn pool(mean max) bn  dropout
# randomforest、boost(XG）、knn

class Unflatten(nn.Module):
    def __init__(self,channel=1):
        super(Unflatten, self).__init__()
        self.channel = channel
    def forward(self,d):
        return d.view(d.size(0),self.channel,d.size(1))


class ExternalAttention(nn.Module):

    def __init__(self, d_model, S):
        super().__init__()
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)  # 是每一行和为1
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        attn = self.mk(queries)  # bs,n,S   (612,1,12)
        # print(attn.size(),'attn1 size')
        attn = self.softmax(attn)  # bs,n,S ()
        attn = attn / torch.sum(attn, dim=2, keepdim=True)  # bs,n,S
        out = self.mv(attn)  # bs,n,d_model

        return out


class Encoder(nn.Module):
  def __init__(self, in_size, latent_size):  # (612,1200)
    super().__init__()
    self.EA = ExternalAttention(d_model=612, S=12)
    self.conv1 = nn.Conv1d(in_channels=1, out_channels=204, kernel_size=4, stride=2)
    self.conv2 = nn.Conv1d(in_channels=204, out_channels=34, kernel_size=3, stride=2)  # (306.0,153.0)
    self.flatten = nn.Flatten()
    self.linear3 = nn.Linear(34*152, latent_size)
    self.relu = nn.ReLU(True)
    self.sigmoid = nn.Sigmoid()

  def forward(self, w):
    try:
        out = self.EA(w)  # (612,1,612)
        # print(out.size(),'w by EA')
        out = self.conv1(out)  # (612,2,305)
        # print(out.shape, '#1')
        out = self.relu(out)
        out = self.conv2(out)  # (612,1,152)
        # print(out.shape, '#2')
        out = self.relu(out)
        out = self.flatten(out)  # (612,152)
        out = self.linear3(out)  # (612,1200)
        out = self.relu(out)
        z = self.sigmoid(out)
    except BaseException as e:
        print("Here1")
        print(e)

    return z

class Decoder(nn.Module):
  def __init__(self, latent_size, out_size):
    super().__init__()
    # self.conv1 = nn.Conv1d(in_channels=1, out_channels=34, kernel_size=6, stride=8)
    # self.conv2 = nn.Conv1d(in_channels=34, out_channels=1, kernel_size=1, stride=1, padding=78, padding_mode='reflect')
    # self.flatten = nn.Flatten()
    # self.linear3 = nn.Linear(1*306, out_size)
    # self.relu = nn.ReLU(True)
    self.linear1 = nn.Linear(latent_size, int(out_size / 4))
    self.linear2 = nn.Linear(int(out_size / 4), int(out_size / 2))
    self.linear3 = nn.Linear(int(out_size / 2), out_size)
    self.relu = nn.ReLU(True)
    self.sigmoid = nn.Sigmoid()
    self.unflatten2 = Unflatten()

  def forward(self, z):
    try :
        out = self.linear1(z)  # (z_shape:(612,1,1200)\out_shape:())
        # print(out.shape, '#3')
        out = self.relu(out)
        out = self.linear2(out)
        # print(out.shape, '#4')
        out = self.relu(out)
        out = self.linear3(out)  # (612,612)
        out = self.relu(out)
        out = self.sigmoid(out)
        w = self.unflatten2(out)
    except BaseException as d:
        print("Here2")
        print(d)
    return w

class UsadModel(nn.Module):
  def __init__(self, w_size, z_size):  # (612,1200)
    super().__init__()
    self.encoder = Encoder(w_size, z_size)
    self.decoder1 = Decoder(z_size, w_size)
    self.decoder2 = Decoder(z_size, w_size)

  def training_step(self, batch, n):
    # print('begin to train')
    batch_tem = batch.reshape(B_S, 612)
    z = self.encoder(batch)  # （612，1200）
    # z_tem = z.reshape(B_S, 1, 1200)  # (612,1,1200)

    w1 = self.decoder1(z)  # （612，612）
    # w1_tem = w1.reshape(B_S, 1, 612)

    w2 = self.decoder2(z)  # （612,612）
    w3_tem = self.encoder(w1)  # (612,1200)
    # w3_tem1 = w3_tem.reshape(B_S, 1, 1200)
    w3 = self.decoder2(w3_tem)  # (612,612)

    loss1 = 1/n*torch.mean((batch_tem-w1)**2)+(1-1/n)*torch.mean((batch_tem-w3)**2)
    loss2 = 1/n*torch.mean((batch_tem-w2)**2)-(1-1/n)*torch.mean((batch_tem-w3)**2)
    return loss1, loss2

  def validation_step(self, batch, n):
    z = self.encoder(batch)
    # z_tem = z.reshape(B_S, 1, 1200)

    # print("validation_step 1")
    w1 = self.decoder1(z)
    # w1_tem = w1.reshape(B_S, 1, 612)

    # print("validation_step 2")
    w2 = self.decoder2(z)
    # print("validation_step 3")
    w3_tem = self.encoder(w1)
    # w3_tem1 = w3_tem.reshape(B_S, 1, 1200)
    # print("validation_step 4")
    w3 = self.decoder2(w3_tem)
    # print("validation_step 5")
    # print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
    # print("validation_step 6")
    loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
    # print("validation_step 7")
    return {'val_loss1': loss1.item(), 'val_loss2': loss2.item()}

  def validation_epoch_end(self, outputs):
    # print("Enter validation_epoch_end")
    b_loss1 = []  # new
    b_loss2 = []  # new

    batch_losses1 = [x['val_loss1'] for x in outputs]

    for loss1 in list(batch_losses1):  # new
        b_loss1.append(torch.tensor(loss1, requires_grad=True))
    epoch_loss1 = torch.stack(b_loss1).mean()

    #epoch_loss1 = mean(batch_losses1)

    # print(b_loss1)
    # print(epoch_loss1)

    batch_losses2 = [x['val_loss2'] for x in outputs]
    for loss2 in list(batch_losses2):  # new
        b_loss2.append(torch.tensor(loss2, requires_grad=True))
    epoch_loss2 = torch.stack(b_loss2).mean()

    #epoch_loss2 = mean(batch_losses2)

    return {'val_loss1': epoch_loss1.item(), 'val_loss2': epoch_loss2.item()}

  def epoch_end(self, epoch, result):
    print("Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}".format(epoch, result['val_loss1'], result['val_loss2']))

def evaluate(model, val_loader, n):

    total_outputs = []
    for [batch] in val_loader:
        # print(type(batch))
        outputs = model.validation_step(to_device(batch, device), n)
        # print(type(outputs))
        # print(outputs)
        # print("Size of outputs: %d" % sys.getsizeof(outputs))
        # print("Size of total_outputs: %d" % sys.getsizeof(total_outputs))
        # print("len: %d" % len(total_outputs))
        total_outputs.append(outputs)  # (存放损失函数值)

    # print("evaluate loop ends")

    # outputs = [model.validation_step(to_device(batch,device), n) for [batch] in val_loader]

    return model.validation_epoch_end(total_outputs)

def training(epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam):  # (3，moudel)
    history = []
    optimizer1 = opt_func(list(model.encoder.parameters())+list(model.decoder1.parameters()))
    optimizer2 = opt_func(list(model.encoder.parameters())+list(model.decoder2.parameters()))
    for epoch in range(epochs):
        for [batch] in train_loader:  # (612,1,612)
            batch = to_device(batch, device)

            # Train AE1
            loss1, loss2 = model.training_step(batch, epoch + 1)
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()
            loss1.item()
            loss2.item()

            # Train AE2
            loss1, loss2 = model.training_step(batch,  epoch + 1)
            loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()
            loss1.item()
            loss2.item()
        # print("Training 1")
        result = evaluate(model, val_loader, epoch+1)
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def EuclideanDistance(t1,t2):
    dim=len(t1.size())
    if dim==2:
        N,C=t1.size()
        M,_=t2.size()
        dist = -2 * torch.matmul(t1, t2.permute(1, 0))
        dist += torch.sum(t1 ** 2, -1).view(N, 1)
        dist += torch.sum(t2 ** 2, -1).view(1, M)
        dist=torch.sqrt(dist)
        return dist
    elif dim==3:
        B,N,_=t1.size()
        _,M,_=t2.size()
        dist = -2 * torch.matmul(t1, t2.permute(0, 2, 1))
        dist += torch.sum(t1 ** 2, -1).view(B, N, 1)
        dist += torch.sum(t2 ** 2, -1).view(B, 1, M)
        dist=torch.sqrt(dist)
        return dist
    else:
        print('error...')


def testing(model, test_loader, alpha=0.5, beta=0.5):
    results = []
    results_tem = []
    for [batch] in test_loader:  # (312,1,612)
        batch = to_device(batch, device)
        try :
            w0 = model.encoder(batch)  # (312,1200)
            # w0_tem = w0.reshape(B_S, 1, 1200)
            w1 = model.decoder1(w0)  # (312,612)
            # w1_re = w1.reshape(B_S, 1, 612)
            # w1_tem = w1.reshape(B_S, 1, 612)
            w2_tem = model.encoder(w1)  # (612,1200)
            # w2_tem1 = w2_tem.reshape(B_S, 1, 1200)
            w2 = model.decoder2(w2_tem)  # （312，612）
            # w2_re = w2.reshape(B_S, 1, 612)
            # print('w2')

            distance_w1 = F.pairwise_distance(batch, w1, p=2).detach().numpy()  # (312,612)
            distance_w2 = F.pairwise_distance(batch, w2, p=2).detach().numpy()  # (312,612)

            distance_w1_mean = distance_w1.mean(axis=1)
            distance_w2_mean = distance_w2.mean(axis=1)
            # print(distance_w1_mean,'%%1')
            # print(distance_w2_mean,'%%2')

            # distance_w1_mean = distance_w1_mean.tolist()
            # distance_w2_mean = distance_w2_mean.tolist()

            results_tem = alpha*distance_w1_mean + beta*distance_w2_mean
            for value in results_tem:
                results.append(value)
#          results.append(alpha*torch.mean(F.pairwise_distance(batch, w1, p=2).item(), axis=1)+beta*torch.mean(F.pairwise_distance(batch, w2, p=2).item(), axis=1))
#          results.append(alpha*torch.mean(EuclideanDistance(batch_res, w1), axis=1)+beta*torch.mean(EuclideanDistance(batch_res, w2), axis=1))
        except BaseException as j:
            print("Here3")
            print(j)

    return results


