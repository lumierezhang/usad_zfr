import torch
import torch.nn as nn
from torch.nn import init

from utils import *

device = get_default_device()


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
        # print('enter EA')
        attn = self.mk(queries)  # bs,n,S   (612,1,12)
        # print(attn.size(),'attn1 size')
        attn = self.softmax(attn)  # bs,n,S ()
        attn = attn / torch.sum(attn, dim=1, keepdim=True)  # bs,n,S
        out = self.mv(attn)  # bs,n,d_model

        return out

class Encoder(nn.Module):
    def __init__(self, in_size, latent_size):  # (612,1200)
        super().__init__()
        self.EA = ExternalAttention(d_model=612, S=12)
        self.linear1 = nn.Linear(in_size, int(in_size / 2))  # (612,306.0)
        self.linear2 = nn.Linear(int(in_size / 2), int(in_size / 4))  # (306.0,153.0)
        self.linear3 = nn.Linear(int(in_size / 4), latent_size)
        self.relu = nn.ReLU(True)

    def forward(self, w):
        out = self.EA(w)  #
        out = self.linear1(out)  # (7919,306)
        out = self.relu(out)
        out = self.linear2(out)  # (7919,153)
        out = self.relu(out)
        out = self.linear3(out)  # (7919,1200)
        z = self.relu(out)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(latent_size, int(out_size / 4))
        self.linear2 = nn.Linear(int(out_size / 4), int(out_size / 2))
        self.linear3 = nn.Linear(int(out_size / 2), out_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        out = self.linear1(z)  # (7919,153)
        out = self.relu(out)
        out = self.linear2(out)  # (7919,306)
        out = self.relu(out)
        out = self.linear3(out)  # (7919,612)
        w = self.sigmoid(out)
        return w


class UsadModel(nn.Module):
    def __init__(self, w_size, z_size):  # (612,1200)
        super().__init__()
        self.encoder = Encoder(w_size, z_size)
        self.decoder1 = Decoder(z_size, w_size)
        self.decoder2 = Decoder(z_size, w_size)

    def training_step(self, batch,n):
        z = self.encoder(batch)  # (7919,1200)
        w1 = self.decoder1(z)  # (7919,612)
        w2 = self.decoder2(z)  # (7919,612)
        w3 = self.decoder2(self.encoder(w1))


        loss1 = 1 / n * torch.mean((batch - w1) ** 2) + (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        loss2 = 1 / n * torch.mean((batch - w2) ** 2) - (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        return loss1, loss2

    def validation_step(self, batch, n):
#        loss_func = nn.SmoothL1Loss(reduction='mean')
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))

        loss1 = 1 / n * torch.mean((batch - w1) ** 2) + (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        loss2 = 1 / n * torch.mean((batch - w2) ** 2) - (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        return {'val_loss1': loss1, 'val_loss2': loss2}

    def validation_epoch_end(self, outputs):
        batch_losses1 = [x['val_loss1'] for x in outputs]
        # print(batch_losses1)
        epoch_loss1 = torch.stack(batch_losses1).mean()
        # print(epoch_loss1)
        batch_losses2 = [x['val_loss2'] for x in outputs]
        epoch_loss2 = torch.stack(batch_losses2).mean()
        return {'val_loss1': epoch_loss1.item(), 'val_loss2': epoch_loss2.item()}

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}".format(epoch, result['val_loss1'], result['val_loss2']))


def evaluate(model, val_loader,n):
    outputs = [model.validation_step(to_device(batch, device),n) for [batch] in val_loader]
    return model.validation_epoch_end(outputs)


def training(epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam):  # (3，moudel)
    history = []
    optimizer1 = opt_func(list(model.encoder.parameters()) + list(model.decoder1.parameters()))
    optimizer2 = opt_func(list(model.encoder.parameters()) + list(model.decoder2.parameters()))
    for epoch in range(epochs):
        for [batch] in train_loader:  # （7919，612）
            batch = to_device(batch, device)

            # Train AE1
            loss1, loss2 = model.training_step(batch, epoch + 1)
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()

            # Train AE2
            loss1, loss2 = model.training_step(batch, epoch + 1)
            loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()

        result = evaluate(model, val_loader, epoch+1)
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def testing(model, test_loader, alpha=.5, beta=.5):
    results = []
    for [batch] in test_loader:  # （7919，612）
        batch = to_device(batch, device)
        w1 = model.decoder1(model.encoder(batch))   # （7919，612）
        w2 = model.decoder2(model.encoder(w1))   # （7919，612）
        results.append(alpha * torch.mean((batch - w1) ** 2, axis=1) + beta * torch.mean((batch - w2) ** 2, axis=1))  # (每次含有7919个)
#   print(results1)
    return results