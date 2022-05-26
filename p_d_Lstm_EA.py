import torch
import torch.nn as nn

from utils import *

device = get_default_device()


# class RegLSTM(nn.Module):
#     def __init__(self):
#         super(RegLSTM, self).__init__()
#         self.rnn = nn.LSTM(input_size=51,hidden_size=100, num_layers=2, dropout=0.1,batch_first=True)
#         # 定义回归层网络，输入的特征维度等于LSTM的输出，输出维度为1 self.reg = nn.Sequential(            nn.Linear(hidden_size, 1)        )
#     def forward(self, x):
#         h0 = torch.zeros(2,612,100)
#         c0 = torch.zeros(2,612,100)
#         x , (hi,ce)= self.rnn(x,(h0,c0))  # x = (612,12,100) hi = (2,612,100) ce = (2,612,100)
#         # seq_len, batch_size, hidden_size= x.shape
#         # x = x.view(x.shape[0], -1)
#         # print(x.shape,'after lstm x shape')
#         # print(hi.shape,'after lstm hi shape')
#         # print(ce.shape,'after lstm ce shape')
#         return (hi,ce)

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):  # (612,1200)
        super().__init__()
        self.lstm = nn.LSTM(input_size,hidden_size, num_layers=2, dropout=0.1,batch_first=True)  # (51,26)
        # self.linear1 = nn.Linear(latent_size, int(latent_size / 2))  # (612,306.0)
        # self.linear2 = nn.Linear(int(latent_size / 2), int(latent_size / 4))  # (306.0,153.0)
        # self.linear3 = nn.Linear(int(latent_size / 4), in_size)
        # self.relu = nn.ReLU(True)

    def forward(self, w,batch_size):
        h0 = torch.zeros(2,batch_size,100)
        c0 = torch.zeros(2,batch_size,100)
        x , (hi,ce)= self.lstm(w,(h0,c0))  # x = (612,12,100) hi = (2,612,100) ce = (2,612,100)
#         out = self.lstm(w)  # (612,12,100)
#         out = self.linear1(out)  # (7919,306)
#         print(out.shape,'after linear1 out shape')
#         out = self.relu(out)
#         out = self.linear2(out)  # (7919,153)
#         print(out.shape,'after linear2 out shape')
#         out = self.relu(out)
#         out = self.linear3(out)  # (7919,1200)
#         print(out.shape,'after linear4 out shape')
#         z = self.relu(out)
        return (hi,ce)


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size , hidden_size, num_layers = 2, batch_first=True, dropout=0.1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, input_size)
        # self.linear1 = nn.Linear(out_size, int(out_size *2 ))
        # self.linear2 = nn.Linear(int(out_size * 8), int(out_size * 16))
        # self.linear3 = nn.Linear(int(out_size * 16), latent_size)
        # self.relu = nn.ReLU(True)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, z,hidden):
        output, (hidden, cell) = self.lstm(z, hidden)
        prediction = self.fc(output)

        # out = self.linear1(z)  # (7919,153)
        # print(out.shape,'after D-linear1 out shape')
        # out = self.relu(out)
        # out = self.linear2(out)  # (7919,306)
        # print(out.shape,'after D-linear2 out shape')
        # out = self.relu(out)
        # out = self.linear3(out)  # (7919,612)
        # print(out.shape,'after D-linear3 out shape')
        # w = self.sigmoid(out)
        return prediction, (hidden, cell)


class UsadModel(nn.Module):
    def __init__(self,w_size,hidden_size):
        super().__init__()
        self.encoder = Encoder(w_size,hidden_size)
        self.decoder1 = Decoder(w_size,hidden_size)
        self.decoder2 = Decoder(w_size,hidden_size)


    def training_step(self, batch,n):
        batch_size, sequence_length, var_length = batch.size()  # var_length = 51 batch_size = 612 sequence = 12
        batch1 = batch.view(batch_size, sequence_length * var_length)
        z = self.encoder(batch,batch_size)  # ((2,612,26)\(2,612,26))
        inv_idx = torch.arange(sequence_length - 1, -1, -1).long()  # (12)
        w1 = []
        w2 = []
        temp_input = torch.zeros((batch_size, 1, var_length), dtype=torch.float).to(batch.device)  # (612,1,51)
        hidden = z
        for t in range(sequence_length):
            temp_input, hidden = self.decoder1(temp_input, hidden)
            temp_input, hidden = self.decoder2(temp_input, hidden)
            w1.append(temp_input)
            w2.append(temp_input)
        reconstruct_output1 = torch.cat(w1, dim=1)[:, inv_idx, :]  # (612,12,51)
        reconstruct_output2 = torch.cat(w2, dim=1)[:, inv_idx, :]  # (612,12,51)

        z2 = self.encoder(reconstruct_output1,batch_size)  # ((2,612,26)\(2,612,26))
        bs, sq, vl = reconstruct_output1.size()  # (bs = 612 \sq = 12\vl = 51)
        ii = torch.arange(sq - 1, -1, -1).long()  # (12)
        w3 = []
        tt = torch.zeros((bs, 1, vl), dtype=torch.float).to(batch.device)  # (612,1,51)
        hidden = z2
        for t in range(sq):
            tt, hidden = self.decoder2(tt, hidden)
            w3.append(temp_input)
        reconstruct_output3 = torch.cat(w3, dim=1)[:, ii, :]  # (612,12,51)
        # w1 = self.decoder1(z)  # (7919,612)
        # w2 = self.decoder2(z)  # (7919,612)
        # w3 = self.decoder2(self.encoder(w1))

        re_1 = reconstruct_output1.view(batch_size,reconstruct_output1.shape[1] * reconstruct_output1.shape[2])
        re_2 = reconstruct_output1.view(batch_size,reconstruct_output2.shape[1] * reconstruct_output2.shape[2])
        re_3 = reconstruct_output1.view(batch_size,reconstruct_output3.shape[1] * reconstruct_output3.shape[2])
        loss1 = 1 / n * torch.mean((batch1 - re_1) ** 2) + (1 - 1 / n) * torch.mean((batch1 - re_3) ** 2)
        loss2 = 1 / n * torch.mean((batch1 - re_2) ** 2) - (1 - 1 / n) * torch.mean((batch1 - re_3) ** 2)
        del re_1
        del re_2
        del re_3
        return loss1, loss2

    def validation_step(self, batch, n):
#        loss_func = nn.SmoothL1Loss(reduction='mean')
        batch_size, sequence_length, var_length = batch.size()  # var_length = 51 batch_size = 612 sequence = 12
        batch1 = batch.view(batch_size, sequence_length * var_length)
        z = self.encoder(batch,batch_size)  # ((2,612,26)\(2,612,26))
        inv_idx = torch.arange(sequence_length - 1, -1, -1).long()  # (12)
        w1 = []
        w2 = []
        temp_input = torch.zeros((batch_size, 1, var_length), dtype=torch.float).to(batch.device)  # (612,1,51)
        hidden = z
        for t in range(sequence_length):
            temp_input, hidden = self.decoder1(temp_input, hidden)
            temp_input, hidden = self.decoder2(temp_input, hidden)
            w1.append(temp_input)
            w2.append(temp_input)
        reconstruct_output1 = torch.cat(w1, dim=1)[:, inv_idx, :]  # (612,12,51)
        reconstruct_output2 = torch.cat(w2, dim=1)[:, inv_idx, :]  # (612,12,51)

        z2 = self.encoder(reconstruct_output1,batch_size)  # ((2,612,26)\(2,612,26))
        bs, sq, vl = reconstruct_output1.size()  # (bs = 612 \sq = 12\vl = 51)
        ii = torch.arange(sq - 1, -1, -1).long()  # (12)
        w3 = []
        tt = torch.zeros((bs, 1, vl), dtype=torch.float).to(batch.device)  # (612,1,51)
        hidden = z2
        for t in range(sq):
            tt, hidden = self.decoder2(tt, hidden)
            w3.append(temp_input)
        reconstruct_output3 = torch.cat(w3, dim=1)[:, ii, :]  # (612,12,51)

        re_1 = reconstruct_output1.view(batch_size,reconstruct_output1.shape[1] * reconstruct_output1.shape[2])
        re_2 = reconstruct_output1.view(batch_size,reconstruct_output2.shape[1] * reconstruct_output2.shape[2])
        re_3 = reconstruct_output1.view(batch_size,reconstruct_output3.shape[1] * reconstruct_output3.shape[2])
        loss1 = 1 / n * torch.mean((batch1 - re_1) ** 2) + (1 - 1 / n) * torch.mean((batch1 - re_3) ** 2)
        loss2 = 1 / n * torch.mean((batch1 - re_2) ** 2) - (1 - 1 / n) * torch.mean((batch1 - re_3) ** 2)
        del re_1
        del re_2
        del re_3
        return {'val_loss1': loss1.item(), 'val_loss2': loss2.item()}

    def validation_epoch_end(self, outputs):
        # batch_losses1 = [x['val_loss1'] for x in outputs]
        # # print(batch_losses1)
        # epoch_loss1 = torch.stack(batch_losses1).mean()
        # # print(epoch_loss1)
        # batch_losses2 = [x['val_loss2'] for x in outputs]
        # epoch_loss2 = torch.stack(batch_losses2).mean()
        b_loss1 = []  # new
        b_loss2 = []  # new

        batch_losses1 = [x['val_loss1'] for x in outputs]

        for loss1 in list(batch_losses1):  # new
            b_loss1.append(torch.tensor(loss1, requires_grad=True))
        epoch_loss1 = torch.stack(b_loss1).mean()

        batch_losses2 = [x['val_loss2'] for x in outputs]
        for loss2 in list(batch_losses2):  # new
            b_loss2.append(torch.tensor(loss2, requires_grad=True))
        epoch_loss2 = torch.stack(b_loss2).mean()
        return {'val_loss1': epoch_loss1.item(), 'val_loss2': epoch_loss2.item()}

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}".format(epoch, result['val_loss1'], result['val_loss2']))


def evaluate(model, val_loader,n):
    outputs = [model.validation_step(to_device(batch, device),n) for [batch] in val_loader]
    # print(type(outputs))
    # print(outputs)
    return model.validation_epoch_end(outputs)


def training(epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam):  # (3，moudel)
    history = []
    optimizer1 = opt_func(list(model.encoder.parameters()) + list(model.decoder1.parameters()))
    optimizer2 = opt_func(list(model.encoder.parameters()) + list(model.decoder2.parameters()))
    for epoch in range(epochs):
        for [batch] in train_loader:  # （612，12 51）
            batch = to_device(batch, device)

            # Train AE1
            loss1, loss2 = model.training_step(batch, epoch + 1)
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()
            loss1.item()
            loss2.item()

            # Train AE2
            loss1, loss2 = model.training_step(batch, epoch + 1)
            loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()
            loss1.item()
            loss2.item()

        result = evaluate(model, val_loader, epoch+1)
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def testing(model, test_loader, alpha=0.5, beta=0.5):
    results = []
    w1 = []
    w2 = []
    for [batch] in test_loader:
        batch = to_device(batch, device)
        batch_size, sequence_length, var_length = batch.size()
        batch1 = batch.view(batch_size, sequence_length * var_length)
        encoder_hidden = model.encoder(batch,batch_size)
        inv_idx = torch.arange(sequence_length - 1, -1, -1).long()
        temp_input = torch.zeros((batch_size, 1, var_length), dtype=torch.float).to(batch.device)
        hidden = encoder_hidden
        for t in range(sequence_length):
            temp_input, hidden = model.decoder1(temp_input, hidden)
            w1.append(temp_input)
        reconstruct_output1 = torch.cat(w1, dim=1)[:, inv_idx, :]  # (612,12,51)

        z2 = model.encoder(reconstruct_output1,batch_size)  # ((2,612,26)\(2,612,26))
        bs, sq, vl = reconstruct_output1.size()  # (bs = 612 \sq = 12\vl = 51)
        ii = torch.arange(sq - 1, -1, -1).long()  # (12)
        tt = torch.zeros((bs, 1, vl), dtype=torch.float).to(batch.device)  # (612,1,51)
        hidden = z2
        for t in range(sq):
            tt, hidden = model.decoder2(tt, hidden)
            w2.append(temp_input)
        reconstruct_output2 = torch.cat(w2, dim=1)[:, ii, :]  # (612,12,51)
        # w1 = model.decoder1(model.encoder(batch))
        # w2 = model.decoder2(model.encoder(w1))  (612,51)
        re_1 = reconstruct_output1.view(batch_size,reconstruct_output1.shape[1] * reconstruct_output1.shape[2])
        re_2 = reconstruct_output2.view(batch_size,reconstruct_output2.shape[1] * reconstruct_output2.shape[2])
        results.append(alpha * torch.mean((batch1 - re_1) ** 2, axis=1) + beta * torch.mean((batch1 - re_2) ** 2, axis=1))
        del re_1
        del re_2
    return results