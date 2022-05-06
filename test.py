import numpy as np
import torch
import torch.nn.functional as F


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



if __name__ == '__main__':
    x = [[0.9826, 0.7930, 1.0000, 0.1],
         [0.9751, 0.7929, 1.0000, 0.1],
         [0.9466, 0.7941, 1.0000, 0.1]]

    y = [[0.5050, 0.4951, 0.5075, 0.456],
         [0.5050, 0.4951, 0.5075, 0.675],
         [0.5050, 0.4951, 0.5075, 0.234]]
    x1 = torch.log(torch.from_numpy(np.array(x, np.float)))
    y1 = torch.log(torch.from_numpy(np.array(y, np.float)))
    print(EuclideanDistance(x1, y1))
    print(F.pairwise_distance(x1, y1, p=4))
    print(torch.mean(F.pairwise_distance(x1, y1, p=4)))
    r1 = torch.mean(EuclideanDistance(x1, y1), axis=0)
    r2 = torch.mean(F.pairwise_distance(x1, y1, p=2), axis=0)
    print(r1)
    print(r2)

    loss_func = torch.nn.SmoothL1Loss(reduce=False, size_average=False)
    input = torch.autograd.Variable(torch.randn(3, 4))
    target = torch.autograd.Variable(torch.randn(3, 4))
    loss = loss_func(input, target)
    print(input)
    print(target)
    print(loss)
    print(input.size(), target.size(), loss.size())