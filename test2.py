import torch
from torch import nn
from einops import rearrange


class ExternelAttention(nn.Module):
    def __init__(self, in_channel, out_channel, num_memory_units=64, num_heads=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads

        self.convert_dims = nn.Linear(in_channel, out_channel)
        self.extend_dims = nn.Linear(in_channel, out_channel * num_heads)

        self.memory_key = nn.Linear(out_channel, num_memory_units)
        self.memory_value = nn.Linear(num_memory_units, out_channel)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_channel * num_heads, out_channel)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        height = x.size(2)  # height = 51  (2,2,51,1)

        x = rearrange(x, 'b c h w -> b (h w) c')  # (2,51,2)
        print(x.size(),'x by rearrange1')

        y = self.convert_dims(x)  # for residual connection  (2,51,78)
        x = self.extend_dims(x)  # (2,51,624)
        print(y.size(),'y by linear')
        print(x.size(),'x by linear')

        x = rearrange(x, 'b n (h c) -> b h n c', h=self.num_heads)  # (2,8,51,78)

        print(x.size(),'x by rearrange2')

        attn = self.memory_key(x)  # (2,8,51,64)
        print(attn.size(),'x by memory_key')
        attn = attn.softmax(dim=-2)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))  # (2,8,51,64)
        print(attn.size(),'x by div')
        attn = self.attn_drop(attn)  # (2,8,51,64)
        print(attn.size(),'x by attn_drop')
        x = self.memory_value(attn)  # (2,8,51,78)
        print(x.size(),'x by memory_value')
        x = rearrange(x, 'b h n c -> b n (h c)')  # (2,51,624)
        print(x.size(),'x by rearrange3')

        x = self.proj(x)  # (2,51,78)
        print(x.size(),'x by proj')
        x = self.proj_drop(x)  # (2,51,78)
        print(x.size(),'x by proj_drop')

        x = rearrange(x, 'b (h w) c -> b c h w', h=height)  # ( 2,78,51,1)
        y = rearrange(y, 'b (h w) c -> b c h w', h=height)  # ( 2,78,51,1)
        print(x.size(),'x by rearrange4')
        print(y.size(),'y by rearrange')

        return (x + y).contiguous()


if __name__ == '__main__':
    x = torch.rand(2,2,51,1)
    ea = ExternelAttention(2, 78)  # （in_channel=2,out_channel=78）
    eax = ea(x) # (2,78,51,1)
    print(eax.size())
    print(eax.view(2, -1).size())  # (2,3978)



'''
class External_attention(nn.Module):
    '''
'''
    Arguments:
        c (int): The input and output channel number. 官方的代码中设为512
    '''
'''
    def __init__(self, c):
        super(External_attention, self).__init__()
        self.conv1 = nn.Conv2d(c, c, 1)
        self.k = 64
        self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)

        self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)        
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            norm_layer(c))        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
 

    def forward(self, x):
        idn = x
        x = self.conv1(x)

        b, c, h, w = x.size()
        n = h*w
        x = x.view(b, c, h*w)   # b * c * n 

        attn = self.linear_0(x) # b, k, n
        #linear_0是第一个memory unit
        attn = F.softmax(attn, dim=-1) # b, k, n

        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True)) #  # b, k, n
        
        x = self.linear_1(attn) # b, c, n
        #linear_1是第二个memory unit
        x = x.view(b, c, h, w)
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x)
        return x
'''