import torch
from torch import nn
class Word_Adapter(nn.Module):
    def __init__(self, input_dim):
        super(Word_Adapter, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.Norm = nn.BatchNorm1d(1)
        self.Sig = nn.Sigmoid()
    def forward(self, v_c, v_w): # v_c, v_w: B x len x dim
        v_c_flatten = torch.flatten(v_c, end_dim = -2) # Bxlen  x dim
        tmp = self.linear(v_c_flatten)
        # tmp = torch.reshape(tmp, (-1, v_w.shape[1], tmp.shape[1])) # B x len x dim
        v_w_flatten = torch.flatten(v_w, end_dim=-2) #Bxlen   x  dim
        # print(tmp.unsqueeze(-1).transpose(-2,-1).shape, v_w_flatten.unsqueeze(-1).shape)
        tmp = torch.bmm(tmp.unsqueeze(-1).transpose(-2,-1), v_w_flatten.unsqueeze(-1))
        # print(tmp.squeeze(-1).shape)
        tmp = self.Norm(tmp.squeeze(-1))
        tmp = self.Sig(tmp) # B x len   x 1
        # print(tmp.reshape((v_c.shape[0],-1))[0])
        feat = torch.mul(v_c_flatten, torch.ones_like(tmp) - tmp) + torch.mul(v_w_flatten, tmp)
        # print(feat.shape)
        return feat.reshape((v_c.shape[0], v_c.shape[1],-1))

