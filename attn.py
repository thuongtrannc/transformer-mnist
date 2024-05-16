import torch
import torch.nn as nn
import torch.nn.functional as F

class MHA(nn.Module):
    def __init__(self, dim, heads, classes):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        self.heads = heads
        self.dim = dim
        self.d_k = dim // heads
        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        self.Wo = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.dropout1 = nn.Dropout(0.1)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout2 = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(dim)
        self.linear3 = nn.Linear(3136, classes)


    def mha(self, q, k, v):
        N = q.size(1)
        q = self.Wq(q).view(-1, N, self.heads, self.d_k).transpose(1, 2)
        k = self.Wk(k).view(-1, N, self.heads, self.d_k).transpose(1, 2)
        v = self.Wv(v).view(-1, N, self.heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        scores = torch.softmax(scores, dim=-1)
        output = torch.matmul(scores, v).transpose(1, 2).contiguous().view(-1, N, self.dim)
        output = self.Wo(output)

        return output, scores


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64, 7 * 7).permute(0, 2, 1)
        x1, scores = self.mha(x, x, x)
        x = x + x1
        x = self.norm1(x)
        x1 = self.linear1(x)
        x1 = F.relu(x1)
        x1 = self.dropout1(x1)
        x1 = self.linear2(x1)
        x = x + self.dropout2(x1)
        x = self.norm2(x)
        x = x.view(-1, 3136)
        x = self.linear3(x)

        return x, scores


