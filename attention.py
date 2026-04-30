import torch
import torch.nn as nn
import math



class SelfAttention(nn.Module):
    def __init__(self,dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,Q,K,V,mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        out = torch.matmul(attn,V)
        return out,attn

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_heads,dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.W_Q = nn.Linear(d_model,d_model)
        self.W_K = nn.Linear(d_model,d_model)
        self.W_V = nn.Linear(d_model,d_model)
        self.fc = nn.Linear(d_model,d_model)

        self.attention = SelfAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self,Q,K,V,mask=None):
        batch_size = Q.size(0)
        Q = self.W_Q(Q).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
        K = self.W_K(K).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
        V = self.W_V(V).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)

        out,attn =self.attention(Q,K,V,mask)
        out = out.transpose(1,2).contiguous().view(batch_size,-1,self.n_heads * self.d_k)
        out = self.fc(out)
        out = self.dropout(out)
        out = self.norm(out+Q),attn
        return out,attn

