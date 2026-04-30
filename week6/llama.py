import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self,x):
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x / torch.sqrt(norm + self.eps)
        return self.scale * x


def apply_rope(q, k):
    # q, k: (B, T, D)
    dim = q.shape[-1]
    half = dim // 2

    freqs = torch.arange(half, dtype=torch.float32, device=q.device) / half
    freqs = 1.0 / (10000 ** freqs)

    t = torch.arange(q.shape[1], device=q.device)
    freqs = torch.outer(t, freqs)

    cos = freqs.cos()[None, :, :]
    sin = freqs.sin()[None, :, :]

    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]

    q_rot = torch.cat([q1 * cos - q2 * sin,
                       q1 * sin + q2 * cos], dim=-1)

    k_rot = torch.cat([k1 * cos - k2 * sin,
                       k1 * sin + k2 * cos], dim=-1)

    return q_rot, k_rot


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, D = x.shape

        qkv = self.qkv(x)  # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape to multi-head
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # merge heads for RoPE
        q = q.reshape(B * self.num_heads, T, self.head_dim)
        k = k.reshape(B * self.num_heads, T, self.head_dim)

        q, k = apply_rope(q, k)

        # reshape back
        q = q.view(B, self.num_heads, T, self.head_dim)
        k = k.view(B, self.num_heads, T, self.head_dim)

        # attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, D)

        return self.out(out)


class SwiGLU(nn.Module):
    def __init__(self,dim:int,hidden_dim:int):
        super().__init__()
        self.w1 = nn.Linear(dim,hidden_dim,bias=False)
        self.w2 = nn.Linear(dim,hidden_dim,bias=False)
        self.w3 = nn.Linear(hidden_dim,dim,bias=False)

    def forward(self,x):
        return self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))

class LLaMABlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        self.attn = Attention(dim, num_heads)
        self.ffn = SwiGLU(dim, dim * 4)

    def forward(self, x):
        # ===== Pre-Norm Attention =====
        h = self.norm1(x)
        x = x + self.attn(h)

        # ===== Pre-Norm FFN =====
        h = self.norm2(x)
        x = x + self.ffn(h)

        return x

