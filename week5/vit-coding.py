import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self,img_size=224,patch_size=16,dim=768):
        super()._init_()
        self.patch_size = patch_size
        self.n_patchs = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    def forward(self,x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self,dim,heads,mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim,heads,batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim,mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim,dim)
        )

    def forward(self,x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    def __init__(self,img_size=224,patch_size=16,dim=768,depth=6,heads=8,mlp_dim=1024,num_classes=10):
        super().__init__()

        self.path_embed = PatchEmbedding(img_size,patch_size,dim)
        n_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1,1,dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1,dim))
        self.layers = nn.ModuleList([TransformerEncoder(dim,heads,mlp_dim)for _ in range(depth)])
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,num_classes))

    def forward(self,x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B,-1,-1)
        x = torch.cat((cls_tokens,x),dim=1)
        x = x + self.pos_embed
        for layer in self.Layers:
            x = layer(x)
        cls_output = x[:,0]
        return self.mlp_head(cls_output)