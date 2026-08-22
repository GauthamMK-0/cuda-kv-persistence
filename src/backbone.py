import torch
import torch.nn.functional as F
from torch import nn

from src.config import BackboneConfig


class SelfAttention(nn.Module):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.qkv_proj = nn.Linear(cfg.hidden_dim, 3 * cfg.hidden_dim)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

    def forward(self, x):
        b, t, _ = x.shape
        qkv = self.qkv_proj(x).view(b, t, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        ctx = F.scaled_dot_product_attention(q, k, v)
        ctx = ctx.transpose(1, 2).reshape(b, t, self.num_heads * self.head_dim)
        return self.out_proj(ctx), k, v


class DiTSelfAttentionBlock(nn.Module):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.attn = SelfAttention(cfg)

    def forward(self, x):
        h, k, v = self.attn(self.norm(x))
        return x + h, k, v


class SelfAttentionBackbone(nn.Module):
    def __init__(self, cfg=None, init="random", seed=None):
        super().__init__()
        if init != "random":
            raise NotImplementedError("pretrained weight loading is deferred to phase 8")
        self.cfg = cfg or BackboneConfig()
        if seed is not None:
            torch.manual_seed(seed)
        self.token_embed = nn.Linear(self.cfg.latent_channels, self.cfg.hidden_dim)
        self.blocks = nn.ModuleList(
            DiTSelfAttentionBlock(self.cfg) for _ in range(self.cfg.num_layers)
        )
        self.final_norm = nn.LayerNorm(self.cfg.hidden_dim)
        self.apply(self._reset_linear_weights)

    @staticmethod
    def _reset_linear_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, latents, timestep=None, context=None):
        del timestep, context
        b = latents.shape[0]
        x = latents.reshape(b, self.cfg.tokens_per_frame, self.cfg.latent_channels)
        x = self.token_embed(x)
        kv_per_layer = []
        for block in self.blocks:
            x, k, v = block(x)
            kv_per_layer.append((k, v))
        return self.final_norm(x), kv_per_layer
