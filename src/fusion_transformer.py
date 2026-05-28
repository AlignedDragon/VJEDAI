import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model=256, num_heads=8, mlp_ratio=4):
        super().__init__()

        self.depth_self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        hidden_dim = d_model * mlp_ratio

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, depth_tokens, scene_tokens):
        x_norm = self.norm1(depth_tokens)
        self_attn_out, _ = self.depth_self_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm
        )

        x = depth_tokens + self_attn_out

        x_norm = self.norm2(x)
        cross_attn_out, _ = self.cross_attn(
            query=x_norm,
            key=scene_tokens,
            value=scene_tokens
        )
        x = x + cross_attn_out

        x_norm = self.norm3(x)
        x = x + self.mlp(x_norm)

        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.skip_proj = nn.Conv2d(1, skip_channels, kernel_size=3, padding=1)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, da_depth, size):
        x = F.interpolate(
            x,
            size=size,
            mode='bilinear',
            align_corners=False
        )

        da_skip = F.interpolate(
            da_depth,
            size=size,
            mode='bilinear',
            align_corners=False
        )

        da_skip = self.skip_proj(da_skip)

        x = torch.cat([x, da_skip], dim=1)
        x = self.conv(x)

        return x

class DepthVJepaFusionTransformer(nn.Module):
    def __init__(self,
                 vjepa_dim=1024,
                 d_model=256,
                 num_heads=8,
                 num_layers=4,
                 token_grid_size=24,
                 max_log_correction=2.0,
                 eps=1e-6):
        super().__init__()

        self.token_grid_size = token_grid_size
        self.max_log_correction = max_log_correction
        self.eps = eps

        self.depth_proj = nn.Linear(1, d_model)
        self.vjepa_proj = nn.Linear(vjepa_dim, d_model)

        self.fusion_blocks = nn.ModuleList([
            CrossAttentionBlock(
                d_model=d_model,
                num_heads=num_heads,
                mlp_ratio=4
            ) for _ in range(num_layers)
        ])

        self.token_conv = nn.Sequential(
            nn.Conv2d(d_model, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up1 = UpBlock(256, 32, 192)
        self.up2 = UpBlock(192, 32, 128)
        self.up3 = UpBlock(128, 32, 64)
        self.up4 = UpBlock(64, 32, 32)

        self.depth_head = nn.Conv2d(32, 1, kernel_size=1)

        nn.init.zeros_(self.depth_head.weight)
        nn.init.zeros_(self.depth_head.bias)

    def forward(self, da_depth, vjepa_tokens, output_size):
        '''
        :param da_depth: [B, H, W]
        :param vjepa_tokens:  [B, 256, 1024]
        :param output_size: tupe (H, W)
        :return:
        '''

        da_depth = da_depth.unsqueeze(1) # [B, 1, H, W]

        B = da_depth.shape[0]
        G = self.token_grid_size
        H, W = output_size

        da_small = F.interpolate(da_depth,
                                 size=(G, G),
                                 mode="bilinear",
                                 align_corners=False) # [B, 1, 24, 24]

        depth_tokens = da_small.flatten(2).transpose(1, 2) # [B, 576, 1]
        depth_tokens = self.depth_proj(depth_tokens) # [B, 576, d_model]

        scene_tokens = self.vjepa_proj(vjepa_tokens) # [B, 576, d_model]

        x = depth_tokens

        for block in self.fusion_blocks:
            x = block(x, scene_tokens)

        x = x.transpose(1, 2).reshape(B, -1, G, G)
        x = self.token_conv(x)

        x = self.up1(x, da_depth, size=(48, 48))
        x = self.up2(x, da_depth, size=(96, 96))
        x = self.up3(x, da_depth, size=(192, 192))
        x = self.up4(x, da_depth, size=(H, W))

        residual = self.depth_head(x)

        da_depth_full = F.interpolate(da_depth,
                                      size=output_size,
                                      mode="bilinear",
                                      align_corners=False)

        refined_depth = da_depth_full + residual
        refined_depth = torch.clamp(refined_depth, min=self.eps)

        return refined_depth