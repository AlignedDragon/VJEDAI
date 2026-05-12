import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthVJepaFusionTransformer(nn.Module):
    def __init__(self,
                 vjepa_dim=1024,
                 d_model=256,
                 num_heads=8,
                 num_layers=2,
                 token_grid_size=24):
        super().__init__()

        self.token_grid_size = token_grid_size

        self.depth_proj = nn.Linear(1, d_model)

        self.vjepa_proj = nn.Linear(vjepa_dim, d_model)

        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])

        self.decoder = nn.Sequential(
            nn.Conv2d(d_model, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.depth_head = nn.Conv2d(32, 1, kernel_size=1)

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

        da_small = F.interpolate(da_depth,
                                 size=(G, G),
                                 mode="bilinear",
                                 align_corners=False) # [B, 1, 24, 24]

        depth_tokens = da_small.flatten(2).transpose(1, 2) # [B, 576, 1]
        depth_tokens = self.depth_proj(depth_tokens) # [B, 576, d_model]

        scene_tokens = self.vjepa_proj(vjepa_tokens) # [B, 576, d_model]

        x = depth_tokens

        for attn, norm in zip(self.cross_attn_layers, self.norm_layers):
            attn_out, _ = attn(query=x,
                               key=scene_tokens,
                               value=scene_tokens)
            x = norm(x + attn_out)

        x = x.transpose(1, 2).reshape(B, -1, G, G) # [B, d_model, 24, 24]

        features = self.decoder(x)

        features = F.interpolate(features,
                                 size=output_size,
                                 mode="bilinear",
                                 align_corners=False)

        residual = self.depth_head(features)

        da_depth_full = F.interpolate(da_depth,
                                      size=output_size,
                                      mode="bilinear",
                                      align_corners=False)

        refined_depth = da_depth_full + residual

        return refined_depth