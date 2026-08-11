"""Vision encoder using MobileNetV5 via timm."""

import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from safetensors import safe_open
import einops
from collections.abc import Sequence
from typing import Any

class VisionExit(nn.Module):
    """The vision exit layer. Downsamples tokens to a required output length."""
    def __init__(self, output_length: int = 256):
        super().__init__()
        self.output_length = output_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, input_length, d)
        batch_size, cur_length, d = x.shape
        if cur_length == self.output_length:
            return x

        cur_width = int(cur_length**0.5)
        if cur_width**2 != cur_length:
             # Handle non-square if needed, but usually it's square
             return x
             
        output_width = int(self.output_length**0.5)
        if output_width**2 != self.output_length:
            return x

        x = x.view(batch_size, cur_width, cur_width, d)
        x = x.permute(0, 3, 1, 2)  # (b, d, h, w) for pooling

        window = cur_width // output_width
        if window > 1:
            x = F.avg_pool2d(x, kernel_size=window, stride=window)

        x = x.permute(0, 2, 3, 1)  # (b, h', w', d)
        return x.reshape(batch_size, self.output_length, d)

class MobileNetV5Encoder(nn.Module):
    def __init__(self, model_name='mobilenetv5_300m_enc', pretrained=False):
        super().__init__()
        # Direct import to avoid timm.create_model registration issues
        import timm.models as models
        if hasattr(models, model_name):
            model_fn = getattr(models, model_name)
            self.model = model_fn(pretrained=pretrained)
        else:
            # Fallback if the user passes a name that isn't the direct function name
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')

    @classmethod
    def from_hf_pretrained(cls,
                           model_path: str,
                           model_name: str = 'mobilenetv5_300m_enc',
                           device: str = "cpu",
                           prefix: str = "model.vision_tower.timm_model",
                           low_cpu_mem_usage: bool = True,
                           **kwargs
                       ) -> "MobileNetV5Encoder":
        """Loads weights from a sharded safetensors directory into a timm model with memory efficiency."""
        import os
        from safetensors import safe_open
        
        # 1. Initialize model (optionally on 'meta' device)
        if low_cpu_mem_usage:
            with torch.device("meta"):
                model = cls(model_name=model_name)
            model.to_empty(device=device)
        else:
            model = cls(model_name=model_name)
            model.to(device)

        if not os.path.exists(model_path):
             # Fallback if model_path is just a filename index or something
             if os.path.exists(os.path.dirname(model_path)):
                 model_path = os.path.dirname(model_path)
             else:
                 model_path = "."

        files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
        
        with torch.no_grad():
            params_dict = dict(model.model.named_parameters())
            for f_name in files:
                file_path = os.path.join(model_path, f_name)
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key.startswith(prefix):
                            new_key = key[len(prefix)+1:] # remove prefix and dot
                            if new_key in params_dict:
                                params_dict[new_key].copy_(f.get_tensor(key))
        
        print(f"Successfully loaded MobileNetV5 weights from {model_path} with prefix {prefix}")
        return model

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.model(x) # (B, D, H, W)
        x = einops.rearrange(x, 'b d h w -> b (h w) d')
        return x

class MobileNetV5FromPatches(nn.Module):
    """MobileNetV5 vision encoder forward pass from patchified media."""

    def __init__(
        self,
        model_name: str = 'mobilenetv5_300m_enc',
        model_path: str | None = None,
        output_length: int = 256,
        apply_stop_gradient: bool = True,
    ):
        super().__init__()
        if model_path:
            self.encoder = MobileNetV5Encoder.from_hf_pretrained(model_path, model_name=model_name)
        else:
            self.encoder = MobileNetV5Encoder(model_name=model_name)
            
        self.exit = VisionExit(output_length=output_length)
        self.apply_stop_gradient = apply_stop_gradient

    def forward(
        self,
        patches: torch.Tensor,  # [b, n, num_patches, patch_dim]
    ) -> torch.Tensor:
        batch_size, num_frames, num_patches, patch_dim = patches.shape

        # Infer patch size from patch_dim (assuming 3 channels)
        p_h = p_w = int((patch_dim // 3)**0.5)
        num_patches_one_side = int(num_patches**0.5)

        # Reconstruct image from patches
        channels = 3
        x = einops.rearrange(
            patches,
            'b n (h_p w_p) (c p_h p_w) -> (b n) c (h_p p_h) (w_p p_w)',
            h_p=num_patches_one_side,
            w_p=num_patches_one_side,
            p_h=p_h,
            p_w=p_w,
            c=channels
        )

        soft_tokens = self.encoder(x)

        # Check if pooling is needed
        if soft_tokens.shape[1] != self.exit.output_length:
            soft_tokens = self.exit(soft_tokens)

        soft_tokens = einops.rearrange(soft_tokens,
                                       '(b n) s d -> b n s d',
                                       b=batch_size
                                   )

        if self.apply_stop_gradient:
            soft_tokens = soft_tokens.detach()

        return soft_tokens

# Alias for compatibility with the rest of the codebase
SigLiPFromPatches = MobileNetV5FromPatches
