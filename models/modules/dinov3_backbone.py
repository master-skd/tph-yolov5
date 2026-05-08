"""
DINOv3 ConvNeXt backbone wrapper for YOLOv5-TPH.

Supports loading DINOv3 pretrained ConvNeXt weights and outputs multi-scale
features at strides [4, 8, 16, 32] for integration with YOLO FPN/PAN heads.

Usage:
    # After model creation, load DINOv3 pretrained weights:
    backbone_layer = model.model[0]  # DINOv3ConvNeXt is the first layer
    backbone_layer.load_dinov3_weights('/path/to/dinov3_convnext_large.pth')

    # Or set environment variable before model creation:
    # export DINOV3_WEIGHTS=/path/to/dinov3_convnext_large.pth
"""

import os
import torch
import torch.nn as nn
from pathlib import Path


class DINOv3ConvNeXt(nn.Module):
    """
    DINOv3 ConvNeXt backbone wrapper.
    Outputs a tuple of 4 multi-scale feature maps at strides [4, 8, 16, 32].
    """

    CHANNELS = {
        'tiny': [96, 192, 384, 768],
        'small': [96, 192, 384, 768],
        'base': [128, 256, 512, 1024],
        'large': [192, 384, 768, 1536],
    }

    def __init__(self, model_name='large', c1=128, c2=256, c3=512, c4=1024):
        """
        Args:
            model_name: ConvNeXt variant ('tiny', 'small', 'base', 'large')
            c1, c2, c3, c4: Output channels for each stage (stride 4/8/16/32)
        """
        super().__init__()
        self.model_name = model_name
        self.out_channels = [c1, c2, c3, c4]

        # Build ConvNeXt backbone with multi-scale feature extraction
        self.backbone = self._build_backbone(model_name)

        # Channel adaptation: ConvNeXt native channels -> target channels
        native_channels = self.CHANNELS[model_name]
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(nc, oc, 1, bias=False),
                nn.BatchNorm2d(oc),
            ) if nc != oc else nn.Identity()
            for nc, oc in zip(native_channels, self.out_channels)
        ])

        # Auto-load weights from environment variable if set
        weights_path = os.environ.get('DINOV3_WEIGHTS', None)
        if weights_path:
            self.load_dinov3_weights(weights_path)

    def _build_backbone(self, model_name):
        """Build ConvNeXt backbone with features_only mode."""
        # Strategy 1: Try timm (best support for features_only)
        try:
            import timm
            timm_name = f'convnext_{model_name}'
            backbone = timm.create_model(timm_name, pretrained=False, features_only=True)
            return backbone
        except (ImportError, Exception):
            pass

        # Strategy 2: Fall back to torchvision
        try:
            import torchvision.models as tv_models
            model_map = {
                'tiny': tv_models.convnext_tiny,
                'small': tv_models.convnext_small,
                'base': tv_models.convnext_base,
                'large': tv_models.convnext_large,
            }
            if model_name not in model_map:
                raise ValueError(f"Unsupported model_name: {model_name}")
            model = model_map[model_name](weights=None)
            return _TorchvisionConvNeXtFeatures(model)
        except (ImportError, Exception) as e:
            raise RuntimeError(
                f"Cannot build ConvNeXt-{model_name}. "
                f"Install timm (pip install timm) or torchvision >= 0.13. Error: {e}"
            )

    def load_dinov3_weights(self, weights_path):
        """
        Load DINOv3 pretrained weights into the ConvNeXt backbone.

        Args:
            weights_path: Path to .pth file with DINOv3 ConvNeXt weights
        """
        if not os.path.exists(weights_path):
            print(f"[DINOv3] WARNING: Weights not found at {weights_path}, using random init")
            return

        print(f"[DINOv3] Loading pretrained weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # Remove prefix if present (e.g., 'backbone.' or 'module.')
        cleaned = {}
        for k, v in state_dict.items():
            new_key = k
            for prefix in ['backbone.', 'module.', 'encoder.']:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            cleaned[new_key] = v

        missing, unexpected = self.backbone.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"[DINOv3] {len(missing)} keys missing (may be normal for features_only mode)")
        if unexpected:
            print(f"[DINOv3] {len(unexpected)} unexpected keys ignored")
        print(f"[DINOv3] Weights loaded successfully")

    def freeze_backbone(self):
        """Freeze backbone parameters (recommended for initial training)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[DINOv3] Backbone frozen")

    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("[DINOv3] Backbone unfrozen")

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Tuple of 4 feature maps at strides [4, 8, 16, 32]
        """
        features = self.backbone(x)
        # Adapt channels to target dimensions
        out = tuple(adapter(f) for adapter, f in zip(self.adapters, features))
        return out


class _TorchvisionConvNeXtFeatures(nn.Module):
    """Extract multi-scale features from torchvision ConvNeXt model."""

    def __init__(self, model):
        super().__init__()
        # torchvision ConvNeXt.features has structure:
        # [0]: stem (Conv2d stride4 + LayerNorm)
        # [1]: stage1 blocks
        # [2]: downsample (LayerNorm + Conv2d stride2)
        # [3]: stage2 blocks
        # [4]: downsample
        # [5]: stage3 blocks
        # [6]: downsample
        # [7]: stage4 blocks
        feats = model.features
        self.stage1 = nn.Sequential(feats[0], feats[1])     # stride 4
        self.stage2 = nn.Sequential(feats[2], feats[3])     # stride 8
        self.stage3 = nn.Sequential(feats[4], feats[5])     # stride 16
        self.stage4 = nn.Sequential(feats[6], feats[7])     # stride 32

    def forward(self, x):
        f1 = self.stage1(x)     # stride 4
        f2 = self.stage2(f1)    # stride 8
        f3 = self.stage3(f2)    # stride 16
        f4 = self.stage4(f3)    # stride 32
        return [f1, f2, f3, f4]


class Index(nn.Module):
    """
    Select a specific feature map from a tuple/list by index.
    Used to extract individual scale features from DINOv3ConvNeXt output.
    """

    def __init__(self, index, channels):
        """
        Args:
            index: Which feature to select (0-3)
            channels: Output channel count (for parse_model tracking)
        """
        super().__init__()
        self.index = index
        self.channels = channels

    def forward(self, x):
        return x[self.index]
