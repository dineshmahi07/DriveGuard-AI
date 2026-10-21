"""
DriveGuard AI — MobileNetV2 Architecture (DriveGuardMobileNetV2)
Lightweight inverted residual architecture designed for real-time mobile/edge deployment.
Supports both transfer learning with ImageNet pre-training and scratch training.
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class DriveGuardMobileNetV2(nn.Module):
    """
    MobileNetV2 adaptation for Binary Driver Drowsiness Detection.
    Features inverted residual blocks and depthwise separable convolutions.
    """
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout_rate: float = 0.3):
        super().__init__()
        self.pretrained = pretrained
        
        if pretrained:
            try:
                weights = MobileNet_V2_Weights.DEFAULT
                self.backbone = mobilenet_v2(weights=weights)
                self.weights_source = "torchvision MobileNet_V2_Weights.IMAGENET1K_V1"
            except Exception as e:
                print(f"Warning: could not download pretrained weights ({e}), initializing randomly.")
                self.backbone = mobilenet_v2(weights=None)
                self.weights_source = "randomly_initialized (fallback)"
        else:
            self.backbone = mobilenet_v2(weights=None)
            self.weights_source = "randomly_initialized_from_scratch"

        # Replace classification head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = DriveGuardMobileNetV2(pretrained=False)
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    print("DriveGuardMobileNetV2 Output shape:", out.shape)
    print("Total trainable parameters:", model.count_parameters())
