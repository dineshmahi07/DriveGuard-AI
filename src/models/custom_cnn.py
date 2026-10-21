"""
DriveGuard AI — Custom CNN Architecture (DriveGuardCNN)
Designed and implemented from scratch for binary driver drowsiness detection.
Features 4 convolutional feature extraction blocks with Batch Normalization,
Global Average Pooling, and Dropout for regularization.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block:
    Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d -> Optional Dropout
    """
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        if dropout_rate > 0.0:
            layers.append(nn.Dropout2d(p=dropout_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DriveGuardCNN(nn.Module):
    """
    DriveGuard Custom CNN Model.
    Trained from scratch to demonstrate deep learning architecture design principles.
    """
    def __init__(self, num_classes: int = 2, in_channels: int = 3, dropout_rate: float = 0.4):
        super().__init__()
        
        # 4-Stage Feature Extractor
        self.block1 = ConvBlock(in_channels, 32, dropout_rate=0.0)      # Output: 32 x 64 x 64
        self.block2 = ConvBlock(32, 64, dropout_rate=0.1)               # Output: 64 x 32 x 32
        self.block3 = ConvBlock(64, 128, dropout_rate=0.2)              # Output: 128 x 16 x 16
        self.block4 = ConvBlock(128, 256, dropout_rate=0.3)             # Output: 256 x 8 x 8

        # Global Average Pooling and Dense Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming normal initialization for convolutional and linear layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = DriveGuardCNN()
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    print("DriveGuardCNN Output shape:", out.shape)
    print("Total trainable parameters:", model.count_parameters())
