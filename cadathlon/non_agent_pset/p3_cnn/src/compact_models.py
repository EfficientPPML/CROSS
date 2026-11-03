"""The two explicitly reduced CNNs in the compact-v1 competition schedule.

These are distinct from the original QuadLeNet/QuadAlexNetTiny problems.
Each retains encrypted convolution, a quadratic activation, and a classifier
while requiring only three multiplication levels instead of seven.
"""
import torch.nn as nn


class _CompactCNN(nn.Module):
    _cadathlon_compact_profile = True

    def __init__(self, input_channels, output_channels, size, kernel):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel,
                              stride=4, padding=kernel // 2)
        width = (size + 3) // 4
        self.fc = nn.Linear(output_channels * width * width, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x * x
        return self.fc(x.view(x.size(0), -1))


class CompactGrayCNN(_CompactCNN):
    """1x28x28 -> Conv(1,2,5,stride4,pad2) -> square -> Linear(98,10)."""
    def __init__(self):
        super().__init__(1, 2, 28, 5)


class CompactRGBCNN(_CompactCNN):
    """3x16x16 -> Conv(3,4,3,stride4,pad1) -> square -> Linear(64,10)."""
    def __init__(self):
        super().__init__(3, 4, 16, 3)
