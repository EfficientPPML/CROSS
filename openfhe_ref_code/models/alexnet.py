import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, ci, co, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ci, co, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(co),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.conv(x)


class LinearBlock(nn.Module):
    def __init__(self, ni, no):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(ni, no),
            nn.BatchNorm1d(no),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.linear(x)


class AlexNetTorch(nn.Module):
    cfg = [64, "M", 192, "M", 384, 256, 256, "A"]

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = self._make_layers()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            LinearBlock(1024, 4096),
            LinearBlock(4096, 4096),
            nn.Linear(4096, num_classes),
        )

    def _make_layers(self):
        layers = []
        in_channels = 3

        for x in self.cfg:
            if x == "M":
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            elif x == "A":
                layers.append(nn.AdaptiveAvgPool2d((2, 2)))
            else:
                layers.append(
                    ConvBlock(
                        in_channels,
                        x,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                in_channels = x

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


@torch.no_grad()
def evaluate(model, dataloader, device="cuda"):
    model.eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
    }


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis

    try:
        from torchsummary import summary
    except ImportError:
        summary = None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = AlexNetTorch(num_classes=10).to(device)
    net.eval()

    x = torch.randn(1, 3, 32, 32).to(device)

    with torch.no_grad():
        y = net(x)

    print("Input shape: ", tuple(x.shape))
    print("Output shape:", tuple(y.shape))

    flops = FlopCountAnalysis(net, x)
    flops.unsupported_ops_warnings(False)
    print("Total FLOPs:", flops.total())

    if summary is not None:
        summary(net, (3, 32, 32), depth=10, device=device)