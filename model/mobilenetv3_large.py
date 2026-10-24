import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


class MobileNetV3LargeClassifier(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = mobilenet_v3_large(weights=weights)
        self.net.classifier[-1] = nn.Linear(self.net.classifier[-1].in_features, num_classes)

    def forward(self, x):
        return self.net(x)


def build_large_model(
    num_classes: int = 2, pretrained: bool = True
) -> MobileNetV3LargeClassifier:
    return MobileNetV3LargeClassifier(num_classes=num_classes, pretrained=pretrained)
