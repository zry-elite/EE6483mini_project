import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class MobileNetV3SmallClassifier(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = mobilenet_v3_small(weights=weights)
        self.net.classifier[-1] = nn.Linear(self.net.classifier[-1].in_features, num_classes)

    def forward(self, x):
        return self.net(x)


def build_model(num_classes: int = 2, pretrained: bool = True) -> MobileNetV3SmallClassifier:
    return MobileNetV3SmallClassifier(num_classes=num_classes, pretrained=pretrained)
