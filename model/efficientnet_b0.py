import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class EfficientNetB0Classifier(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = efficientnet_b0(weights=weights)
        self.net.classifier[-1] = nn.Linear(self.net.classifier[-1].in_features, num_classes)

    def forward(self, x):
        return self.net(x)


def build_efficientnet_b0(
    num_classes: int = 2, pretrained: bool = True
) -> EfficientNetB0Classifier:
    return EfficientNetB0Classifier(num_classes=num_classes, pretrained=pretrained)
