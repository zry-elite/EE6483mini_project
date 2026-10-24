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


class MobileNetV3LargeCIFARClassifier(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        net = mobilenet_v3_large(weights=weights)

        # 32x32 inputs cannot survive 5 stride-2 stages; drop two of them.
        net.features[0][0].stride = (1, 1)
        for module in net.features[1:]:
            depthwise = next(
                (
                    layer
                    for layer in module.modules()
                    if isinstance(layer, nn.Conv2d)
                    and layer.groups > 1
                    and layer.stride == (2, 2)
                ),
                None,
            )
            if depthwise is not None:
                depthwise.stride = (1, 1)
                if hasattr(module, "stride"):
                    module.stride = 1
                break

        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        self.net = net

    def forward(self, x):
        return self.net(x)


def build_large_cifar_model(
    num_classes: int = 10, pretrained: bool = True
) -> MobileNetV3LargeCIFARClassifier:
    return MobileNetV3LargeCIFARClassifier(
        num_classes=num_classes, pretrained=pretrained
    )
