from .efficientnet_b0 import EfficientNetB0Classifier, build_efficientnet_b0
from .mobilenetv3_large import MobileNetV3LargeClassifier, build_large_model
from .mobilenetv3_small import MobileNetV3SmallClassifier, build_model as build_small_model

BACKBONES = {
    "mobilenet_v3_small": build_small_model,
    "mobilenet_v3_large": build_large_model,
    "efficientnet_b0": build_efficientnet_b0,
}


def build_model(
    num_classes: int = 2,
    pretrained: bool = True,
    backbone: str = "mobilenet_v3_small",
):
    try:
        builder = BACKBONES[backbone]
    except KeyError:
        raise ValueError(
            f"unknown backbone: {backbone}, expected one of {sorted(BACKBONES)}"
        ) from None
    return builder(num_classes=num_classes, pretrained=pretrained)


__all__ = [
    "MobileNetV3SmallClassifier",
    "MobileNetV3LargeClassifier",
    "EfficientNetB0Classifier",
    "build_model",
]
