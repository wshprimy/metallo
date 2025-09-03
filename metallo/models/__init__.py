from .toynet import ToyNet, ToyNetConfig
from .metallonet import MetalloNet, MetalloNetConfig

__all__ = [
    "ToyNet",
    "ToyNetConfig",
]

MODEL_MAPPING = {
    "toynet": {"model": ToyNet, "config": ToyNetConfig},
    "metallonet": {"model": MetalloNet, "config": MetalloNetConfig},
}
