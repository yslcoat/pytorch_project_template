import logging

import torch
import torch.nn as nn

from timm.utils.distributed import is_primary

_logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    """Small example model, replace or register your own in MODEL_REGISTRY."""

    def __init__(self, num_classes=10, in_chans=3, width=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_chans, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(width * 2, num_classes)

    def forward(self, x):
        return self.head(self.features(x))


MODEL_REGISTRY = {
    "simple_cnn": SimpleCNN,
}


def _build_model(configs):
    kwargs = dict(
        num_classes=configs.num_classes,
        in_chans=configs.in_chans,
        **configs.model_kwargs,
    )
    if configs.model in MODEL_REGISTRY:
        return MODEL_REGISTRY[configs.model](**kwargs)
    # Fall back to timm for any model name not registered locally, e.g. model: resnet50
    import timm

    return timm.create_model(configs.model, **kwargs)


def create_model(configs, device, model_dtype=None):
    model = _build_model(configs)

    if configs.grad_checkpointing:
        if hasattr(model, "set_grad_checkpointing"):
            model.set_grad_checkpointing(enable=True)
            _logger.info("Gradient checkpointing enabled.")
        else:
            _logger.warning(
                "Gradient checkpointing configured, but the model does not have a "
                "'set_grad_checkpointing' method. Ignoring."
            )

    model.to(device=device, dtype=model_dtype)
    if configs.channels_last:
        model.to(memory_format=torch.channels_last)

    if configs.distributed and configs.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if is_primary(configs):
            _logger.info("Converted model to use Synchronized BatchNorm.")

    if is_primary(configs):
        _logger.info(
            f"Model {configs.model} created, param count: {sum(p.numel() for p in model.parameters())}"
        )
    return model
