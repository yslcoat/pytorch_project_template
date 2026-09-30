import argparse
import logging

import yaml

_logger = logging.getLogger(__name__)

# Every key the training pipeline reads. A YAML config only needs to set the keys it wants to change.
DEFAULTS = dict(
    # experiment / output
    experiment="",
    output="output",
    seed=42,
    log_interval=50,
    val_interval=1,
    resume="",
    # device
    device="cuda",
    amp=False,
    amp_dtype="float16",
    model_dtype=None,
    channels_last=False,
    sync_bn=False,
    # model
    model="simple_cnn",
    model_kwargs={},
    num_classes=10,
    in_chans=3,
    grad_checkpointing=False,
    # data
    dataset="synthetic",
    data_dir="",
    train_split="train",
    val_split="val",
    img_size=32,
    batch_size=64,
    validation_batch_size=None,
    workers=4,
    pin_mem=True,
    persistent_workers=True,
    train_num_samples=2048,  # synthetic dataset only
    val_num_samples=512,  # synthetic dataset only
    # learning task
    task="classification",
    smoothing=0.0,
    # optimizer
    opt="adamw",
    lr=1e-3,
    momentum=0.9,
    weight_decay=0.05,
    clip_grad=None,
    grad_accum_steps=1,
    # schedule
    epochs=10,
    sched="cosine",
    warmup_epochs=1,
    warmup_lr=1e-6,
    min_lr=1e-6,
)


def _parse_override(item):
    if "=" not in item:
        raise argparse.ArgumentTypeError(
            f'Override "{item}" must have the form key=value.'
        )
    key, value = item.split("=", 1)
    # YAML parsing turns "0.1" into a float, "true" into a bool, "[1, 2]" into a list, etc.
    return key.strip().replace("-", "_"), yaml.safe_load(value)


def parse_configs(argv=None):
    """Build the run configuration: DEFAULTS <- YAML file <- command line key=value overrides."""
    parser = argparse.ArgumentParser(description="Training Configuration")
    parser.add_argument(
        "-c", "--config", type=str, default="", help="Path to a YAML configuration file"
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        type=_parse_override,
        metavar="KEY=VALUE",
        help="Override config values, e.g. lr=0.01 epochs=5",
    )
    args = parser.parse_args(argv)

    configs = dict(DEFAULTS)
    if args.config:
        with open(args.config, "r") as f:
            file_cfg = yaml.safe_load(f) or {}
        configs.update(file_cfg)
    for key, value in args.overrides:
        if key not in configs:
            _logger.warning(
                f'Override key "{key}" is not a known config option, adding it anyway.'
            )
        configs[key] = value

    configs = argparse.Namespace(**configs)
    configs.grad_accum_steps = max(1, configs.grad_accum_steps)
    if configs.validation_batch_size is None:
        configs.validation_batch_size = configs.batch_size
    return configs


def configs_to_yaml(configs):
    return yaml.safe_dump(vars(configs), default_flow_style=False)
