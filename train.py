#!/usr/bin/env python3
"""Training entry point.

python train.py -c configs/example.yaml [key=value ...]
torchrun --nproc_per_node=2 train.py -c configs/example.yaml
"""
import logging
import os
from contextlib import nullcontext
from datetime import datetime
from functools import partial

import torch
from torch.nn.parallel import DistributedDataParallel

from timm.utils import NativeScaler
from timm.utils.distributed import init_distributed_device, is_primary
from timm.utils.log import setup_default_logging
from timm.utils.random import random_seed

from utils.configs_parser import parse_configs, configs_to_yaml
from models.model_factory import create_model
from data.dataset_factory import create_dataset
from data.dataloader_factory import create_dataloader
from learning_tasks.learning_task_factory import create_learning_task
from training_utils.optim_factory import create_optimizer, create_scheduler
from training_utils.trainer import Trainer


_logger = logging.getLogger("train")


def setup_precision(configs, device):
    """Resolve model dtype and AMP settings. Returns (model_dtype, amp_autocast, loss_scaler)."""
    model_dtype = None
    if configs.model_dtype:
        assert configs.model_dtype in ("float32", "float16", "bfloat16")
        model_dtype = getattr(torch, configs.model_dtype)
        if model_dtype == torch.float16:
            _logger.warning(
                "float16 is not recommended for training, for half precision bfloat16 is recommended."
            )

    amp_autocast = nullcontext
    loss_scaler = None
    if configs.amp:
        assert (
            model_dtype is None or model_dtype == torch.float32
        ), "float32 model dtype must be used with AMP"
        assert configs.amp_dtype in ("float16", "bfloat16")
        amp_dtype = torch.bfloat16 if configs.amp_dtype == "bfloat16" else torch.float16
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        if device.type == "cuda" and amp_dtype == torch.float16:
            # loss scaler only used for float16 (half) dtype, bfloat16 does not need it
            loss_scaler = NativeScaler(device=device.type)
        if is_primary(configs):
            _logger.info(
                f"Using native Torch AMP ({configs.amp_dtype}). Training in mixed precision."
            )
    elif is_primary(configs):
        _logger.info(f"AMP not enabled. Training in {model_dtype or torch.float32}.")
    return model_dtype, amp_autocast, loss_scaler


def setup_output_dir(configs):
    if not is_primary(configs) or not configs.output:
        return None
    exp_name = configs.experiment or f"{datetime.now():%Y%m%d-%H%M%S}-{configs.model}"
    output_dir = os.path.join(configs.output, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "configs.yaml"), "w") as f:
        f.write(configs_to_yaml(configs))
    return output_dir


def main():
    setup_default_logging()
    configs = parse_configs()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    elif configs.device == "cuda":
        _logger.warning("CUDA not available, falling back to CPU.")
        configs.device = "cpu"

    device = init_distributed_device(configs)
    if configs.distributed:
        _logger.info(
            "Training in distributed mode with multiple processes, 1 device per process. "
            f"Process {configs.rank}, total {configs.world_size}, device {configs.device}."
        )
    else:
        _logger.info(f"Training with a single process on 1 device ({configs.device}).")
    assert configs.rank >= 0

    model_dtype, amp_autocast, loss_scaler = setup_precision(configs, device)
    random_seed(configs.seed, configs.rank)

    model = create_model(configs, device, model_dtype)
    task = create_learning_task(configs, model).to(device)

    train_dataset = create_dataset(configs, is_training=True)
    eval_dataset = create_dataset(configs, is_training=False)
    train_loader = create_dataloader(configs, train_dataset, is_training=True)
    eval_loader = create_dataloader(configs, eval_dataset, is_training=False)

    optimizer = create_optimizer(configs, model)
    updates_per_epoch = (
        len(train_loader) + configs.grad_accum_steps - 1
    ) // configs.grad_accum_steps
    lr_scheduler = create_scheduler(configs, optimizer, updates_per_epoch)

    if configs.distributed:
        task = DistributedDataParallel(
            task, device_ids=[device] if device.type == "cuda" else None
        )

    trainer = Trainer(
        configs,
        task,
        train_loader,
        eval_loader,
        optimizer,
        lr_scheduler,
        device,
        output_dir=setup_output_dir(configs),
        amp_autocast=amp_autocast,
        loss_scaler=loss_scaler,
        model_dtype=model_dtype,
    )
    if configs.resume:
        trainer.resume(configs.resume)
    trainer.train()

    if configs.distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
