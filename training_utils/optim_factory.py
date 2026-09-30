import math

import torch


def _param_groups(model, weight_decay):
    """No weight decay for biases and normalization parameters (all 1-d tensors)."""
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim <= 1 else decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def create_optimizer(configs, model):
    params = _param_groups(model, configs.weight_decay)
    opt = configs.opt.lower()
    if opt == "sgd":
        return torch.optim.SGD(
            params, lr=configs.lr, momentum=configs.momentum, nesterov=True
        )
    if opt == "adamw":
        return torch.optim.AdamW(params, lr=configs.lr)
    raise ValueError(f'Unknown optimizer "{configs.opt}".')


def create_scheduler(configs, optimizer, updates_per_epoch):
    """Per-update LR schedule: linear warmup from warmup_lr, then cosine decay to min_lr (or constant)."""
    total_updates = configs.epochs * updates_per_epoch
    warmup_updates = configs.warmup_epochs * updates_per_epoch
    base_lr = configs.lr

    def lr_factor(step):
        if step < warmup_updates:
            lr = (
                configs.warmup_lr
                + (base_lr - configs.warmup_lr) * step / warmup_updates
            )
        elif configs.sched == "cosine":
            progress = (step - warmup_updates) / max(1, total_updates - warmup_updates)
            lr = configs.min_lr + 0.5 * (base_lr - configs.min_lr) * (
                1 + math.cos(math.pi * min(1.0, progress))
            )
        elif configs.sched == "constant":
            lr = base_lr
        else:
            raise ValueError(f'Unknown scheduler "{configs.sched}".')
        return lr / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
