import logging
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from timm.utils.clip_grad import dispatch_clip_grad
from timm.utils.distributed import is_primary

_logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        configs,
        task,
        train_loader,
        eval_loader,
        optimizer,
        lr_scheduler,
        device,
        output_dir=None,
        amp_autocast=nullcontext,
        loss_scaler=None,
        model_dtype=None,
    ):
        self.configs = configs
        self.task = task  # possibly DDP wrapped, use for forward/backward
        self.task_module = (
            task.module if isinstance(task, DistributedDataParallel) else task
        )
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.output_dir = output_dir
        self.amp_autocast = amp_autocast
        self.loss_scaler = loss_scaler
        self.model_dtype = model_dtype

        self.start_epoch = 0
        self.best_metric = None
        self.best_epoch = None
        self.is_primary = is_primary(configs)

    def _to_device(self, input, target):
        input = input.to(device=self.device, dtype=self.model_dtype, non_blocking=True)
        target = target.to(device=self.device, non_blocking=True)
        if self.configs.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        return input, target

    def process_batch(self, input, target, accum_steps, need_update):
        """Forward + backward for one loader batch. Steps the optimizer when need_update is True."""
        input, target = self._to_device(input, target)

        sync_ctx = (
            self.task.no_sync()
            if isinstance(self.task, DistributedDataParallel) and not need_update
            else nullcontext()
        )
        with sync_ctx:
            with self.amp_autocast():
                result = self.task(input, target)
            loss = result["loss"]
            scaled_loss = loss / accum_steps

            params = self.task_module.parameters()
            if self.loss_scaler is not None:
                self.loss_scaler(
                    scaled_loss,
                    self.optimizer,
                    clip_grad=self.configs.clip_grad,
                    parameters=params,
                    need_update=need_update,
                )
            else:
                scaled_loss.backward()
                if need_update:
                    if self.configs.clip_grad is not None:
                        dispatch_clip_grad(params, value=self.configs.clip_grad)
                    self.optimizer.step()

        if need_update:
            self.optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()
        return loss.detach(), input.shape[0]

    def process_epoch(self, epoch):
        self.task.train()
        loader = self.train_loader
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        num_batches = len(loader)
        accum_steps = self.configs.grad_accum_steps
        updates_per_epoch = (num_batches + accum_steps - 1) // accum_steps
        # the final update of the epoch may accumulate fewer batches
        last_accum_steps = num_batches % accum_steps or accum_steps
        last_accum_start = num_batches - last_accum_steps

        loss_sum = torch.zeros((), device=self.device)
        sample_count = 0
        start_time = time.time()
        self.optimizer.zero_grad(set_to_none=True)
        for batch_idx, (input, target) in enumerate(loader):
            batch_accum_steps = (
                last_accum_steps if batch_idx >= last_accum_start else accum_steps
            )
            need_update = (
                batch_idx == num_batches - 1 or (batch_idx + 1) % accum_steps == 0
            )
            loss, batch_size = self.process_batch(
                input, target, batch_accum_steps, need_update
            )
            loss_sum += loss * batch_size
            sample_count += batch_size

            update_idx = batch_idx // accum_steps
            if need_update and (
                update_idx % self.configs.log_interval == 0
                or batch_idx == num_batches - 1
            ):
                if self.is_primary:
                    lr = self.optimizer.param_groups[0]["lr"]
                    elapsed = time.time() - start_time
                    _logger.info(
                        f"Train: {epoch} [{update_idx:>4d}/{updates_per_epoch}]  "
                        f"Loss: {loss.item():#.3g} ({(loss_sum / sample_count).item():#.3g})  "
                        f"LR: {lr:.3e}  "
                        f"Throughput: {sample_count / elapsed:>7.1f} samples/s"
                    )

        return {"loss": self._reduce_mean(loss_sum, sample_count)}

    @torch.inference_mode()
    def validate(self, epoch):
        self.task_module.eval()
        loss_sum = torch.zeros((), device=self.device)
        metric_sums = {}
        sample_count = 0
        for input, target in self.eval_loader:
            input, target = self._to_device(input, target)
            with self.amp_autocast():
                result = self.task_module(input, target)
            batch_size = input.shape[0]
            loss_sum += result["loss"].float() * batch_size
            for name, value in self.task_module.metrics(
                result["output"], target
            ).items():
                metric_sums[name] = metric_sums.get(name, 0.0) + value
            sample_count += batch_size

        metrics = {"loss": self._reduce_mean(loss_sum, sample_count)}
        for name, value in metric_sums.items():
            metrics[name] = self._reduce_mean(value, sample_count)
        if self.is_primary:
            _logger.info(
                f"Eval: {epoch}  "
                + "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            )
        return metrics

    def _reduce_mean(self, value_sum, count):
        """Sample-weighted mean of a per-rank sum across all ranks."""
        t = torch.stack(
            [
                torch.as_tensor(value_sum, device=self.device).float(),
                torch.tensor(float(count), device=self.device),
            ]
        )
        if self.configs.distributed:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return (t[0] / t[1].clamp(min=1)).item()

    def _checkpoint_state(self, epoch, metrics):
        state = {
            "epoch": epoch,
            "model": self.task_module.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "metrics": metrics,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "configs": vars(self.configs),
        }
        if self.loss_scaler is not None:
            state["loss_scaler"] = self.loss_scaler.state_dict()
        return state

    def save_checkpoint(self, epoch, metrics):
        if not self.is_primary or not self.output_dir:
            return
        eval_metric = self.task_module.eval_metric
        value = metrics.get(eval_metric)
        is_best = False
        if value is not None:
            higher = self.task_module.eval_metric_higher_is_better
            is_best = self.best_metric is None or (
                value > self.best_metric if higher else value < self.best_metric
            )
            if is_best:
                self.best_metric, self.best_epoch = value, epoch

        state = self._checkpoint_state(epoch, metrics)
        last_path = os.path.join(self.output_dir, "last.pth")
        torch.save(state, last_path + ".tmp")
        os.replace(last_path + ".tmp", last_path)
        if is_best:
            torch.save(state, os.path.join(self.output_dir, "best.pth"))
            _logger.info(f"New best {eval_metric}: {value:.4f} (epoch {epoch})")

    def resume(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.task_module.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if self.loss_scaler is not None and "loss_scaler" in checkpoint:
            self.loss_scaler.load_state_dict(checkpoint["loss_scaler"])
        self.best_metric = checkpoint.get("best_metric")
        self.best_epoch = checkpoint.get("best_epoch")
        self.start_epoch = checkpoint["epoch"] + 1
        if self.is_primary:
            _logger.info(f"Resumed from {path}, starting at epoch {self.start_epoch}.")

    def train(self):
        epochs = self.configs.epochs
        for epoch in range(self.start_epoch, epochs):
            train_metrics = self.process_epoch(epoch)

            is_last = epoch == epochs - 1
            if (epoch + 1) % self.configs.val_interval == 0 or is_last:
                eval_metrics = (
                    self.validate(epoch) if self.eval_loader is not None else {}
                )
                self.save_checkpoint(
                    epoch, {"train_loss": train_metrics["loss"], **eval_metrics}
                )

        if self.is_primary and self.best_metric is not None:
            _logger.info(
                f"Best {self.task_module.eval_metric}: {self.best_metric:.4f} (epoch {self.best_epoch})"
            )
        return self.best_metric
