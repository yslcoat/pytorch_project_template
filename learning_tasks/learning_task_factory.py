import torch
import torch.nn as nn


class LearningTask(nn.Module):
    """Wraps a model and its loss. Subclass this to add new tasks (segmentation, distillation, ...).

    forward(input, target) returns a dict with at least 'loss' and 'output'.
    metrics(output, target) returns a dict of per-batch metric sums, averaged over samples by the Trainer.
    """

    eval_metric = "loss"  # metric used to select the best checkpoint
    eval_metric_higher_is_better = False

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input, target):
        raise NotImplementedError

    def metrics(self, output, target):
        return {}


class ClassificationTask(LearningTask):
    eval_metric = "top1"
    eval_metric_higher_is_better = True

    def __init__(self, model, smoothing=0.0):
        super().__init__(model)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)

    def forward(self, input, target):
        output = self.model(input)
        return {"loss": self.criterion(output, target), "output": output}

    @torch.no_grad()
    def metrics(self, output, target):
        topk = min(5, output.shape[-1])
        pred = output.topk(topk, dim=-1).indices
        correct = pred.eq(target.unsqueeze(-1))
        return {
            "top1": correct[:, :1].sum().float() * 100.0,
            f"top{topk}": correct.any(dim=-1).sum().float() * 100.0,
        }


def create_learning_task(configs, model):
    if configs.task == "classification":
        return ClassificationTask(model, smoothing=configs.smoothing)
    raise ValueError(f'Unknown task "{configs.task}".')
