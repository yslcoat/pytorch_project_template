from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def create_dataloader(configs, dataset, is_training=False):
    sampler = None
    if configs.distributed:
        # validation is also sharded, metrics are reduced across ranks in the Trainer
        sampler = DistributedSampler(
            dataset,
            num_replicas=configs.world_size,
            rank=configs.rank,
            shuffle=is_training,
        )

    return DataLoader(
        dataset,
        batch_size=configs.batch_size if is_training else configs.validation_batch_size,
        shuffle=is_training and sampler is None,
        sampler=sampler,
        num_workers=configs.workers,
        pin_memory=configs.pin_mem,
        persistent_workers=configs.persistent_workers and configs.workers > 0,
        drop_last=is_training,
    )
