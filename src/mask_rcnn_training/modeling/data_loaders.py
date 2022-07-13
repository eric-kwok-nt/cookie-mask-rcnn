"""This module contains functions that assist in loading datasets
for the model training pipeline."""

from pathlib import Path
import torch
from torchvision.transforms import InterpolationMode
from . import train_utils as utils
from .coco_utils import get_coco
from . import transforms as T
from .group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups


def get_dataloader(current_working_dir, args):
    """Load datasets specified through YAML config.

    Paramaters
    ----------
    args : dict
        Dictionary containing the pipeline's configuration passed from
        Hydra.

    Returns
    -------
    dict
        Dictionary object for which its values are "torch.utils.data.DataLoader"
        objects.
    """

    data_path = Path(current_working_dir) / args["train"]["data_path"]

    # Get the dataloader for the training dataset
    train_collate_fn = utils.collate_fn
    if args["train"]["copy_paste"]:
        copypaste = T.SimpleCopyPaste(
            resize_interpolation=InterpolationMode.BILINEAR, blending=True
        )

        def copypaste_collate_fn(batch):
            return copypaste(*utils.collate_fn(batch))

        train_collate_fn = copypaste_collate_fn

    trg_dataset = get_coco(
        data_path,
        "train",
        transforms=_get_transform(
            train=True,
            scale_jitter=args["train"]["scale_jitter"],
            rnd_photometric_distort=args["train"]["rnd_photometric_distort"],
        ),
        mode="instances",
    )

    train_sampler = torch.utils.data.RandomSampler(trg_dataset)
    if args["train"]["aspect_ratio_group_factor"] >= 0:
        group_ids = create_aspect_ratio_groups(
            trg_dataset, k=args["train"]["aspect_ratio_group_factor"]
        )
        train_batch_sampler = GroupedBatchSampler(
            train_sampler, group_ids, args["train"]["batch_size"]
        )
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args["train"]["batch_size"], drop_last=True
        )

    trg_dataloader = torch.utils.data.DataLoader(
        trg_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=args["train"]["num_workers"],
        collate_fn=train_collate_fn,
        prefetch_factor=args["train"]["prefetch_factor"],
    )

    # Get the loader for the validation dataset
    val_dataset = get_coco(
        data_path,
        "val",
        transforms=_get_transform(
            train=False,
            scale_jitter=False,
            rnd_photometric_distort=False,
        ),
        mode="instances",
    )
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=args["train"]["batch_size"],
        shuffle=False,
        num_workers=args["train"]["num_workers"],
        collate_fn=utils.collate_fn,
        prefetch_factor=args["train"]["prefetch_factor"],
    )

    dataloaders = {
        "train": trg_dataloader,
        "val": val_dataloader,
    }

    return dataloaders


# def _get_transform(
#     train: bool,
#     scale_jitter: bool = False,
#     rnd_photometric_distort: bool = False,
# ):
#     transforms = []
#     transforms.append(T.ToTensor())
#     if train:
#         transforms.append(T.RandomHorizontalFlip(0.5))
#         if rnd_photometric_distort:
#             transforms.append(
#                 T.RandomPhotometricDistort(
#                     contrast=(0.5, 1.5),
#                     saturation=(0.5, 1.5),
#                     hue=(-0.05, 0.05),
#                     brightness=(0.875, 1.125),
#                     p=0.5,
#                 )
#             )
#         if scale_jitter:
#             transforms.append(T.ScaleJitter((800, 1333)))
#     return T.Compose(transforms)


def _get_transform(train: bool):
    transforms = []
    if train:
        transforms.append(T.ScaleJitter(target_size=(800, 1333)))
        transforms.append(T.FixedSizeCrop(size=(800, 1333), fill=(123.0, 117.0, 104.0)))
        transforms.append(T.RandomHorizontalFlip(p=0.5))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)
