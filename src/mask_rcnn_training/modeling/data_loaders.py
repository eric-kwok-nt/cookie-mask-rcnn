"""This module contains functions that assist in loading datasets
for the model training pipeline."""

from pathlib import Path
import torch
from . import train_utils as utils
from .coco_utils import get_coco
from . import transforms as T


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

    trg_dataset = get_coco(
        data_path,
        "train",
        transforms=_get_transform(train=True),
        mode="instances",
    )
    trg_dataloader = torch.utils.data.DataLoader(
        trg_dataset,
        batch_size=args["train"]["batch_size"],
        shuffle=True,
        num_workers=args["train"]["num_workers"],
        collate_fn=utils.collate_fn,
        prefetch_factor=args["train"]["prefetch_factor"],
    )

    val_dataset = get_coco(
        data_path,
        "val",
        transforms=_get_transform(train=False),
        mode="instances",
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
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


def _get_transform(train: bool):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
