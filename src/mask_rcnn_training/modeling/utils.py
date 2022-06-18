"""Any miscellaneous utilities/functions to assist the model
training workflow are to be contained here."""

from pathlib import Path
import hydra
from . import train_utils
import torch


def export_model(args, train_data):
    """Serialises and exports the trained weights with it training parameters.

    Arguments
    ----------
    args : dict
        Dictionary containing the pipeline's configuration passed from
        Hydra.
    train_data : dict
        model: Trained model,
        optimizer: Trained optimizer,
        lr_scheduler: Trained learning rate scheduler,
        epoch: Current epoch,
    """
    train_utils.save_on_master(
        {
            "model_state_dict": train_data["model"].state_dict(),
            "optimizer": train_data["optimizer"].state_dict(),
            "lr_scheduler": train_data["lr_scheduler"].state_dict(),
            "epoch": train_data["epoch"],
        },
        Path(hydra.utils.get_original_cwd())
        / "models/maskrcnn_{}_{}.pth".format(
            args["train"]["backbone"], train_data["epoch"]
        ),
    )


def load_model(path, model, optimizer=None, lr_scheduler=None):
    """Function to load the predictive model.

    A utility function to be used for loading a PyTorch model / checkpoint
    saved in '.pth' format.

    Arguments
    ----------
    path : str
        Path to a directory containing a Keras model in
        'SavedModel' format.
    model : torch.nn.Module
        MaskRCNN model to be loaded
    optimizer : (Optional) torch.optim.Optimizer
        Optimizer to be loaded
    lr_scheduler : (Optional) torch.optim.lr_scheduler.LrScheduler
        Learning rate scheduler to be loaded

    Returns
    -------
    dict : A dictionary containing the loaded model, optimizer, lr_scheduler,
        and epoch. If the optimizer or lr_scheduler are not provided, they are
        not loaded. The keys in the dictionary are 'model', 'optimizer',
        'lr_scheduler', and 'epoch'.
    """
    output_dict = dict()

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    last_epoch = checkpoint["epoch"]

    output_dict["epoch"] = last_epoch
    output_dict["model"] = model
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        output_dict["optimizer"] = optimizer
        if lr_scheduler is not None:
            try:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                output_dict["lr_scheduler"] = lr_scheduler
                return output_dict
            except KeyError:
                pass

    return output_dict
