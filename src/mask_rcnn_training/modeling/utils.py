"""Any miscellaneous utilities/functions to assist the model
training workflow are to be contained here."""

from pathlib import Path
import hydra
import train_utils
import torch


def export_model(args, train_data):
    """Serialises and exports the trained weights with it training parameters.

    Parameters
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


def load_model(path, model, optimizer=None):
    """Function to load the predictive model.

    A utility function to be used for loading a PyTorch model / checkpoint
    saved in '.pth' format.

    Parameters
    ----------
    path : str
        Path to a directory containing a Keras model in
        'SavedModel' format.
    model : torch.nn.Module
        MaskRCNN model to be loaded
    optimizer : (Optional) torch.optim.Optimizer

    Returns
    -------
    model : torch.nn.Module
        Loaded model.
    optimizer : (Optional) torch.optim.Optimizer
        Loaded optimizer. Loaded if the input optimizer is not None.
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        return model, optimizer
    return model
