"""This module provides definitions of predictive models to be
trained."""

from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def maskrcnn_model(args):
    """Initialise a maskrcnn model.

    Paramaters
    ----------
    args : dict
        Dictionary containing the pipeline's configuration passed from
        Hydra.

    Returns
    -------
    nn.Module
        MaskRCNN model.
    """

    backbone = resnet_fpn_backbone(
        backbone_name=args["train"]["backbone"],
        pretrained=True,
        norm_layer=misc_nn_ops.FrozenBatchNorm2d,
        trainable_layers=args["train"]["trainable_layers"],
        _skip_resize=True,
    )
    model = MaskRCNN(backbone, num_classes=91)

    return model
