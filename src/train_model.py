import os
import logging
import hydra
import torch
from torch.utils.tensorboard import SummaryWriter
import mask_rcnn_training as mrt
from mask_rcnn_training.modeling.train_engine import train_one_epoch, evaluate


@hydra.main(config_path="../conf/base", config_name="pipelines.yml")
def main(args):
    """This main function does the following:
    - load logging config
    - initialise experiment tracking (MLflow)
    - loads training, validation and test data
    - initialises model layers and compile
    - trains, evaluates, and then exports the model
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter("runs/maskrcnn_experiment_1")

    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    logger_config_path = os.path.join(
        hydra.utils.get_original_cwd(), "conf/base/logging.yml"
    )
    mrt.general_utils.setup_logging(logger_config_path)

    datasets = mrt.modeling.data_loaders.get_dataloader(
        hydra.utils.get_original_cwd(), args
    )

    model = mrt.modeling.models.maskrcnn_model(args)
    model.to(device)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    logger.info("Training the model...")

    for epoch in range(args["train"]["epochs"]):
        metric_logger = train_one_epoch(
            model, optimizer, datasets["train"], device, epoch, print_freq=10, writer=writer
        )
        lr_scheduler.step()
        logger.info("Evaluating the model...")
        evaluate(model, datasets["val"], device)

        logger.info("Exporting the model...")
        mrt.modeling.utils.export_model(model)

    
    writer.close()
    logger.info("Model training has completed.")


if __name__ == "__main__":
    main()
