import os
import logging
import hydra
import torch
from torch.utils.tensorboard import SummaryWriter
import mask_rcnn_training as mrt
from mask_rcnn_training.modeling.train_engine import train_one_epoch, evaluate
from mask_rcnn_training.modeling.coco_eval import COCOStats


def load_optimizer_scheduler(args, params):
    optimizer = torch.optim.SGD(
        params,
        lr=args["train"]["initial_lr"],
        momentum=args["train"]["momentum"],
        weight_decay=args["train"]["weight_decay"],
    )
    if args["train"]["lr_scheduler"] == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args["train"]["lr_scheduler_step_size"],
            gamma=args["train"]["lr_scheduler_gamma"],
        )
    elif args["train"]["lr_scheduler"] == "reduceonplateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args["train"]["lr_scheduler_gamma"],
            patience=args["train"]["lr_scheduler_patience"],
        )
    return optimizer, lr_scheduler


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
    C_stats = COCOStats()

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

    train_data = dict()
    last_epoch = -1

    optimizer, lr_scheduler = load_optimizer_scheduler(args, params)

    if args["train"]["saved_model_path"] is not None:
        logger.info(f"Loading model from {args['train']['saved_model_path']}")
        if not args["train"]["new_optimizer"]:
            if not args["train"]["new_lr_scheduler"]:
                train_data = mrt.modeling.utils.load_model(
                    args["train"]["saved_model_path"], model, optimizer, lr_scheduler
                )
            else:
                train_data = mrt.modeling.utils.load_model(
                    args["train"]["saved_model_path"], model, optimizer
                )
        else:
            train_data = mrt.modeling.utils.load_model(
                args["train"]["saved_model_path"], model
            )
        last_epoch = train_data["epoch"]
        logger.info(f"Starting from epoch {last_epoch}")
    logger.info("Training the model...")

    step = [0]
    for epoch in range(last_epoch + 1, args["train"]["epochs"]):
        metric_logger = train_one_epoch(
            model,
            optimizer,
            datasets["train"],
            device,
            epoch,
            print_freq=10,
            writer=writer,
            step=step,
        )
        logger.info("Evaluating the model...")
        coco_evaluator = evaluate(model, datasets["val"], device)

        segm_ap = coco_evaluator.coco_eval["segm"].stats[0]
        if args["train"]["lr_scheduler"] == "step":
            lr_scheduler.step()
        elif args["train"]["lr_scheduler"] == "reduceonplateau":
            lr_scheduler.step(metrics=segm_ap)

        logger.info("Exporting the model...")
        train_data["model"] = model
        train_data["optimizer"] = optimizer
        train_data["lr_scheduler"] = lr_scheduler
        train_data["epoch"] = epoch
        mrt.modeling.utils.export_model(args, train_data)

        C_stats.update_bbox_results(coco_evaluator.coco_eval["bbox"].stats)
        C_stats.update_segm_results(coco_evaluator.coco_eval["segm"].stats)

        for k, v in C_stats.overall_results.items():
            writer.add_scalar(k, v, step[0])

    hparam_keys = [
        "epochs",
        "backbone",
        "trainable_layers",
        "batch_size",
        "initial_lr",
        "momentum",
        "weight_decay",
        "lr_scheduler_step_size",
        "lr_scheduler_gamma",
    ]

    writer.add_hparams(
        hparam_dict={key: args["train"][key] for key in hparam_keys},
        metric_dict=C_stats.overall_results,
    )

    writer.close()
    logger.info("Model training has completed.")


if __name__ == "__main__":
    main()
