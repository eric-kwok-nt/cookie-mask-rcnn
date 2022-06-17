import os
import logging
import hydra
import mlflow
import torch
import mask_rcnn_training as mrt
from mask_rcnn_training.modeling.train_engine import train_one_epoch, evaluate
from mask_rcnn_training.modeling.coco_eval import COCOStats


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
    C_stats = COCOStats()

    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    logger_config_path = os.path.join(
        hydra.utils.get_original_cwd(), "conf/base/logging.yml"
    )
    mrt.general_utils.setup_logging(logger_config_path)

    mlflow_init_status, mlflow_run = mrt.general_utils.mlflow_init(
        args,
        setup_mlflow=args["train"]["setup_mlflow"],
        autolog=args["train"]["mlflow_autolog"],
    )
    mrt.general_utils.mlflow_log(mlflow_init_status, "log_params", params=args["train"])

    if "POLYAXON_RUN_UUID" in os.environ:
        mrt.general_utils.mlflow_log(
            mlflow_init_status,
            "log_param",
            key="polyaxon_run_uuid",
            value=os.environ["POLYAXON_RUN_UUID"],
        )

    datasets = mrt.modeling.data_loaders.get_dataloader(
        hydra.utils.get_original_cwd(), args
    )

    model = mrt.modeling.models.maskrcnn_model(args)
    model.to(device)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params,
        lr=args["train"]["initial_lr"],
        momentum=args["train"]["momentum"],
        weight_decay=args["train"]["weight_decay"],
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args["train"]["lr_scheduler_step_size"],
        gamma=args["train"]["lr_scheduler_gamma"],
    )
    logger.info("Training the model...")

    step = [0]
    train_data = dict()
    for epoch in range(args["train"]["epochs"]):
        _ = train_one_epoch(
            model,
            optimizer,
            datasets["train"],
            device,
            epoch,
            print_freq=10,
            step=step,
            mlflow_init_status=mlflow_init_status,
        )
        lr_scheduler.step()

        logger.info("Evaluating the model...")
        coco_evaluator = evaluate(model, datasets["val"], device)

        logger.info("Logging the model performance on MLflow...")
        C_stats.update_bbox_results(coco_evaluator.coco_eval["bbox"].stats)
        C_stats.update_segm_results(coco_evaluator.coco_eval["segm"].stats)
        mrt.general_utils.mlflow_log(
            mlflow_init_status,
            "log_metrics",
            metrics=C_stats.overall_results,
            step=step[0],
        )

        logger.info("Exporting the model...")
        train_data["model"] = model
        train_data["optimizer"] = optimizer
        train_data["lr_scheduler"] = lr_scheduler
        train_data["epoch"] = epoch
        mrt.modeling.utils.export_model(args, train_data)

    if mlflow_init_status:
        artifact_uri = mlflow.get_artifact_uri()
        logger.info("Artifact URI: {}".format(artifact_uri))
        mrt.general_utils.mlflow_log(
            mlflow_init_status, "log_params", params={"artifact_uri": artifact_uri}
        )
        logger.info(
            "Model training with MLflow run ID {} has completed.".format(
                mlflow_run.info.run_id
            )
        )
        mlflow.end_run()
    else:
        logger.info("Model training has completed.")


if __name__ == "__main__":
    main()
