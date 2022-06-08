import mask_rcnn_training as mrt
import mask_rcnn_training_fastapi as mrt_fapi


PRED_MODEL = mrt.modeling.utils.load_model(
    mrt_fapi.config.SETTINGS.PRED_MODEL_PATH)
