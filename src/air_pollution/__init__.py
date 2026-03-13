"""Air pollution forecasting package."""

from .data import FEATURE_COLUMNS, TARGET_COLUMN, build_inference_frame, prepare_training_frame

__all__ = ["FEATURE_COLUMNS", "TARGET_COLUMN", "build_inference_frame", "prepare_training_frame"]