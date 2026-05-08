from .condition_builder import ConditionBranchOutput, DualConditionBuilder
from .expression_adapter import ExpressionAdapter
from .time_embedding import sinusoidal_embedding_1d
from .diffusion_forward import model_fn_emotion_identity

__all__ = [
    "ConditionBranchOutput",
    "DualConditionBuilder",
    "ExpressionAdapter",
    "model_fn_emotion_identity",
    "sinusoidal_embedding_1d",
    "ModelManager",
    "load_state_dict",
]


def __getattr__(name):
    if name == "ModelManager":
        from .model_manager import ModelManager

        return ModelManager
    if name == "load_state_dict":
        from .utils import load_state_dict

        return load_state_dict
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
