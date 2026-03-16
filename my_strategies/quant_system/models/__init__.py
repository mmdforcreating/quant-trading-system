from .catboost_model import CatBoostQuantModel
from .ridge_model import RidgeQuantModel

MODEL_REGISTRY = {
    "CatBoostQuantModel": CatBoostQuantModel,
    "RidgeQuantModel": RidgeQuantModel,
}

try:
    from .lgbm_model import LightGBMQuantModel, LambdaRankQuantModel
    MODEL_REGISTRY["LightGBMQuantModel"] = LightGBMQuantModel
    MODEL_REGISTRY["LambdaRankQuantModel"] = LambdaRankQuantModel
except ImportError:
    pass

try:
    from .extratrees_model import ExtraTreesQuantModel
    MODEL_REGISTRY["ExtraTreesQuantModel"] = ExtraTreesQuantModel
except ImportError:
    pass

try:
    from .highfreq_gru import HighFreqGRUModel
    MODEL_REGISTRY["HighFreqGRUModel"] = HighFreqGRUModel
except ImportError:
    pass

try:
    from .gru_wf_adapter import GRUQuantModel
    MODEL_REGISTRY["GRUQuantModel"] = GRUQuantModel
except ImportError:
    pass

__all__ = list(MODEL_REGISTRY.keys()) + ["MODEL_REGISTRY"]
