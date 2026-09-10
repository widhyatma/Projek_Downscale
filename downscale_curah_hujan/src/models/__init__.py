"""
Model Benchmark Suite Package (13 Geospatial, Spline, ML, and Fusion Algorithms)
"""

from .baselines import IDWModel, GWRModel
from .geostats import OrdinaryKrigingModel, KEDModel, RegressionKrigingModel
from .topospline import ANUSPLINModel, PRISMModel
from .tree_models import RFSIModel, SpatialRandomForestModel, SpatialXGBoostModel, SpatialLightGBMModel
from .fusion import ConditionalMergingModel

__all__ = [
    "IDWModel",
    "GWRModel",
    "OrdinaryKrigingModel",
    "KEDModel",
    "RegressionKrigingModel",
    "ANUSPLINModel",
    "PRISMModel",
    "RFSIModel",
    "SpatialRandomForestModel",
    "SpatialXGBoostModel",
    "SpatialLightGBMModel",
    "ConditionalMergingModel"
]
