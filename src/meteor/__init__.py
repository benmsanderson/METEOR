"""
Init
"""

# Import impacts submodule for easy access
from . import impacts  # noqa: F401
from . import _version
from .cmip6_meteor_data_getter import Cmip6MeteorDataGetter  # noqa: F401
from .meteor import MeteorPatternScaling  # noqa: F401
from .noise_generator import (  # noqa: F401
    MeteorNoiseGenerator,
    train_noise_model_from_cmip6,
    train_noise_model_from_composite,
)
from .precipitation_transform import (  # noqa: F401
    fit_distribution_parameters_1d,
    fit_distribution_parameters_3d,
    apply_distribution_transform,
    apply_empirical_quantile_mapping,
)
from .prpatt import (  # noqa: F401
    global_mean,
    regional_mean,
    extract_point,
    list_ar6_regions,
)
from .meteor_interface import MeteorInterface  # noqa: F401
from .ensemble_output import EnsembleOutput, VariableOutput  # noqa: F401

__version__ = _version.get_versions()["version"]
