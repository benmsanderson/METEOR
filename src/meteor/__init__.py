"""
Init
"""

# Import impacts submodule for easy access
from . import impacts  # noqa: F401
from . import _version
from .cmip6_meteor_data_getter import Cmip6MeteorDataGetter  # noqa: F401
from .ensemble_output import EnsembleOutput, VariableOutput  # noqa: F401
from .meteor import MeteorPatternScaling  # noqa: F401
from .meteor_interface import MeteorInterface  # noqa: F401
from .noise_generator import (  # noqa: F401
    MeteorNoiseGenerator,
    train_noise_model_from_cmip6,
    train_noise_model_from_composite,
)
from .precipitation_transform import (  # noqa: F401
    apply_distribution_transform,
    apply_empirical_quantile_mapping,
    fit_distribution_parameters_1d,
    fit_distribution_parameters_3d,
)
from .prpatt import (  # noqa: F401
    create_region_mask,
    extract_point,
    global_mean,
    list_ar6_regions,
    regional_mean,
)

__version__ = _version.get_versions()["version"]
