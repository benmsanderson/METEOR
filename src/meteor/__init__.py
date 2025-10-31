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
    train_noise_model_from_composite,
)

__version__ = _version.get_versions()["version"]
