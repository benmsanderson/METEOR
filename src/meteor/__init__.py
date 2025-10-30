"""
Init
"""

from . import _version
from .cmip6_meteor_data_getter import Cmip6MeteorDataGetter  # noqa: F401
from .meteor import MeteorPatternScaling  # noqa: F401
from .noise_generator import (
    MeteorNoiseGenerator,
    train_noise_model_from_composite,
)  # noqa: F401

# Import impacts submodule for easy access
from . import impacts  # noqa: F401

__version__ = _version.get_versions()["version"]
