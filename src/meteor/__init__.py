"""
Init
"""
from . import _version
from .meteor import MeteorPatternScaling  # noqa: F401

__version__ = _version.get_versions()["version"]
