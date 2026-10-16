"""
GGCM crop yield emulation for METEOR
=====================================

Vendored and adapted from the GGCMI Phase 2 emulator
(Franke et al., 2020, https://doi.org/10.5194/gmd-13-3995-2020).

Coefficient files (~110 MB each) are downloaded on demand from
Zenodo record 3592453 and stored in METEOR's cache directory.
The AgMERRA 1980-2010 climatological baseline is bundled with this package.
"""

from .baseline import load_agmerra_baseline
from .coefficients import get_yields, load_coefficients
from .data_catalog import (
    CROP_MODELS,
    CROPS,
    get_available_crops,
    get_available_models,
    get_download_url,
    get_filename,
    is_available,
)
from .downloader import GgcmDownloader

__all__ = [
    "load_agmerra_baseline",
    "get_yields",
    "load_coefficients",
    "GgcmDownloader",
    "CROPS",
    "CROP_MODELS",
    "is_available",
    "get_filename",
    "get_download_url",
    "get_available_crops",
    "get_available_models",
]
