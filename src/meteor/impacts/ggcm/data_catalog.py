"""
Static catalog of GGCM coefficient files hosted on Zenodo record 3592453.

Reference: Franke, J. A., et al. (2020). The GGCMI Phase 2 emulators: global
gridded crop model yield responses to changes in CO2, temperature, water, and
nitrogen. Geoscientific Model Development, 13, 3995-4018.
https://doi.org/10.5194/gmd-13-3995-2020
"""

ZENODO_RECORD_ID = "3592453"
ZENODO_BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

CROPS = ["maize", "rice", "soy", "spring_wheat", "winter_wheat"]

CROP_MODELS = [
    "CARAIB",
    "EPIC-TAMU",
    "GEPIC",
    "JULES",
    "LPJ-GUESS",
    "LPJmL",
    "pDSSAT",
    "PEPIC",
    "PROMET",
]

# Availability matrix: model → crop → list of available variants
FILE_AVAILABILITY = {
    "CARAIB": {
        "maize": ["A0", "A1"],
        "rice": ["A0", "A1"],
        "soy": ["A0", "A1"],
        "spring_wheat": ["A0", "A1"],
        "winter_wheat": ["A0", "A1"],
    },
    "EPIC-TAMU": {
        "maize": ["A0", "A1"],
        "rice": ["A0", "A1"],
        "soy": ["A0", "A1"],
        "spring_wheat": ["A0", "A1"],
        "winter_wheat": ["A0", "A1"],
    },
    "GEPIC": {
        "maize": ["A0", "A1"],
        "rice": ["A0", "A1"],
        "soy": ["A0", "A1"],
        "spring_wheat": ["A0", "A1"],
        "winter_wheat": ["A0", "A1"],
    },
    "JULES": {
        "maize": ["A0"],
        "rice": ["A0"],
        "soy": ["A0"],
        "spring_wheat": ["A0"],
    },
    "LPJ-GUESS": {
        "maize": ["A0", "A1"],
        "rice": ["A0", "A1"],
        "spring_wheat": ["A0", "A1"],
        "winter_wheat": ["A0", "A1"],
    },
    "LPJmL": {
        "maize": ["A0", "A1"],
        "rice": ["A0", "A1"],
        "soy": ["A0", "A1"],
        "spring_wheat": ["A0", "A1"],
        "winter_wheat": ["A0", "A1"],
    },
    "pDSSAT": {
        "maize": ["A0", "A1"],
        "rice": ["A0", "A1"],
        "soy": ["A0", "A1"],
        "spring_wheat": ["A0", "A1"],
        "winter_wheat": ["A0", "A1"],
    },
    "PEPIC": {
        "maize": ["A0", "A1"],
        "rice": ["A0", "A1"],
        "soy": ["A0", "A1"],
        "spring_wheat": ["A0", "A1"],
        "winter_wheat": ["A0", "A1"],
    },
    "PROMET": {
        "maize": ["A0", "A1"],
        "rice": ["A0", "A1"],
        "soy": ["A0", "A1"],
        "spring_wheat": ["A0", "A1"],
        "winter_wheat": ["A0", "A1"],
    },
}


def is_available(crop_model, crop, variant="A0"):
    """Return True if this model/crop/variant combination exists on Zenodo."""
    if crop_model not in FILE_AVAILABILITY:
        return False
    if crop not in FILE_AVAILABILITY[crop_model]:
        return False
    return variant in FILE_AVAILABILITY[crop_model][crop]


def get_filename(crop_model, crop, variant="A0"):
    """Return the standardised filename for a given model/crop/variant."""
    return f"{crop_model}_{crop}_ggcmi_phase2_emulator_{variant}.nc4"


def get_download_url(crop_model, crop, variant="A0"):
    """Return the full Zenodo download URL.

    Raises ValueError if the combination is not available.
    """
    if not is_available(crop_model, crop, variant):
        raise ValueError(f"Not available: {crop_model} / {crop} / {variant}")
    return f"{ZENODO_BASE_URL}/{get_filename(crop_model, crop, variant)}"


def get_available_crops(crop_model=None):
    """Return list of crops, optionally filtered to a specific model."""
    if crop_model is None:
        return CROPS
    return list(FILE_AVAILABILITY.get(crop_model, {}).keys())


def get_available_models(crop=None):
    """Return list of models, optionally filtered to those supporting a crop."""
    if crop is None:
        return CROP_MODELS
    return [m for m in CROP_MODELS if crop in FILE_AVAILABILITY.get(m, {})]


def get_available_variants(crop_model, crop):
    """Return list of available variants for a model/crop combination."""
    return FILE_AVAILABILITY.get(crop_model, {}).get(crop, [])
