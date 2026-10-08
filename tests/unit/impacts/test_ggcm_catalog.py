"""
Unit tests for the GGCMI Phase 2 data catalog and downloader.
==============================================================

Verifies catalog metadata and the downloader's validation logic without
making any network requests.

Catalog covers 9 crop models × 5 crops × 2 adaptation variants (A0 / A1)
as described in Table 1 of Franke et al. (2020), GMD 13, 3995-4018.
"""

from unittest.mock import MagicMock, patch

import pytest

from meteor.impacts.ggcm import data_catalog as catalog
from meteor.impacts.ggcm.downloader import GgcmDownloader

# ---------------------------------------------------------------------------
# Catalog: known models and crops
# ---------------------------------------------------------------------------


def test_all_nine_models_present():
    """Nine crop models participated in GGCMI Phase 2 (Table 1)."""
    expected = {
        "CARAIB",
        "EPIC-TAMU",
        "GEPIC",
        "JULES",
        "LPJ-GUESS",
        "LPJmL",
        "pDSSAT",
        "PEPIC",
        "PROMET",
    }
    assert expected == set(catalog.CROP_MODELS)


def test_five_crops_present():
    expected = {"maize", "rice", "soy", "spring_wheat", "winter_wheat"}
    assert expected == set(catalog.CROPS)


def test_LPJmL_supports_all_crops_A0_A1():
    """LPJmL provides both A0 and A1 for all five crops."""
    for crop in catalog.CROPS:
        assert catalog.is_available("LPJmL", crop, "A0")
        assert catalog.is_available("LPJmL", crop, "A1")


def test_JULES_A0_only():
    """JULES participated only in A0 scenarios."""
    for crop in ["maize", "rice", "soy", "spring_wheat"]:
        assert catalog.is_available("JULES", crop, "A0")
        assert not catalog.is_available("JULES", crop, "A1")


def test_JULES_no_winter_wheat():
    """JULES did not provide winter wheat data."""
    assert not catalog.is_available("JULES", "winter_wheat", "A0")


def test_LPJ_GUESS_no_soy():
    """LPJ-GUESS did not simulate soy."""
    assert not catalog.is_available("LPJ-GUESS", "soy", "A0")


# ---------------------------------------------------------------------------
# Catalog: is_available()
# ---------------------------------------------------------------------------


def test_returns_false_for_unknown_model():
    assert not catalog.is_available("UNKNOWN_MODEL", "maize", "A0")


def test_returns_false_for_unknown_crop():
    assert not catalog.is_available("LPJmL", "quinoa", "A0")


def test_returns_false_for_unknown_variant():
    assert not catalog.is_available("LPJmL", "maize", "A2")


def test_returns_true_for_known_combination():
    assert catalog.is_available("pDSSAT", "spring_wheat", "A1")


# ---------------------------------------------------------------------------
# Catalog: get_filename()
# ---------------------------------------------------------------------------


def test_filename_pattern():
    name = catalog.get_filename("LPJmL", "maize", "A0")
    assert name == "LPJmL_maize_ggcmi_phase2_emulator_A0.nc4"


def test_filename_variant_A1():
    name = catalog.get_filename("pDSSAT", "rice", "A1")
    assert name == "pDSSAT_rice_ggcmi_phase2_emulator_A1.nc4"


def test_filename_contains_model_crop_variant():
    name = catalog.get_filename("GEPIC", "winter_wheat", "A0")
    assert "GEPIC" in name
    assert "winter_wheat" in name
    assert "A0" in name
    assert name.endswith(".nc4")


# ---------------------------------------------------------------------------
# Catalog: get_download_url()
# ---------------------------------------------------------------------------


def test_url_contains_record_id():
    url = catalog.get_download_url("LPJmL", "maize", "A0")
    assert catalog.ZENODO_RECORD_ID in url


def test_url_ends_with_filename():
    url = catalog.get_download_url("LPJmL", "maize", "A0")
    assert url.endswith(catalog.get_filename("LPJmL", "maize", "A0"))


def test_url_raises_for_unavailable():
    with pytest.raises(ValueError, match="Not available"):
        catalog.get_download_url("JULES", "maize", "A1")


def test_url_raises_for_unknown_model():
    with pytest.raises(ValueError):
        catalog.get_download_url("FAKE_MODEL", "maize", "A0")


# ---------------------------------------------------------------------------
# Catalog: get_available_crops() and get_available_models()
# ---------------------------------------------------------------------------


def test_get_available_crops_no_filter():
    crops = catalog.get_available_crops()
    assert set(crops) == set(catalog.CROPS)


def test_get_available_crops_filtered_by_model():
    crops = catalog.get_available_crops("JULES")
    # JULES has no winter wheat
    assert "winter_wheat" not in crops


def test_get_available_models_no_filter():
    models = catalog.get_available_models()
    assert set(models) == set(catalog.CROP_MODELS)


def test_get_available_models_filtered_by_crop():
    # LPJ-GUESS doesn't simulate soy; check it's excluded
    soy_models = catalog.get_available_models("soy")
    assert "LPJ-GUESS" not in soy_models
    assert "LPJmL" in soy_models


def test_get_available_variants_known():
    variants = catalog.get_available_variants("LPJmL", "maize")
    assert set(variants) == {"A0", "A1"}


def test_get_available_variants_A0_only():
    variants = catalog.get_available_variants("JULES", "maize")
    assert variants == ["A0"]


def test_get_available_variants_unknown_model():
    assert catalog.get_available_variants("UNKNOWN", "maize") == []


def test_get_available_variants_unknown_crop():
    assert catalog.get_available_variants("LPJmL", "quinoa") == []


# ---------------------------------------------------------------------------
# Downloader: validation without network calls
# ---------------------------------------------------------------------------


def test_get_filepath_returns_correct_path(tmp_path):
    dl = GgcmDownloader(str(tmp_path))
    fp = dl.get_filepath("LPJmL", "maize", "A0")
    assert fp == tmp_path / "LPJmL_maize_ggcmi_phase2_emulator_A0.nc4"


def test_ensure_files_raises_for_unavailable_crop(tmp_path):
    dl = GgcmDownloader(str(tmp_path))
    with pytest.raises(ValueError, match="not available"):
        dl.ensure_files(["quinoa"], "LPJmL", "A0")


def test_ensure_files_raises_for_unavailable_model(tmp_path):
    dl = GgcmDownloader(str(tmp_path))
    with pytest.raises(ValueError):
        dl.ensure_files(["maize"], "FAKE_MODEL", "A0")


def test_ensure_files_raises_for_JULES_A1(tmp_path):
    dl = GgcmDownloader(str(tmp_path))
    with pytest.raises(ValueError):
        dl.ensure_files(["maize"], "JULES", "A1")


def test_ensure_files_returns_early_if_all_present(tmp_path):
    """No download attempted when all files already exist."""
    dl = GgcmDownloader(str(tmp_path))
    # Create a dummy file so the downloader thinks it's already cached
    dummy = tmp_path / catalog.get_filename("LPJmL", "maize", "A0")
    dummy.touch()
    # Should return without error and without trying to import requests
    dl.ensure_files(["maize"], "LPJmL", "A0")  # no exception


def test_cache_dir_created_on_init(tmp_path):
    subdir = tmp_path / "ggcm" / "nested"
    GgcmDownloader(str(subdir))
    assert subdir.is_dir()


def test_ensure_files_raises_import_error_when_requests_missing(tmp_path):
    """If requests is not installed, ensure_files raises ImportError."""
    dl = GgcmDownloader(str(tmp_path))  # maize file does NOT exist
    with patch.dict("sys.modules", {"requests": None, "tqdm": None}):
        with pytest.raises(ImportError, match="requests"):
            dl.ensure_files(["maize"], "LPJmL", "A0")


def test_ensure_files_downloads_missing_files(tmp_path):
    """When files are missing, ensure_files calls _download_file for each."""
    dl = GgcmDownloader(str(tmp_path))
    mock_requests = MagicMock()
    mock_tqdm_cls = MagicMock()
    with (
        patch(
            "meteor.impacts.ggcm.downloader.GgcmDownloader._download_file"
        ) as mock_dl,
        patch(
            "builtins.__import__",
            side_effect=_make_importer(mock_requests, mock_tqdm_cls),
        ),
    ):
        dl.ensure_files(["maize", "rice"], "LPJmL", "A0")
        assert mock_dl.call_count == 2


def test_download_file_streams_chunks(tmp_path):
    """_download_file writes response chunks to disk."""
    dest = tmp_path / "test.nc4"
    chunks = [b"chunk1", b"chunk2", b"chunk3"]

    mock_resp = MagicMock()
    mock_resp.headers = {"content-length": str(sum(len(c) for c in chunks))}
    mock_resp.iter_content.return_value = iter(chunks)

    mock_requests = MagicMock()
    mock_requests.get.return_value = mock_resp

    mock_pbar = MagicMock()
    mock_pbar.__enter__ = MagicMock(return_value=mock_pbar)
    mock_pbar.__exit__ = MagicMock(return_value=False)
    mock_tqdm_cls = MagicMock(return_value=mock_pbar)

    dl = GgcmDownloader(str(tmp_path))
    dl._download_file("http://example.com/test.nc4", dest, mock_requests, mock_tqdm_cls)

    assert dest.exists()
    assert dest.read_bytes() == b"".join(chunks)
    mock_requests.get.assert_called_once_with(
        "http://example.com/test.nc4", headers={}, stream=True, timeout=60
    )


def test_download_file_resumes_partial(tmp_path):
    """_download_file sends a Range header when a partial file exists."""
    dest = tmp_path / "partial.nc4"
    existing = b"already_here"
    dest.write_bytes(existing)

    new_chunk = b"_rest"
    mock_resp = MagicMock()
    mock_resp.headers = {"content-length": str(len(new_chunk))}
    mock_resp.iter_content.return_value = iter([new_chunk])

    mock_requests = MagicMock()
    mock_requests.get.return_value = mock_resp

    mock_pbar = MagicMock()
    mock_pbar.__enter__ = MagicMock(return_value=mock_pbar)
    mock_pbar.__exit__ = MagicMock(return_value=False)
    mock_tqdm_cls = MagicMock(return_value=mock_pbar)

    dl = GgcmDownloader(str(tmp_path))
    dl._download_file(
        "http://example.com/partial.nc4", dest, mock_requests, mock_tqdm_cls
    )

    assert dest.read_bytes() == existing + new_chunk
    _, kwargs = mock_requests.get.call_args
    assert kwargs["headers"] == {"Range": f"bytes={len(existing)}-"}


# ---------------------------------------------------------------------------
# Helper for patching builtins.__import__ selectively
# ---------------------------------------------------------------------------

_real_import = (
    __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
)


def _make_importer(mock_requests, mock_tqdm_cls):
    """Return an __import__ side-effect that returns mocks for requests/tqdm."""

    def _import(name, *args, **kwargs):
        if name == "requests":
            return mock_requests
        if name == "tqdm":
            mod = MagicMock()
            mod.tqdm = mock_tqdm_cls
            return mod
        return _real_import(name, *args, **kwargs)

    return _import
