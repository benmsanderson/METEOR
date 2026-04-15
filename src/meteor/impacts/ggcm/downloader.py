"""
Download GGCMI Phase 2 coefficient files from Zenodo into METEOR's cache.

Coefficient files (~110 MB each, 82 total across all models and crops) are
hosted at https://zenodo.org/records/3592453.  This module integrates with
METEOR's CacheHandler so all GGCM data lands alongside cmip6/, pattern_scaling/,
etc. under the single METEOR cache root.
"""

import logging
from pathlib import Path

from . import data_catalog as catalog

log = logging.getLogger(__name__)

CHUNK_SIZE = 8192  # 8 KB streaming chunks


class GgcmDownloader:
    """Download and manage GGCM polynomial coefficient files.

    Parameters
    ----------
    cache_dir : str
        Directory where coefficient files will be stored.  Typically obtained
        from ``cache_handler.get_subdir('ggcm')``.
    """

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_filepath(self, crop_model, crop, variant):
        """Return the expected local path for a coefficient file."""
        return self.cache_dir / catalog.get_filename(crop_model, crop, variant)

    def ensure_files(self, crops, crop_model, variant="A0"):
        """Ensure all required coefficient files are present.

        Validates the requested combination against the catalog, identifies
        any missing files, and (after a single user prompt) downloads them
        from Zenodo with progress bars.

        Parameters
        ----------
        crops : list of str
            Crop names to check (e.g. ``['maize', 'spring_wheat']``).
        crop_model : str
            GGCM model name (e.g. ``'LPJmL'``).
        variant : str
            ``'A0'`` (no adaptation) or ``'A1'`` (with adaptation).

        Raises
        ------
        ValueError
            If a requested crop/model/variant combination is not in the catalog.
        ImportError
            If ``requests`` is not installed.
        """
        for crop in crops:
            if not catalog.is_available(crop_model, crop, variant):
                available = catalog.get_available_crops(crop_model)
                raise ValueError(
                    f"Crop '{crop}' not available for model '{crop_model}' "
                    f"with variant '{variant}'. "
                    f"Available crops: {available}"
                )

        missing = [
            crop
            for crop in crops
            if not self.get_filepath(crop_model, crop, variant).exists()
        ]

        if not missing:
            return

        try:
            import requests  # pylint: disable=import-outside-toplevel
            from tqdm import tqdm  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ImportError(
                "The 'requests' package is required to download GGCM files.\n"
                "Install it with:  pip install requests"
            ) from exc

        size_mb = len(missing) * 110
        print(
            f"\nDownloading {len(missing)} GGCM coefficient file(s) for "
            f"{crop_model} ({variant}) from Zenodo record {catalog.ZENODO_RECORD_ID}:"
        )
        for crop in missing:
            print(f"  - {catalog.get_filename(crop_model, crop, variant)}")
        print(f"  (~{size_mb} MB total  →  {self.cache_dir})")

        for crop in missing:
            url = catalog.get_download_url(crop_model, crop, variant)
            dest = self.get_filepath(crop_model, crop, variant)
            log.info("Downloading %s", dest.name)
            self._download_file(url, dest, requests, tqdm)

    def _download_file(self, url, filepath, requests, tqdm):
        """Stream a single file with resume capability and a progress bar."""
        resume_bytes = 0
        headers = {}
        if filepath.exists():
            resume_bytes = filepath.stat().st_size
            headers = {"Range": f"bytes={resume_bytes}-"}

        resp = requests.get(url, headers=headers, stream=True, timeout=60)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0)) + resume_bytes
        mode = "ab" if resume_bytes else "wb"

        with (
            open(filepath, mode) as fh,
            tqdm(
                total=total,
                initial=resume_bytes,
                unit="B",
                unit_scale=True,
                desc=filepath.name,
            ) as pbar,
        ):
            for chunk in resp.iter_content(CHUNK_SIZE):
                fh.write(chunk)
                pbar.update(len(chunk))
