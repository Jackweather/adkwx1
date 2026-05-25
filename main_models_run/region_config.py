import os
import shutil
from collections import OrderedDict
from pathlib import Path


REGION_SPECS = OrderedDict(
    [
        ("conus", {"label": "CONUS", "extent": (-130, -65, 20, 54)}),
        ("northeast", {"label": "Northeast", "extent": (-84, -66, 35, 48)}),
        ("southeast", {"label": "Southeast", "extent": (-95, -74, 24, 38)}),
        ("midwest", {"label": "Midwest", "extent": (-104, -80, 36, 50)}),
        ("southern_plains", {"label": "Southern Plains", "extent": (-107, -90, 28, 40)}),
        ("northern_plains", {"label": "Northern Plains", "extent": (-107, -92, 40, 50)}),
        ("rockies", {"label": "Rockies", "extent": (-116, -101, 35, 49)}),
        ("southwest", {"label": "Southwest", "extent": (-125, -108, 31, 42)}),
        ("northwest", {"label": "Northwest", "extent": (-126, -110, 41, 50)}),
    ]
)

DEFAULT_REGION = "conus"
DEFAULT_ACTIVE_REGIONS = ("conus", "northeast", "southeast", "midwest")
REGION_ENV_VAR = "ADKWX_REGIONS"


def _normalize_region_names(region_names):
    normalized = []
    for region_name in region_names:
        key = region_name.strip().lower().replace("-", "_").replace(" ", "_")
        if key in REGION_SPECS and key not in normalized:
            normalized.append(key)

    if DEFAULT_REGION not in normalized:
        normalized.insert(0, DEFAULT_REGION)
    return tuple(normalized)


def get_active_region_names():
    raw_value = os.getenv(REGION_ENV_VAR, "")
    if raw_value.strip():
        return _normalize_region_names(raw_value.split(","))
    return _normalize_region_names(DEFAULT_ACTIVE_REGIONS)


ACTIVE_REGION_NAMES = get_active_region_names()
REGION_LABELS = {name: spec["label"] for name, spec in REGION_SPECS.items()}
CONUS_EXTENT = REGION_SPECS[DEFAULT_REGION]["extent"]


def get_region_extent(region_name):
    return REGION_SPECS[region_name]["extent"]


def prepare_region_png_dirs(base_png_dir: Path, region_names=None):
    active_region_names = tuple(region_names or ACTIVE_REGION_NAMES)
    base_png_dir.mkdir(parents=True, exist_ok=True)

    for existing_png in base_png_dir.glob("*.png"):
        existing_png.unlink()

    for child in base_png_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)

    for region_name in active_region_names:
        (base_png_dir / region_name).mkdir(parents=True, exist_ok=True)
