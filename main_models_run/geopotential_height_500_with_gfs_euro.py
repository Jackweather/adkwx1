import bz2
import gc
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
import requests
from scipy.spatial import cKDTree
import xarray as xr
from ecmwf.opendata import Client
from region_config import ACTIVE_REGION_NAMES, CONUS_EXTENT, REGION_LABELS, get_region_extent, prepare_region_png_dirs


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = Path("/var/data") if Path("/var/data").exists() else SCRIPT_DIR
BASE_DIR = OUTPUT_ROOT / "EURO_GFS_GH500_OUTPUT"
GRIB_DIR = BASE_DIR / "grib"
PNG_DIR = BASE_DIR / "png"
LOG_FILE = BASE_DIR / "errors_euro_gfs_gh500.txt"

MAX_DOWNLOAD_RETRIES = 3
LOCAL_TZ = ZoneInfo("America/New_York")
GFS_FILTER_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
ICON_BASE_URL = "https://opendata.dwd.de/weather/nwp/icon/grib"
GDPS_BASE_URL = "https://dd.weather.gc.ca"
PLOT_EXTENT = CONUS_EXTENT
ICON_MARGIN_DEGREES = 3.0
GRAVITY = 9.80665
FILL_INTERVAL_DAM = 3
CONTOUR_INTERVAL_DAM = 6
HEIGHT_CMAP = plt.get_cmap("Spectral_r")


def get_run_configuration(run_hour):
    # Always produce blended PNGs through FH240 (primary and non-primary).
    # ICON should only be used through FH180; beyond that it is ignored.
    run_hour_int = int(run_hour)
    is_primary_cycle = run_hour_int in {0, 12}
    return {
        "blend_max_hour": 240,
        "gfs_max_hour": 384,
        "ecmwf_max_hour": 360 if is_primary_cycle else 144,
        "gdps_max_hour": 240,
        # ICON cutoff at FH180 regardless of cycle; use ICON only up to 180h
        "icon_max_hour": 180,
    }

def get_forecast_steps(run_hour):
    run_config = get_run_configuration(run_hour)
    return list(range(6, run_config["blend_max_hour"] + 1, 6))


def prepare_output_dirs():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    if GRIB_DIR.exists():
        shutil.rmtree(GRIB_DIR)
    GRIB_DIR.mkdir(parents=True, exist_ok=True)
    prepare_region_png_dirs(PNG_DIR, ACTIVE_REGION_NAMES)

    if LOG_FILE.exists():
        LOG_FILE.unlink()


def log_error(step_str, context, error):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] FH{step_str} | {context} | {error}\n")


def find_latest_gfs_run():
    now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    candidate = now_utc.replace(hour=(now_utc.hour // 6) * 6)

    for offset in range(16):
        test_time = candidate - timedelta(hours=6 * offset)
        run_date = test_time.strftime("%Y%m%d")
        run_hour = f"{test_time.hour:02d}"
        test_url = (
            "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/"
            f"gfs.{run_date}/{run_hour}/atmos/gfs.t{run_hour}z.pgrb2.0p25.f000"
        )
        try:
            response = requests.head(test_url, timeout=20, allow_redirects=True)
        except requests.RequestException:
            continue
        if response.status_code == 200:
            print(f"Using latest GFS run: {run_date} {run_hour}Z")
            return run_date, run_hour

    raise RuntimeError("Unable to locate a recent GFS run on NOMADS.")


def build_gfs_url(run_date, run_hour, step_str, variable_query):
    query = (
        f"file=gfs.t{run_hour}z.pgrb2.0p25.f{step_str}"
        f"&{variable_query}"
        "&subregion="
        "&leftlon=220&rightlon=300&toplat=55&bottomlat=20"
        f"&dir={quote(f'/gfs.{run_date}/{run_hour}/atmos')}"
    )
    return f"{GFS_FILTER_URL}?{query}"


def find_latest_icon_run():
    now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    candidate = now_utc.replace(hour=(now_utc.hour // 6) * 6)

    for offset in range(16):
        test_time = candidate - timedelta(hours=6 * offset)
        run_date = test_time.strftime("%Y%m%d")
        run_hour = f"{test_time.hour:02d}"
        test_url = build_icon_field_url(run_date, run_hour, "fi", "000", "500", "FI")
        try:
            response = requests.head(test_url, timeout=20, allow_redirects=True)
        except requests.RequestException:
            continue
        if response.status_code == 200:
            print(f"Using latest ICON run: {run_date} {run_hour}Z")
            return run_date, run_hour

    raise RuntimeError("Unable to locate a recent ICON run on DWD open data.")


def build_icon_field_url(run_date, run_hour, subdir, step_str, level_code, suffix):
    return (
        f"{ICON_BASE_URL}/{run_hour}/{subdir}/"
        f"icon_global_icosahedral_pressure-level_{run_date}{run_hour}_{step_str}_{level_code}_{suffix}.grib2.bz2"
    )


def build_icon_coord_url(run_date, run_hour, subdir, suffix):
    return (
        f"{ICON_BASE_URL}/{run_hour}/{subdir}/"
        f"icon_global_icosahedral_time-invariant_{run_date}{run_hour}_{suffix}.grib2.bz2"
    )


def find_latest_gdps_run():
    now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    candidate = now_utc.replace(hour=(now_utc.hour // 6) * 6)

    for offset in range(16):
        test_time = candidate - timedelta(hours=6 * offset)
        run_date = test_time.strftime("%Y%m%d")
        run_hour = f"{test_time.hour:02d}"
        test_url = build_gdps_field_url(run_date, run_hour, "000", "GeopotentialHeight_IsbL-0500")
        try:
            response = requests.head(test_url, timeout=20, allow_redirects=True)
        except requests.RequestException:
            continue
        if response.status_code == 200:
            print(f"Using latest GDPS run: {run_date} {run_hour}Z")
            return run_date, run_hour

    raise RuntimeError("Unable to locate a recent GDPS run on Environment Canada open data.")


def build_gdps_field_url(run_date, run_hour, step_str, field_name):
    filename = f"{run_date}T{run_hour}Z_MSC_GDPS_{field_name}_LatLon0.15_PT{step_str}H.grib2"
    return f"{GDPS_BASE_URL}/{run_date}/WXO-DD/model_gdps/15km/{run_hour}/{step_str}/{filename}"


def download_file(url, destination, label):
    if destination.exists():
        return

    last_error = None
    for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
        print(f"Downloading {label} (attempt {attempt}/{MAX_DOWNLOAD_RETRIES})")
        try:
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            with destination.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        handle.write(chunk)
            return
        except requests.RequestException as exc:
            last_error = exc
            destination.unlink(missing_ok=True)
            time.sleep(2)

    raise RuntimeError(f"Failed to download {label}: {last_error}")


def download_bz2_file(url, destination, label):
    if destination.exists():
        return

    last_error = None
    for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
        print(f"Downloading {label} (attempt {attempt}/{MAX_DOWNLOAD_RETRIES})")
        try:
            response = requests.get(url, timeout=180)
            response.raise_for_status()
            decompressed = bz2.decompress(response.content)
            destination.write_bytes(decompressed)
            return
        except Exception as exc:
            last_error = exc
            destination.unlink(missing_ok=True)
            time.sleep(2)

    raise RuntimeError(f"Failed to download {label}: {last_error}")


def normalize_longitudes(lons):
    lons = np.asarray(lons, dtype=np.float64)
    return np.where(lons > 180.0, lons - 360.0, lons)


def ensure_ascending(values, lats, lons):
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    values = np.asarray(values, dtype=np.float32)

    if lats[0] > lats[-1]:
        lats = lats[::-1]
        values = np.flip(values, axis=0)

    sort_idx = np.argsort(lons)
    lons = lons[sort_idx]
    values = values[:, sort_idx]
    return values, lats, lons


def as_data_array(values, lats, lons, name):
    fixed_values, fixed_lats, fixed_lons = ensure_ascending(values, lats, normalize_longitudes(lons))
    return xr.DataArray(
        fixed_values,
        coords={"latitude": fixed_lats, "longitude": fixed_lons},
        dims=("latitude", "longitude"),
        name=name,
    )


def to_height_dam(values, units=""):
    height = np.asarray(values, dtype=np.float32)
    normalized_units = str(units or "").strip().lower()

    if "m**2" in normalized_units or "m2" in normalized_units or "s**-2" in normalized_units or np.nanmean(np.abs(height)) > 20000:
        height = height / GRAVITY

    if np.nanmean(np.abs(height)) > 1500:
        height = height / 10.0

    return height


def open_scalar_pressure_level(path, level_hpa, variable_names=None, output_name="field"):
    dataset = xr.open_dataset(
        path,
        engine="cfgrib",
        filter_by_keys={"typeOfLevel": "isobaricInhPa", "level": level_hpa},
    )
    try:
        candidates = list(variable_names or [])
        if not candidates:
            candidates = list(dataset.data_vars)

        selected_name = None
        for candidate in candidates:
            if candidate in dataset.data_vars:
                selected_name = candidate
                break
        if selected_name is None:
            selected_name = next(iter(dataset.data_vars))

        field = dataset[selected_name].squeeze()
        values = to_height_dam(field.values, field.attrs.get("units", ""))
        return as_data_array(values, dataset["latitude"].values, dataset["longitude"].values, output_name)
    finally:
        dataset.close()


def open_gfs_gh500(path):
    return open_scalar_pressure_level(path, 500, output_name="gfs_gh500_dam")


def open_ecmwf_gh500(path):
    return open_scalar_pressure_level(path, 500, ["gh", "z"], "ecmwf_gh500_dam")


def open_gdps_gh500(path):
    dataset = xr.open_dataset(path, engine="cfgrib")
    try:
        variable_name = next(iter(dataset.data_vars))
        field = dataset[variable_name].squeeze()
        values = to_height_dam(field.values, field.attrs.get("units", ""))
        return as_data_array(values, dataset["latitude"].values, dataset["longitude"].values, "gdps_gh500_dam")
    finally:
        dataset.close()


def open_icon_scalar_field(path, variable_name):
    dataset = xr.open_dataset(path, engine="cfgrib")
    try:
        candidate_names = [variable_name] if isinstance(variable_name, str) else list(variable_name)
        for candidate_name in candidate_names:
            if candidate_name in dataset.data_vars:
                field = dataset[candidate_name].squeeze()
                return to_height_dam(field.values, field.attrs.get("units", ""))

        if len(dataset.data_vars) == 1:
            only_name = next(iter(dataset.data_vars))
            field = dataset[only_name].squeeze()
            return to_height_dam(field.values, field.attrs.get("units", ""))

        raise KeyError(
            f"No ICON variable matched {candidate_names}; available variables: {sorted(dataset.data_vars)}"
        )
    finally:
        dataset.close()


def open_icon_coord_field(path, variable_name):
    dataset = xr.open_dataset(path, engine="cfgrib")
    try:
        return np.asarray(dataset[variable_name].squeeze().values, dtype=np.float64)
    finally:
        dataset.close()


def build_icon_indexer(icon_lats, icon_lons, gfs_field):
    lon_min, lon_max, lat_min, lat_max = PLOT_EXTENT
    region_mask = (
        (icon_lats >= lat_min - ICON_MARGIN_DEGREES)
        & (icon_lats <= lat_max + ICON_MARGIN_DEGREES)
        & (icon_lons >= lon_min - ICON_MARGIN_DEGREES)
        & (icon_lons <= lon_max + ICON_MARGIN_DEGREES)
    )

    selected_indices = np.flatnonzero(region_mask)
    source_points = np.column_stack((icon_lats[selected_indices], icon_lons[selected_indices]))
    lat2d, lon2d = np.meshgrid(gfs_field.latitude.values, gfs_field.longitude.values, indexing="ij")
    target_points = np.column_stack((lat2d.ravel(), lon2d.ravel()))
    tree = cKDTree(source_points)
    _, nearest_indices = tree.query(target_points)
    return selected_indices[nearest_indices], gfs_field.shape


def remap_icon_to_gfs(icon_values, icon_indexer, gfs_field, name):
    source_indices, target_shape = icon_indexer
    remapped = np.asarray(icon_values[source_indices], dtype=np.float32).reshape(target_shape)
    return xr.DataArray(remapped, coords=gfs_field.coords, dims=gfs_field.dims, name=name)


def regrid_to_gfs(source_field, gfs_field):
    return source_field.interp(
        latitude=gfs_field.latitude.values,
        longitude=gfs_field.longitude.values,
        method="linear",
    )


def format_valid_time(run_datetime, forecast_hour):
    valid_utc = run_datetime + timedelta(hours=forecast_hour)
    valid_local = valid_utc.astimezone(LOCAL_TZ)
    local_time = valid_local.strftime("%I %p").lstrip("0")
    return valid_utc, local_time, valid_local.strftime("%A")


def get_model_weights(gfs_run, icon_run, ecmwf_run, gdps_run):
    return {
        "gfs": 1.0,
        "icon": 1.0,
        "ecmwf": 1.0,
        "gdps": 1.0,
    }


def weighted_average_fields(weighted_fields, name):
    total_weight = sum(weight for _, weight in weighted_fields)
    if total_weight <= 0:
        raise ValueError(f"No weighted fields available for {name}.")

    template = weighted_fields[0][0]
    weighted_sum = np.zeros_like(template.values, dtype=np.float32)
    for field, weight in weighted_fields:
        weighted_sum += field.values.astype(np.float32) * weight

    return xr.DataArray(
        weighted_sum / total_weight,
        coords=template.coords,
        dims=template.dims,
        name=name,
    )


def get_height_levels(height_field):
    min_value = float(np.nanmin(height_field.values))
    max_value = float(np.nanmax(height_field.values))
    contour_start = int(np.floor(min_value / CONTOUR_INTERVAL_DAM) * CONTOUR_INTERVAL_DAM)
    contour_end = int(np.ceil(max_value / CONTOUR_INTERVAL_DAM) * CONTOUR_INTERVAL_DAM)
    fill_start = int(np.floor(min_value / FILL_INTERVAL_DAM) * FILL_INTERVAL_DAM)
    fill_end = int(np.ceil(max_value / FILL_INTERVAL_DAM) * FILL_INTERVAL_DAM)

    contour_levels = np.arange(contour_start, contour_end + CONTOUR_INTERVAL_DAM, CONTOUR_INTERVAL_DAM)
    fill_levels = np.arange(fill_start, fill_end + FILL_INTERVAL_DAM, FILL_INTERVAL_DAM)

    if contour_levels.size < 2:
        contour_levels = np.array([contour_start - CONTOUR_INTERVAL_DAM, contour_end + CONTOUR_INTERVAL_DAM], dtype=np.int32)
    if fill_levels.size < 2:
        fill_levels = np.array([fill_start - FILL_INTERVAL_DAM, fill_end + FILL_INTERVAL_DAM], dtype=np.int32)

    return fill_levels, contour_levels


def plot_average_fields(avg_gh500, available_runs, forecast_hour, step_str):
    gfs_run = available_runs[0][1]
    valid_utc, local_time, local_day = format_valid_time(gfs_run, forecast_hour)
    lon2d, lat2d = np.meshgrid(avg_gh500.longitude.values, avg_gh500.latitude.values)
    fill_levels, contour_levels = get_height_levels(avg_gh500)

    model_text = " + ".join(name for name, _ in available_runs)
    run_text = " | ".join(f"{name} {run_time:%HZ}" for name, run_time in available_runs)
    weight_text = "Equal weights of available models"
    norm = BoundaryNorm(fill_levels, HEIGHT_CMAP.N, clip=False)

    for region_name in ACTIVE_REGION_NAMES:
        region_extent = get_region_extent(region_name)

        fig = plt.figure(figsize=(10, 7), dpi=300, facecolor="white")
        ax = fig.add_axes([0.06, 0.18, 0.88, 0.68], projection=ccrs.PlateCarree())
        ax.set_extent(region_extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.OCEAN, facecolor="white")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.3)
        ax.add_feature(cfeature.RIVERS, linewidth=0.4, edgecolor="blue")
        ax.add_feature(cfeature.LAKES, facecolor="lightblue", edgecolor="blue", linewidth=0.3, zorder=0)

        filled = ax.contourf(
            lon2d,
            lat2d,
            avg_gh500.values,
            levels=fill_levels,
            cmap=HEIGHT_CMAP,
            norm=norm,
            extend="both",
            transform=ccrs.PlateCarree(),
            zorder=50,
        )

        contours = ax.contour(
            lon2d,
            lat2d,
            avg_gh500.values,
            levels=contour_levels,
            colors="black",
            linewidths=0.65,
            transform=ccrs.PlateCarree(),
            zorder=100,
        )
        ax.clabel(contours, fmt="%d", fontsize=6)

        ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=200)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=200)
        ax.add_feature(cfeature.STATES, linewidth=0.3, zorder=200)
        ax.add_feature(cfeature.RIVERS, linewidth=0.4, edgecolor="blue", zorder=200)

        cbar_ax = fig.add_axes([0.12, 0.08, 0.76, 0.018])
        colorbar = fig.colorbar(filled, cax=cbar_ax, orientation="horizontal")
        colorbar.set_label("Avg 500 hPa geopotential height (dam)", fontsize=7)
        colorbar.ax.tick_params(labelsize=6)

        ax.set_title(
            (
                f"{model_text} 500 hPa geopotential height | {REGION_LABELS[region_name]} | FH{step_str} | "
                f"Valid {valid_utc:%Y-%m-%d %HZ} ({local_time} ET, {local_day})\n"
                f"Runs: {run_text}\n"
                f"{weight_text}"
            ),
            fontsize=7,
            pad=6,
        )

        output_path = PNG_DIR / region_name / f"euro_gfs_gh500_avg_{step_str}.png"
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved PNG: {output_path}")


def main():
    prepare_output_dirs()

    gfs_run_date, gfs_run_hour = find_latest_gfs_run()
    gfs_run = datetime.strptime(f"{gfs_run_date} {gfs_run_hour}", "%Y%m%d %H").replace(tzinfo=timezone.utc)
    run_config = get_run_configuration(gfs_run_hour)
    forecast_steps = get_forecast_steps(gfs_run_hour)
    print(f"Processing forecast hours through FH{forecast_steps[-1]:03d} for GFS {gfs_run_hour}Z")
    print(
        "Blend limits | "
        f"GFS: FH{run_config['gfs_max_hour']:03d} | "
        f"ECMWF: FH{run_config['ecmwf_max_hour']:03d} | "
        f"ICON: FH{run_config['icon_max_hour']:03d} | "
        f"GDPS: FH{run_config['gdps_max_hour']:03d}"
    )
    icon_run_date, icon_run_hour = find_latest_icon_run()
    icon_run = datetime.strptime(f"{icon_run_date} {icon_run_hour}", "%Y%m%d %H").replace(tzinfo=timezone.utc)
    gdps_run_date, gdps_run_hour = find_latest_gdps_run()
    gdps_run = datetime.strptime(f"{gdps_run_date} {gdps_run_hour}", "%Y%m%d %H").replace(tzinfo=timezone.utc)

    ecmwf_client = Client(source="azure", model="ifs", resol="0p25")
    ecmwf_latest = ecmwf_client.latest(type="fc", step=6, param="gh", levtype="pl", levelist=500)
    ecmwf_run = ecmwf_latest.replace(tzinfo=timezone.utc) if ecmwf_latest.tzinfo is None else ecmwf_latest.astimezone(timezone.utc)
    print(f"Using latest ECMWF run: {ecmwf_run:%Y-%m-%d %HZ}")
    model_weights = get_model_weights(gfs_run, icon_run, ecmwf_run, gdps_run)

    icon_clat_path = GRIB_DIR / f"icon_clat_{icon_run_date}{icon_run_hour}.grib2"
    icon_clon_path = GRIB_DIR / f"icon_clon_{icon_run_date}{icon_run_hour}.grib2"
    download_bz2_file(
        build_icon_coord_url(icon_run_date, icon_run_hour, "clat", "CLAT"),
        icon_clat_path,
        "ICON CLAT",
    )
    download_bz2_file(
        build_icon_coord_url(icon_run_date, icon_run_hour, "clon", "CLON"),
        icon_clon_path,
        "ICON CLON",
    )
    icon_lats = open_icon_coord_field(icon_clat_path, "tlat")
    icon_lons = open_icon_coord_field(icon_clon_path, "tlon")
    icon_indexer = None

    for forecast_hour in forecast_steps:
        step_str = f"{forecast_hour:03d}"
        ecmwf_available = forecast_hour <= run_config["ecmwf_max_hour"]
        icon_available = forecast_hour <= run_config["icon_max_hour"]
        gdps_available = forecast_hour <= run_config["gdps_max_hour"]

        gfs_gh500_path = GRIB_DIR / f"gfs_gh500_f{step_str}.grib2"
        ecmwf_gh500_path = GRIB_DIR / f"ecmwf_gh500_f{step_str}.grib2"
        icon_gh500_path = GRIB_DIR / f"icon_gh500_f{step_str}.grib2"
        gdps_gh500_path = GRIB_DIR / f"gdps_gh500_f{step_str}.grib2"

        try:
            download_file(
                build_gfs_url(gfs_run_date, gfs_run_hour, step_str, "var_HGT=on&lev_500_mb=on"),
                gfs_gh500_path,
                f"GFS 500 hPa height FH{step_str}",
            )

            if ecmwf_available:
                ecmwf_client.retrieve(
                    date=int(ecmwf_run.strftime("%Y%m%d")),
                    time=ecmwf_run.hour,
                    type="fc",
                    step=forecast_hour,
                    levtype="pl",
                    levelist=500,
                    param=["gh"],
                    target=str(ecmwf_gh500_path),
                )

            if icon_available:
                download_bz2_file(
                    build_icon_field_url(icon_run_date, icon_run_hour, "fi", step_str, "500", "FI"),
                    icon_gh500_path,
                    f"ICON 500 hPa height FH{step_str}",
                )

            if gdps_available:
                download_file(
                    build_gdps_field_url(gdps_run_date, gdps_run_hour, step_str, "GeopotentialHeight_IsbL-0500"),
                    gdps_gh500_path,
                    f"GDPS 500 hPa height FH{step_str}",
                )

            gfs_gh500 = open_gfs_gh500(gfs_gh500_path)

            if ecmwf_available:
                ecmwf_gh500 = regrid_to_gfs(open_ecmwf_gh500(ecmwf_gh500_path), gfs_gh500)
            else:
                ecmwf_gh500 = None

            if gdps_available:
                gdps_gh500 = regrid_to_gfs(open_gdps_gh500(gdps_gh500_path), gfs_gh500)
            else:
                gdps_gh500 = None

            if icon_indexer is None:
                icon_indexer = build_icon_indexer(icon_lats, icon_lons, gfs_gh500)

            if icon_available:
                icon_gh500 = remap_icon_to_gfs(
                    open_icon_scalar_field(icon_gh500_path, ["fi", "gh", "z"]),
                    icon_indexer,
                    gfs_gh500,
                    "icon_gh500_dam",
                )
            else:
                icon_gh500 = None

            available_runs = [("GFS", gfs_run)]
            gh500_fields = [(gfs_gh500, model_weights["gfs"])]

            if ecmwf_available:
                available_runs.append(("ECMWF", ecmwf_run))
                gh500_fields.append((ecmwf_gh500, model_weights["ecmwf"]))

            if icon_available:
                available_runs.append(("ICON", icon_run))
                gh500_fields.append((icon_gh500, model_weights["icon"]))

            if gdps_available:
                available_runs.append(("GDPS", gdps_run))
                gh500_fields.append((gdps_gh500, model_weights["gdps"]))

            avg_gh500 = weighted_average_fields(gh500_fields, "avg_gh500")
            plot_average_fields(avg_gh500, available_runs, forecast_hour, step_str)

        except Exception as exc:
            log_error(step_str, "Failed to process forecast hour", exc)
            print(f"Skipping FH{step_str}: {exc}")
        finally:
            gfs_gh500_path.unlink(missing_ok=True)
            ecmwf_gh500_path.unlink(missing_ok=True)
            icon_gh500_path.unlink(missing_ok=True)
            gdps_gh500_path.unlink(missing_ok=True)
            gc.collect()

    print(f"Finished generating GFS + ECMWF + ICON + GDPS 500 hPa geopotential-height PNGs in {PNG_DIR}")


if __name__ == "__main__":
    main()
