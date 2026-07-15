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
import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
import numpy as np
import requests
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
from scipy.spatial import cKDTree
import xarray as xr
from ecmwf.opendata import Client
from region_config import ACTIVE_REGION_NAMES, CONUS_EXTENT, REGION_LABELS, get_region_extent, prepare_region_png_dirs


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = Path("/var/data") if Path("/var/data").exists() else SCRIPT_DIR
BASE_DIR = OUTPUT_ROOT / "EURO_GFS_TOTAL_CLOUD_OUTPUT"
GRIB_DIR = BASE_DIR / "grib"
PNG_DIR = BASE_DIR / "png"
LOG_FILE = BASE_DIR / "errors_euro_gfs_total_cloud.txt"

MAX_DOWNLOAD_RETRIES = 3
LOCAL_TZ = ZoneInfo("America/New_York")
GFS_FILTER_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
ICON_BASE_URL = "https://opendata.dwd.de/weather/nwp/icon/grib"
GDPS_BASE_URL = "https://dd.weather.gc.ca"
PLOT_EXTENT = CONUS_EXTENT
ICON_MARGIN_DEGREES = 3.0

CLOUD_LEVELS = np.arange(0, 110, 10)
CLOUD_CMAP = LinearSegmentedColormap.from_list(
    "cloud_cover",
    ["#0b1f4d", "#244f8f", "#4f7fc1", "#88addd", "#c5d9f0", "#ffffff"],
    N=256,
)
CLOUD_NORM = BoundaryNorm(CLOUD_LEVELS, CLOUD_CMAP.N, clip=True)
MSLP_LEVELS = np.arange(960, 1060, 4)


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


def log_optional_model_failure(step_str, model_name, error):
    log_error(step_str, f"{model_name} unavailable for this hour; continuing with remaining models", error)


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


def log_optional_model_failure(step_str, model_name, error):
    log_error(step_str, f"{model_name} unavailable for this hour; continuing with remaining models", error)


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
        test_url = build_icon_field_url(run_date, run_hour, "pmsl", "000", "PMSL")
        try:
            response = requests.head(test_url, timeout=20, allow_redirects=True)
        except requests.RequestException:
            continue
        if response.status_code == 200:
            print(f"Using latest ICON run: {run_date} {run_hour}Z")
            return run_date, run_hour

    raise RuntimeError("Unable to locate a recent ICON run on DWD open data.")


def build_icon_field_url(run_date, run_hour, subdir, step_str, suffix):
    return (
        f"{ICON_BASE_URL}/{run_hour}/{subdir}/"
        f"icon_global_icosahedral_single-level_{run_date}{run_hour}_{step_str}_{suffix}.grib2.bz2"
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
        test_url = build_gdps_field_url(run_date, run_hour, "000", "Pressure_MSL")
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


def normalize_cloud_cover_percent(field):
    max_value = float(np.nanmax(field.values)) if field.size else 0.0
    if max_value <= 1.01:
        return field * 100.0
    return field.clip(min=0.0, max=100.0)


def open_gfs_cloud_cover(path):
    last_error = None
    for step_type in ("instant", "avg"):
        try:
            dataset = xr.open_dataset(path, engine="cfgrib", filter_by_keys={"stepType": step_type})
        except Exception as exc:
            last_error = exc
            continue

        try:
            variable_name = next((name for name in ("tcc", "tcdc") if name in dataset.data_vars), next(iter(dataset.data_vars)))
            field = as_data_array(
                dataset[variable_name].squeeze().values,
                dataset["latitude"].values,
                dataset["longitude"].values,
                "gfs_cloud_cover",
            )
            return normalize_cloud_cover_percent(field)
        finally:
            dataset.close()

    raise RuntimeError(f"Unable to open GFS total cloud cover from {path.name}: {last_error}")


def open_gfs_mslp(path):
    dataset = xr.open_dataset(path, engine="cfgrib", filter_by_keys={"typeOfLevel": "meanSea"})
    try:
        variable_name = "mslet" if "mslet" in dataset.data_vars else next(iter(dataset.data_vars))
        field = dataset[variable_name].squeeze().values / 100.0
        return as_data_array(field, dataset["latitude"].values, dataset["longitude"].values, "gfs_mslp")
    finally:
        dataset.close()


def open_ecmwf_fields(path):
    mean_sea_dataset = xr.open_dataset(path, engine="cfgrib", filter_by_keys={"typeOfLevel": "meanSea"})
    surface_dataset = xr.open_dataset(path, engine="cfgrib", filter_by_keys={"typeOfLevel": "surface"})
    try:
        msl = as_data_array(
            mean_sea_dataset["msl"].squeeze().values / 100.0,
            mean_sea_dataset["latitude"].values,
            mean_sea_dataset["longitude"].values,
            "ecmwf_mslp",
        )
        cloud = as_data_array(
            surface_dataset["tcc"].squeeze().values,
            surface_dataset["latitude"].values,
            surface_dataset["longitude"].values,
            "ecmwf_cloud_cover",
        )
        return msl, normalize_cloud_cover_percent(cloud)
    finally:
        mean_sea_dataset.close()
        surface_dataset.close()


def open_gdps_scalar_field(path, name, scale=1.0):
    dataset = xr.open_dataset(path, engine="cfgrib")
    try:
        variable_name = name if name in dataset.data_vars else next(iter(dataset.data_vars))
        field = dataset[variable_name].squeeze().values * scale
        return as_data_array(field, dataset["latitude"].values, dataset["longitude"].values, f"gdps_{variable_name}")
    finally:
        dataset.close()


def open_gdps_mslp(path):
    return open_gdps_scalar_field(path, "prmsl", scale=0.01)


def open_gdps_cloud_cover(path):
    field = open_gdps_scalar_field(path, "tcc")
    return normalize_cloud_cover_percent(field)


def open_icon_coord_field(path, variable_name):
    dataset = xr.open_dataset(path, engine="cfgrib")
    try:
        return np.asarray(dataset[variable_name].squeeze().values, dtype=np.float64)
    finally:
        dataset.close()


def open_icon_scalar_field(path, variable_name, scale=1.0):
    dataset = xr.open_dataset(path, engine="cfgrib")
    try:
        candidate_names = [variable_name] if isinstance(variable_name, str) else list(variable_name)
        for candidate_name in candidate_names:
            if candidate_name in dataset.data_vars:
                return np.asarray(dataset[candidate_name].squeeze().values, dtype=np.float32) * scale

        if len(dataset.data_vars) == 1:
            only_name = next(iter(dataset.data_vars))
            return np.asarray(dataset[only_name].squeeze().values, dtype=np.float32) * scale

        raise KeyError(
            f"No ICON variable matched {candidate_names}; available variables: {sorted(dataset.data_vars)}"
        )
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


def plot_lows_highs(ax, lon2d, lat2d, data, extent, min_distance=5, edge_buffer=2):
    smoothed_data = gaussian_filter(data, sigma=3)
    local_min = minimum_filter(smoothed_data, size=10) == smoothed_data
    local_max = maximum_filter(smoothed_data, size=10) == smoothed_data

    points = []
    lon_min, lon_max, lat_min, lat_max = extent
    mask = local_min | local_max

    for lon, lat, value, is_low in zip(
        lon2d[mask].flatten(),
        lat2d[mask].flatten(),
        data[mask].flatten(),
        local_min[mask].flatten(),
    ):
        if not (lon_min + edge_buffer <= lon <= lon_max - edge_buffer and lat_min + edge_buffer <= lat <= lat_max - edge_buffer):
            continue
        points.append((lon, lat, value, "L" if is_low else "H"))

    lows = sorted((point for point in points if point[3] == "L"), key=lambda item: item[2])
    highs = sorted((point for point in points if point[3] == "H"), key=lambda item: -item[2])

    def filter_points(input_points):
        filtered = []
        for point in input_points:
            if all(np.hypot(point[0] - other[0], point[1] - other[1]) > min_distance for other in filtered):
                filtered.append(point)
        return filtered

    def filter_same_contour(input_points):
        filtered = []
        contour_interval = MSLP_LEVELS[1] - MSLP_LEVELS[0]
        for point in input_points:
            if all(abs(point[2] - other[2]) > contour_interval for other in filtered):
                filtered.append(point)
        return filtered

    lows = filter_same_contour(filter_points(lows))[:2]
    highs = filter_same_contour(filter_points(highs))[:2]

    for lon, lat, value, label in lows + highs:
        color = "red" if label == "L" else "blue"
        main_text = ax.text(
            lon,
            lat,
            label,
            color=color,
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            transform=ccrs.PlateCarree(),
            zorder=400,
            clip_on=False,
        )
        main_text.set_path_effects([mpe.withStroke(linewidth=2.0, foreground="white")])

        value_text = ax.text(
            lon,
            lat - 1,
            f"{value:.0f}",
            color=color,
            fontsize=6,
            fontweight="bold",
            ha="center",
            va="center",
            transform=ccrs.PlateCarree(),
            zorder=400,
            clip_on=False,
        )
        value_text.set_path_effects([mpe.withStroke(linewidth=2.0, foreground="white")])


def plot_average_fields(avg_cloud_cover, avg_mslp, available_runs, forecast_hour, step_str):
    gfs_run = available_runs[0][1]
    valid_utc, local_time, local_day = format_valid_time(gfs_run, forecast_hour)
    lon2d, lat2d = np.meshgrid(avg_cloud_cover.longitude.values, avg_cloud_cover.latitude.values)

    model_text = " + ".join(name for name, _ in available_runs)
    run_text = " | ".join(f"{name} {run_time:%HZ}" for name, run_time in available_runs)
    weight_text = "Equal weights of available models"

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

        cloud_mesh = ax.contourf(
            lon2d,
            lat2d,
            avg_cloud_cover.values,
            levels=CLOUD_LEVELS,
            cmap=CLOUD_CMAP,
            norm=CLOUD_NORM,
            extend="neither",
            transform=ccrs.PlateCarree(),
            alpha=0.94,
            zorder=50,
        )

        mslp_contours = ax.contour(
            lon2d,
            lat2d,
            avg_mslp.values,
            levels=MSLP_LEVELS,
            colors="black",
            linewidths=0.5,
            transform=ccrs.PlateCarree(),
        )
        ax.clabel(mslp_contours, fmt="%d", fontsize=6)

        ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=200)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=200)
        ax.add_feature(cfeature.STATES, linewidth=0.3, zorder=200)
        ax.add_feature(cfeature.RIVERS, linewidth=0.4, edgecolor="blue", zorder=200)

        top_mslp_contours = ax.contour(
            lon2d,
            lat2d,
            avg_mslp.values,
            levels=MSLP_LEVELS,
            colors="black",
            linewidths=0.6,
            transform=ccrs.PlateCarree(),
            zorder=210,
        )
        ax.clabel(top_mslp_contours, fmt="%d", fontsize=6)
        plot_lows_highs(ax, lon2d, lat2d, avg_mslp.values, extent=region_extent)

        cbar_left = 0.12
        cbar_bottom = 0.08
        cbar_width = 0.76
        cbar_height = 0.018
        cbar_ax_cloud = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
        cloud_bar = fig.colorbar(cloud_mesh, cax=cbar_ax_cloud, orientation="horizontal")
        cloud_bar.set_ticks(CLOUD_LEVELS)
        cloud_bar.set_label("Total cloud cover (%)", fontsize=7)
        cloud_bar.ax.tick_params(labelsize=6)

        ax.set_title(
            (
                f"{model_text} total cloud cover & MSLP | {REGION_LABELS[region_name]} | FH{step_str} | "
                f"Valid {valid_utc:%Y-%m-%d %HZ} ({local_time} ET, {local_day})\n"
                f"Runs: {run_text}\n"
                f"{weight_text}"
            ),
            fontsize=7,
            pad=6,
        )

        output_path = PNG_DIR / region_name / f"euro_gfs_total_cloud_mslp_avg_{step_str}.png"
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
    ecmwf_latest = ecmwf_client.latest(type="fc", param="msl", step=6)
    ecmwf_run = ecmwf_latest.replace(tzinfo=timezone.utc) if ecmwf_latest.tzinfo is None else ecmwf_latest.astimezone(timezone.utc)
    print(f"Using latest EURO run: {ecmwf_run:%Y-%m-%d %HZ}")
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
        ecmwf_enabled = forecast_hour <= run_config["ecmwf_max_hour"]
        icon_enabled = forecast_hour <= run_config["icon_max_hour"]
        gdps_enabled = forecast_hour <= run_config["gdps_max_hour"]

        gfs_cloud_path = GRIB_DIR / f"gfs_cloud_f{step_str}.grib2"
        gfs_mslp_path = GRIB_DIR / f"gfs_mslp_f{step_str}.grib2"
        ecmwf_path = GRIB_DIR / f"ecmwf_cloud_fields_f{step_str}.grib2"
        icon_pmsl_path = GRIB_DIR / f"icon_pmsl_f{step_str}.grib2"
        icon_cloud_path = GRIB_DIR / f"icon_clct_f{step_str}.grib2"
        gdps_mslp_path = GRIB_DIR / f"gdps_mslp_f{step_str}.grib2"
        gdps_cloud_path = GRIB_DIR / f"gdps_cloud_f{step_str}.grib2"

        try:
            download_file(
                build_gfs_url(gfs_run_date, gfs_run_hour, step_str, "var_MSLET=on&lev_mean_sea_level=on"),
                gfs_mslp_path,
                f"GFS MSLP FH{step_str}",
            )
            download_file(
                build_gfs_url(
                    gfs_run_date,
                    gfs_run_hour,
                    step_str,
                    "var_TCDC=on&lev_entire_atmosphere=on",
                ),
                gfs_cloud_path,
                f"GFS Total Cloud Cover FH{step_str}",
            )

            ecmwf_available = ecmwf_enabled
            if ecmwf_available:
                try:
                    ecmwf_client.retrieve(
                        date=int(ecmwf_run.strftime("%Y%m%d")),
                        time=ecmwf_run.hour,
                        type="fc",
                        step=forecast_hour,
                        param=["msl", "tcc"],
                        target=str(ecmwf_path),
                    )
                except Exception as exc:
                    log_optional_model_failure(step_str, "EURO", exc)
                    ecmwf_available = False

            icon_available = icon_enabled
            if icon_available:
                try:
                    download_bz2_file(
                        build_icon_field_url(icon_run_date, icon_run_hour, "pmsl", step_str, "PMSL"),
                        icon_pmsl_path,
                        f"ICON PMSL FH{step_str}",
                    )
                    download_bz2_file(
                        build_icon_field_url(icon_run_date, icon_run_hour, "clct", step_str, "CLCT"),
                        icon_cloud_path,
                        f"ICON Total Cloud Cover FH{step_str}",
                    )
                except Exception as exc:
                    log_optional_model_failure(step_str, "ICON", exc)
                    icon_available = False

            gdps_available = gdps_enabled
            if gdps_available:
                try:
                    download_file(
                        build_gdps_field_url(gdps_run_date, gdps_run_hour, step_str, "Pressure_MSL"),
                        gdps_mslp_path,
                        f"GDPS MSLP FH{step_str}",
                    )
                    download_file(
                        build_gdps_field_url(gdps_run_date, gdps_run_hour, step_str, "TotalCloudCover_Sfc"),
                        gdps_cloud_path,
                        f"GDPS Total Cloud Cover FH{step_str}",
                    )
                except Exception as exc:
                    log_optional_model_failure(step_str, "GDPS", exc)
                    gdps_available = False

            gfs_mslp = open_gfs_mslp(gfs_mslp_path)
            gfs_cloud_cover = open_gfs_cloud_cover(gfs_cloud_path)

            if ecmwf_available:
                try:
                    ecmwf_mslp, ecmwf_cloud_cover = open_ecmwf_fields(ecmwf_path)
                    ecmwf_mslp = regrid_to_gfs(ecmwf_mslp, gfs_mslp)
                    ecmwf_cloud_cover = regrid_to_gfs(ecmwf_cloud_cover, gfs_mslp)
                except Exception as exc:
                    log_optional_model_failure(step_str, "EURO", exc)
                    ecmwf_available = False
                    ecmwf_mslp = None
                    ecmwf_cloud_cover = None
            else:
                ecmwf_mslp = None
                ecmwf_cloud_cover = None

            if gdps_available:
                try:
                    gdps_mslp = regrid_to_gfs(open_gdps_mslp(gdps_mslp_path), gfs_mslp)
                    gdps_cloud_cover = regrid_to_gfs(open_gdps_cloud_cover(gdps_cloud_path), gfs_mslp)
                except Exception as exc:
                    log_optional_model_failure(step_str, "GDPS", exc)
                    gdps_available = False
                    gdps_mslp = None
                    gdps_cloud_cover = None
            else:
                gdps_mslp = None
                gdps_cloud_cover = None

            if icon_indexer is None:
                icon_indexer = build_icon_indexer(icon_lats, icon_lons, gfs_mslp)

            if icon_available:
                try:
                    icon_mslp = remap_icon_to_gfs(
                        open_icon_scalar_field(icon_pmsl_path, "prmsl", scale=0.01),
                        icon_indexer,
                        gfs_mslp,
                        "icon_mslp",
                    )
                    icon_cloud_cover = normalize_cloud_cover_percent(
                        remap_icon_to_gfs(
                            open_icon_scalar_field(icon_cloud_path, ["clct", "tcc"]),
                            icon_indexer,
                            gfs_mslp,
                            "icon_cloud_cover",
                        )
                    )
                except Exception as exc:
                    log_optional_model_failure(step_str, "ICON", exc)
                    icon_available = False
                    icon_mslp = None
                    icon_cloud_cover = None
            else:
                icon_mslp = None
                icon_cloud_cover = None

            available_runs = [("GFS", gfs_run)]
            cloud_fields = [(gfs_cloud_cover, model_weights["gfs"])]
            mslp_fields = [(gfs_mslp, model_weights["gfs"])]

            if ecmwf_available:
                available_runs.append(("EURO", ecmwf_run))
                cloud_fields.append((ecmwf_cloud_cover, model_weights["ecmwf"]))
                mslp_fields.append((ecmwf_mslp, model_weights["ecmwf"]))

            if icon_available:
                available_runs.append(("ICON", icon_run))
                cloud_fields.append((icon_cloud_cover, model_weights["icon"]))
                mslp_fields.append((icon_mslp, model_weights["icon"]))

            if gdps_available:
                available_runs.append(("GDPS", gdps_run))
                cloud_fields.append((gdps_cloud_cover, model_weights["gdps"]))
                mslp_fields.append((gdps_mslp, model_weights["gdps"]))

            avg_cloud_cover = weighted_average_fields(cloud_fields, "avg_cloud_cover")
            avg_mslp = weighted_average_fields(mslp_fields, "avg_mslp")

            plot_average_fields(
                avg_cloud_cover,
                avg_mslp,
                available_runs,
                forecast_hour,
                step_str,
            )

        except Exception as exc:
            log_error(step_str, "Failed to process forecast hour", exc)
            print(f"Skipping FH{step_str}: {exc}")
        finally:
            gfs_cloud_path.unlink(missing_ok=True)
            gfs_mslp_path.unlink(missing_ok=True)
            ecmwf_path.unlink(missing_ok=True)
            icon_pmsl_path.unlink(missing_ok=True)
            icon_cloud_path.unlink(missing_ok=True)
            gdps_mslp_path.unlink(missing_ok=True)
            gdps_cloud_path.unlink(missing_ok=True)
            gc.collect()

    print(f"Finished generating GFS + ECMWF + ICON + GDPS total cloud cover/MSLP PNGs in {PNG_DIR}")


if __name__ == "__main__":
    main()
