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
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import requests
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
from scipy.spatial import cKDTree
import xarray as xr
from ecmwf.opendata import Client
from region_config import ACTIVE_REGION_NAMES, CONUS_EXTENT, REGION_LABELS, get_region_extent, prepare_region_png_dirs


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = Path("/var/data") if Path("/var/data").exists() else SCRIPT_DIR
BASE_DIR = OUTPUT_ROOT / "EURO_GFS_TOTAL_SNOW_OUTPUT"
GRIB_DIR = BASE_DIR / "grib"
PNG_DIR = BASE_DIR / "png"
LOG_FILE = BASE_DIR / "errors_euro_gfs_total_snow.txt"

MAX_DOWNLOAD_RETRIES = 3
LOCAL_TZ = ZoneInfo("America/New_York")
GFS_FILTER_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
ICON_BASE_URL = "https://opendata.dwd.de/weather/nwp/icon/grib"
GDPS_BASE_URL = "https://dd.weather.gc.ca"
PLOT_EXTENT = CONUS_EXTENT
ICON_MARGIN_DEGREES = 3.0

TOTAL_SNOW_LEVELS = [
    0, 0.1, 1, 2, 3, 4, 5, 6, 7, 8,
    10, 12, 16, 20, 24, 36, 48, 56,
    64, 72, 84, 100,
]
TOTAL_SNOW_COLORS = [
    "#ffffff", "#0d1a4a", "#1565c0", "#42a5f5", "#90caf9", "#e3f2fd",
    "#b39ddb", "#7e57c2", "#512da8", "#c2185b", "#f06292", "#81c784",
    "#388e3c", "#1b5e20", "#bdbdbd", "#757575", "#424242", "#212121",
    "#F4F805", "#FDAE04", "#F70909",
]
TOTAL_SNOW_CMAP = ListedColormap(TOTAL_SNOW_COLORS)
TOTAL_SNOW_NORM = BoundaryNorm(TOTAL_SNOW_LEVELS, len(TOTAL_SNOW_COLORS), clip=False)
MSLP_LEVELS = np.arange(960, 1060, 4)
MM_TO_INCHES = 1.0 / 25.4
METERS_TO_MM = 1000.0


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


def get_dataset_var(dataset, candidate_names):
    names = [candidate_names] if isinstance(candidate_names, str) else list(candidate_names)
    for candidate_name in names:
        if candidate_name in dataset.data_vars:
            return candidate_name

    if len(dataset.data_vars) == 1:
        return next(iter(dataset.data_vars))

    raise KeyError(f"No variable matched {names}; available variables: {sorted(dataset.data_vars)}")


def convert_depth_to_mm(field):
    values = np.asarray(field, dtype=np.float32)
    if np.nanmax(values) <= 20.0:
        values = values * METERS_TO_MM
    return values


def open_gfs_mslp(path):
    dataset = xr.open_dataset(path, engine="cfgrib", filter_by_keys={"typeOfLevel": "meanSea"})
    try:
        variable_name = "mslet" if "mslet" in dataset.data_vars else next(iter(dataset.data_vars))
        field = dataset[variable_name].squeeze().values / 100.0
        return as_data_array(field, dataset["latitude"].values, dataset["longitude"].values, "gfs_mslp")
    finally:
        dataset.close()
        gc.collect()


def open_gfs_snow_depth(path):
    dataset = xr.open_dataset(path, engine="cfgrib", filter_by_keys={"stepType": "instant"})
    try:
        variable_name = get_dataset_var(dataset, ["snod", "sde", "sd", "sdep"])
        field = convert_depth_to_mm(dataset[variable_name].squeeze().values)
        return as_data_array(field, dataset["latitude"].values, dataset["longitude"].values, "gfs_snow_depth_mm")
    finally:
        dataset.close()
        gc.collect()


def open_ecmwf_fields(path):
    mean_sea_dataset = xr.open_dataset(path, engine="cfgrib", filter_by_keys={"typeOfLevel": "meanSea"})
    instant_dataset = xr.open_dataset(path, engine="cfgrib", filter_by_keys={"stepType": "instant"})
    try:
        msl = as_data_array(
            mean_sea_dataset["msl"].squeeze().values / 100.0,
            mean_sea_dataset["latitude"].values,
            mean_sea_dataset["longitude"].values,
            "ecmwf_mslp",
        )
        snow_depth_name = get_dataset_var(instant_dataset, ["sd", "sde"])
        snow_depth = as_data_array(
            convert_depth_to_mm(instant_dataset[snow_depth_name].squeeze().values),
            instant_dataset["latitude"].values,
            instant_dataset["longitude"].values,
            "ecmwf_snow_depth_mm",
        )
        return msl, snow_depth
    finally:
        mean_sea_dataset.close()
        instant_dataset.close()
        gc.collect()


def open_gdps_scalar_field(path, names, scale=1.0):
    dataset = xr.open_dataset(path, engine="cfgrib")
    try:
        variable_name = get_dataset_var(dataset, names)
        field = dataset[variable_name].squeeze().values * scale
        return as_data_array(field, dataset["latitude"].values, dataset["longitude"].values, f"gdps_{variable_name}")
    finally:
        dataset.close()
        gc.collect()


def open_gdps_mslp(path):
    return open_gdps_scalar_field(path, ["prmsl"], scale=0.01)


def open_gdps_snow_depth(path):
    field = open_gdps_scalar_field(path, ["snd", "sndepth", "sd", "sde", "unknown"])
    field.values[:] = convert_depth_to_mm(field.values)
    field.name = "gdps_snow_depth_mm"
    return field


def open_icon_coord_field(path, variable_name):
    dataset = xr.open_dataset(path, engine="cfgrib")
    try:
        return np.asarray(dataset[variable_name].squeeze().values, dtype=np.float64)
    finally:
        dataset.close()
        gc.collect()


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
        gc.collect()


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


def positive_snow_increment(current_depth, previous_depth):
    return xr.where(current_depth - previous_depth > 0, current_depth - previous_depth, 0)


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


def plot_average_fields(avg_total_snow, avg_mslp, available_runs, forecast_hour, step_str):
    gfs_run = available_runs[0][1]
    valid_utc, local_time, local_day = format_valid_time(gfs_run, forecast_hour)
    lon2d, lat2d = np.meshgrid(avg_total_snow.longitude.values, avg_total_snow.latitude.values)
    avg_total_snow_inches = avg_total_snow * MM_TO_INCHES

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

        snow_mesh = ax.contourf(
            lon2d,
            lat2d,
            avg_total_snow_inches.values,
            levels=TOTAL_SNOW_LEVELS,
            cmap=TOTAL_SNOW_CMAP,
            norm=TOTAL_SNOW_NORM,
            extend="max",
            transform=ccrs.PlateCarree(),
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
        cbar_ax_snow = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
        snow_bar = fig.colorbar(snow_mesh, cax=cbar_ax_snow, orientation="horizontal")
        snow_bar.set_ticks(TOTAL_SNOW_LEVELS)
        snow_bar.set_label("Total snowfall from positive snow-depth change (in)", fontsize=7)
        snow_bar.ax.tick_params(labelsize=6)

        ax.set_title(
            (
                f"{model_text} total snowfall & MSLP | {REGION_LABELS[region_name]} | FH{step_str} | "
                f"Valid {valid_utc:%Y-%m-%d %HZ} ({local_time} ET, {local_day})\n"
                f"Runs: {run_text}\n"
                f"{weight_text}"
            ),
            fontsize=7,
            pad=6,
        )

        output_path = PNG_DIR / region_name / f"euro_gfs_total_snow_mslp_avg_{step_str}.png"
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
    print(f"Using latest ECMWF run: {ecmwf_run:%Y-%m-%d %HZ}")
    model_weights = get_model_weights(gfs_run, icon_run, ecmwf_run, gdps_run)

    icon_clat_path = GRIB_DIR / f"icon_clat_{icon_run_date}{icon_run_hour}.grib2"
    icon_clon_path = GRIB_DIR / f"icon_clon_{icon_run_date}{icon_run_hour}.grib2"
    gfs_baseline_snow_path = GRIB_DIR / f"gfs_snow_depth_f000.grib2"
    ecmwf_baseline_path = GRIB_DIR / "ecmwf_baseline_f000.grib2"
    icon_baseline_snow_path = GRIB_DIR / "icon_snow_depth_f000.grib2"
    gdps_baseline_snow_path = GRIB_DIR / "gdps_snow_depth_f000.grib2"

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
    download_file(
        build_gfs_url(gfs_run_date, gfs_run_hour, "000", "var_SNOD=on&lev_surface=on"),
        gfs_baseline_snow_path,
        "GFS snow depth FH000",
    )
    icon_lats = open_icon_coord_field(icon_clat_path, "tlat")
    icon_lons = open_icon_coord_field(icon_clon_path, "tlon")
    gfs_baseline_depth = open_gfs_snow_depth(gfs_baseline_snow_path)

    ecmwf_baseline_depth = None
    icon_baseline_depth = None
    gdps_baseline_depth = None

    try:
        ecmwf_client.retrieve(
            date=int(ecmwf_run.strftime("%Y%m%d")),
            time=ecmwf_run.hour,
            type="fc",
            step=0,
            param=["sd"],
            target=str(ecmwf_baseline_path),
        )
        _, ecmwf_baseline_depth = open_ecmwf_fields(ecmwf_baseline_path)
    except Exception as exc:
        log_optional_model_failure("000", "ECMWF", exc)
        ecmwf_baseline_depth = None

    try:
        download_bz2_file(
            build_icon_field_url(icon_run_date, icon_run_hour, "h_snow", "000", "H_SNOW"),
            icon_baseline_snow_path,
            "ICON snow depth FH000",
        )
        icon_baseline_depth = open_icon_scalar_field(icon_baseline_snow_path, ["h_snow", "sd", "sde"])
        icon_baseline_depth = convert_depth_to_mm(icon_baseline_depth)
    except Exception as exc:
        log_optional_model_failure("000", "ICON", exc)
        icon_baseline_depth = None

    try:
        download_file(
            build_gdps_field_url(gdps_run_date, gdps_run_hour, "000", "SnowDepth_Sfc"),
            gdps_baseline_snow_path,
            "GDPS snow depth FH000",
        )
        gdps_baseline_depth = open_gdps_snow_depth(gdps_baseline_snow_path)
    except Exception as exc:
        log_optional_model_failure("000", "GDPS", exc)
        gdps_baseline_depth = None

    previous_gfs_depth = gfs_baseline_depth.copy(deep=True)
    gfs_total_snow = xr.zeros_like(gfs_baseline_depth)
    previous_ecmwf_depth = None
    ecmwf_total_snow = None
    previous_icon_depth = None
    icon_total_snow = None
    previous_gdps_depth = None
    gdps_total_snow = None
    icon_indexer = None

    for forecast_hour in forecast_steps:
        step_str = f"{forecast_hour:03d}"
        ecmwf_enabled = forecast_hour <= run_config["ecmwf_max_hour"]
        icon_enabled = forecast_hour <= run_config["icon_max_hour"]
        gdps_enabled = forecast_hour <= run_config["gdps_max_hour"]

        gfs_mslp_path = GRIB_DIR / f"gfs_mslp_f{step_str}.grib2"
        gfs_snow_depth_path = GRIB_DIR / f"gfs_snow_depth_f{step_str}.grib2"
        ecmwf_path = GRIB_DIR / f"ecmwf_fields_f{step_str}.grib2"
        icon_pmsl_path = GRIB_DIR / f"icon_pmsl_f{step_str}.grib2"
        icon_snow_depth_path = GRIB_DIR / f"icon_snow_depth_f{step_str}.grib2"
        gdps_mslp_path = GRIB_DIR / f"gdps_mslp_f{step_str}.grib2"
        gdps_snow_depth_path = GRIB_DIR / f"gdps_snow_depth_f{step_str}.grib2"

        try:
            download_file(
                build_gfs_url(gfs_run_date, gfs_run_hour, step_str, "var_MSLET=on&lev_mean_sea_level=on"),
                gfs_mslp_path,
                f"GFS MSLP FH{step_str}",
            )
            download_file(
                build_gfs_url(gfs_run_date, gfs_run_hour, step_str, "var_SNOD=on&lev_surface=on"),
                gfs_snow_depth_path,
                f"GFS snow depth FH{step_str}",
            )

            ecmwf_available = ecmwf_enabled and ecmwf_baseline_depth is not None
            if ecmwf_available:
                try:
                    ecmwf_client.retrieve(
                        date=int(ecmwf_run.strftime("%Y%m%d")),
                        time=ecmwf_run.hour,
                        type="fc",
                        step=forecast_hour,
                        param=["msl", "sd"],
                        target=str(ecmwf_path),
                    )
                except Exception as exc:
                    log_optional_model_failure(step_str, "ECMWF", exc)
                    ecmwf_available = False

            icon_available = icon_enabled and icon_baseline_depth is not None
            if icon_available:
                try:
                    download_bz2_file(
                        build_icon_field_url(icon_run_date, icon_run_hour, "pmsl", step_str, "PMSL"),
                        icon_pmsl_path,
                        f"ICON PMSL FH{step_str}",
                    )
                    download_bz2_file(
                        build_icon_field_url(icon_run_date, icon_run_hour, "h_snow", step_str, "H_SNOW"),
                        icon_snow_depth_path,
                        f"ICON snow depth FH{step_str}",
                    )
                except Exception as exc:
                    log_optional_model_failure(step_str, "ICON", exc)
                    icon_available = False

            gdps_available = gdps_enabled and gdps_baseline_depth is not None
            if gdps_available:
                try:
                    download_file(
                        build_gdps_field_url(gdps_run_date, gdps_run_hour, step_str, "Pressure_MSL"),
                        gdps_mslp_path,
                        f"GDPS MSLP FH{step_str}",
                    )
                    download_file(
                        build_gdps_field_url(gdps_run_date, gdps_run_hour, step_str, "SnowDepth_Sfc"),
                        gdps_snow_depth_path,
                        f"GDPS snow depth FH{step_str}",
                    )
                except Exception as exc:
                    log_optional_model_failure(step_str, "GDPS", exc)
                    gdps_available = False

            gfs_mslp = open_gfs_mslp(gfs_mslp_path)
            gfs_snow_depth = open_gfs_snow_depth(gfs_snow_depth_path)
            gfs_snow_increment = positive_snow_increment(gfs_snow_depth, previous_gfs_depth)
            previous_gfs_depth = gfs_snow_depth.copy(deep=True)
            gfs_total_snow = gfs_total_snow + gfs_snow_increment

            if ecmwf_available:
                try:
                    ecmwf_mslp, ecmwf_snow_depth = open_ecmwf_fields(ecmwf_path)
                    ecmwf_mslp = regrid_to_gfs(ecmwf_mslp, gfs_mslp)
                    ecmwf_snow_depth = regrid_to_gfs(ecmwf_snow_depth, gfs_mslp)
                    if previous_ecmwf_depth is None:
                        previous_ecmwf_depth = regrid_to_gfs(ecmwf_baseline_depth, gfs_mslp)
                        ecmwf_total_snow = xr.zeros_like(previous_ecmwf_depth)
                    ecmwf_snow_increment = positive_snow_increment(ecmwf_snow_depth, previous_ecmwf_depth)
                    previous_ecmwf_depth = ecmwf_snow_depth.copy(deep=True)
                    ecmwf_total_snow = ecmwf_total_snow + ecmwf_snow_increment
                except Exception as exc:
                    log_optional_model_failure(step_str, "ECMWF", exc)
                    ecmwf_available = False
                    ecmwf_mslp = None
                    ecmwf_total_snow = None
            else:
                ecmwf_mslp = None
                ecmwf_total_snow = None

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
                    icon_snow_depth = remap_icon_to_gfs(
                        convert_depth_to_mm(open_icon_scalar_field(icon_snow_depth_path, ["h_snow", "sd", "sde"])),
                        icon_indexer,
                        gfs_mslp,
                        "icon_snow_depth_mm",
                    )
                    if previous_icon_depth is None:
                        previous_icon_depth = remap_icon_to_gfs(icon_baseline_depth, icon_indexer, gfs_mslp, "icon_baseline_depth_mm")
                        icon_total_snow = xr.zeros_like(previous_icon_depth)
                    icon_snow_increment = positive_snow_increment(icon_snow_depth, previous_icon_depth)
                    previous_icon_depth = icon_snow_depth.copy(deep=True)
                    icon_total_snow = icon_total_snow + icon_snow_increment
                except Exception as exc:
                    log_optional_model_failure(step_str, "ICON", exc)
                    icon_available = False
                    icon_mslp = None
                    icon_total_snow = None
            else:
                icon_mslp = None
                icon_total_snow = None

            if gdps_available:
                try:
                    gdps_mslp = regrid_to_gfs(open_gdps_mslp(gdps_mslp_path), gfs_mslp)
                    gdps_snow_depth = regrid_to_gfs(open_gdps_snow_depth(gdps_snow_depth_path), gfs_mslp)
                    if previous_gdps_depth is None:
                        previous_gdps_depth = regrid_to_gfs(gdps_baseline_depth, gfs_mslp)
                        gdps_total_snow = xr.zeros_like(previous_gdps_depth)
                    gdps_snow_increment = positive_snow_increment(gdps_snow_depth, previous_gdps_depth)
                    previous_gdps_depth = gdps_snow_depth.copy(deep=True)
                    gdps_total_snow = gdps_total_snow + gdps_snow_increment
                except Exception as exc:
                    log_optional_model_failure(step_str, "GDPS", exc)
                    gdps_available = False
                    gdps_mslp = None
                    gdps_total_snow = None
            else:
                gdps_mslp = None
                gdps_total_snow = None

            available_runs = [("GFS", gfs_run)]
            snow_fields = [(gfs_total_snow, model_weights["gfs"])]
            mslp_fields = [(gfs_mslp, model_weights["gfs"])]

            if ecmwf_available:
                available_runs.append(("ECMWF", ecmwf_run))
                snow_fields.append((ecmwf_total_snow, model_weights["ecmwf"]))
                mslp_fields.append((ecmwf_mslp, model_weights["ecmwf"]))

            if icon_available:
                available_runs.append(("ICON", icon_run))
                snow_fields.append((icon_total_snow, model_weights["icon"]))
                mslp_fields.append((icon_mslp, model_weights["icon"]))

            if gdps_available:
                available_runs.append(("GDPS", gdps_run))
                snow_fields.append((gdps_total_snow, model_weights["gdps"]))
                mslp_fields.append((gdps_mslp, model_weights["gdps"]))

            avg_total_snow = weighted_average_fields(snow_fields, "avg_total_snow")
            avg_mslp = weighted_average_fields(mslp_fields, "avg_mslp")

            plot_average_fields(
                avg_total_snow,
                avg_mslp,
                available_runs,
                forecast_hour,
                step_str,
            )

        except Exception as exc:
            log_error(step_str, "Failed to process forecast hour", exc)
            print(f"Skipping FH{step_str}: {exc}")
        finally:
            gfs_mslp_path.unlink(missing_ok=True)
            gfs_snow_depth_path.unlink(missing_ok=True)
            ecmwf_path.unlink(missing_ok=True)
            icon_pmsl_path.unlink(missing_ok=True)
            icon_snow_depth_path.unlink(missing_ok=True)
            gdps_mslp_path.unlink(missing_ok=True)
            gdps_snow_depth_path.unlink(missing_ok=True)
            gc.collect()

    gfs_baseline_snow_path.unlink(missing_ok=True)
    ecmwf_baseline_path.unlink(missing_ok=True)
    icon_baseline_snow_path.unlink(missing_ok=True)
    gdps_baseline_snow_path.unlink(missing_ok=True)
    icon_clat_path.unlink(missing_ok=True)
    icon_clon_path.unlink(missing_ok=True)
    print(f"Finished generating GFS + ECMWF + ICON + GDPS total snowfall/MSLP PNGs in {PNG_DIR}")


if __name__ == "__main__":
    main()
