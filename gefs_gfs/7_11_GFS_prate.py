import os
import shutil
import requests
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe  # <-- ADDED: ensure text is outlined and on top
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time
import gc
from scipy.ndimage import zoom, gaussian_filter, minimum_filter, maximum_filter  # Add imports for filtering
from datetime import datetime, timedelta
import pytz  # Add this import for timezone handling


current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR_AVG = os.path.join(current_dir, '7_11_GFS_OUTPUT')

# Make sure AVG subfolders exist (recreate grib cleanly)
grib_dir = os.path.join(BASE_DIR_AVG, "grib")
png_dir = os.path.join(BASE_DIR_AVG, "png")
os.makedirs(BASE_DIR_AVG, exist_ok=True)

# Remove entire grib folder if it exists (ensures it's truly cleared), then recreate it
if os.path.isdir(grib_dir):
    try:
        shutil.rmtree(grib_dir)
        print(f"Removed existing grib directory: {grib_dir}")
    except Exception as e:
        print(f"Failed to remove grib directory {grib_dir}: {e}")
os.makedirs(grib_dir, exist_ok=True)
os.makedirs(png_dir, exist_ok=True)

# NEW: directory to store previous-run averaged arrays for accuracy comparisons
prev_dir = os.path.join(BASE_DIR_AVG, "prev")
os.makedirs(prev_dir, exist_ok=True)

# File to track completed forecast steps for the current run
processed_steps_file = os.path.join(BASE_DIR_AVG, "processed_steps_7_11.txt")

# Clear the processed steps file at the start of a new run
if os.path.exists(processed_steps_file):
    os.remove(processed_steps_file)

# Helper function to load processed steps
def load_processed_steps():
    if os.path.exists(processed_steps_file):
        with open(processed_steps_file, "r") as f:
            return set(line.strip() for line in f)
    return set()

# Helper function to save a processed step
def save_processed_step(step):
    with open(processed_steps_file, "a") as f:
        f.write(f"{step}\n")

def load_grib_field(path, variable_name, *, filter_by_keys=None, scale=1.0, dtype=np.float32):
    dataset = xr.open_dataset(path, engine="cfgrib", filter_by_keys=filter_by_keys)
    try:
        data = np.asarray(dataset[variable_name].values, dtype=dtype).squeeze()
        lats = np.asarray(dataset["latitude"].values) if "latitude" in dataset else None
        lons = np.asarray(dataset["longitude"].values) if "longitude" in dataset else None
    finally:
        dataset.close()

    if scale != 1.0:
        data *= scale
    return data, lats, lons

def load_csnow_array(path, dtype=np.float32):
    last_error = None
    for kwargs in ({}, {'filter_by_keys': {'stepType': 'avg'}}, {'filter_by_keys': {'typeOfLevel': 'surface'}}):
        dataset = None
        try:
            dataset = xr.open_dataset(path, engine="cfgrib", **kwargs)
            if dataset.data_vars:
                csnow_vars = [v for v in dataset.data_vars if 'csnow' in v.lower()]
                csnow_var_name = csnow_vars[0] if csnow_vars else list(dataset.data_vars)[0]
                return np.asarray(dataset[csnow_var_name].values, dtype=dtype).squeeze()
        except Exception as exc:
            last_error = exc
        finally:
            if dataset is not None:
                dataset.close()
    raise ValueError(f"No data variables found in {path}") from last_error

def resize_to_match(data, target_shape, order):
    if data.shape == target_shape:
        return data
    zoom_factors = (target_shape[0] / data.shape[0], target_shape[1] / data.shape[1])
    return zoom(data, zoom_factors, order=order)

# -----------------------------
# CLEAR ONLY AVG PNGs
# -----------------------------
avg_png_dir = png_dir
for file in os.listdir(avg_png_dir):
    file_path = os.path.join(avg_png_dir, file)
    try:
        os.remove(file_path)
        print(f"Deleted old AVG PNG: {file_path}")
    except Exception as e:
        print(f"Failed to delete {file_path}: {e}")

# -----------------------------
# DETERMINE RUN HOUR BASED ON CURRENT TIME (EST)
# -----------------------------
def determine_run_hour():
    # Convert current UTC time to EST
    utc_now = datetime.utcnow()
    est_now = utc_now - timedelta(hours=5)  # Adjust UTC to EST (standard time)
    
    # If daylight saving time is active, adjust EST offset
    if est_now.month in [3, 4, 5, 6, 7, 8, 9, 10, 11]:  # Approximate DST months
        est_now = utc_now - timedelta(hours=4)

    hour = est_now.hour

    # Determine the run hour based on EST time
    if hour >= 23:  # 11 PM EST or later
        return "00"
    elif hour >= 17:  # 5 PM EST or later
        return "18"
    elif hour >= 11:  # 11 AM EST or later
        return "12"
    elif hour >= 5:  # 5 AM EST or later
        return "06"
    else:
        return "18"  # Default to the previous day's 18Z run if before 5 AM EST

# -----------------------------
# HELPER FUNCTION TO FORMAT LOCAL TIME
# -----------------------------
def format_local_time(run_date, run_hour, forecast_hour):
    """
    Convert the run date, run hour, and forecast hour to local time (EST) and return the time and day.
    """
    # Combine run_date and run_hour to create a complete UTC datetime
    run_datetime = datetime.strptime(f"{run_date} {run_hour}", "%Y%m%d %H")  # Run datetime in UTC
    forecast_datetime = run_datetime + timedelta(hours=forecast_hour)  # Add forecast hours

    # Convert UTC to EST
    forecast_datetime_est = forecast_datetime - timedelta(hours=5)  # Adjust for EST (standard time)
    if forecast_datetime_est.month in [3, 4, 5, 6, 7, 8, 9, 10, 11]:  # Approximate DST months
        forecast_datetime_est = forecast_datetime - timedelta(hours=4)  # Adjust for DST

    # Format the local time as a 12-hour clock
    local_time = forecast_datetime_est.strftime("%I %p").lstrip("0")  # Remove leading zero
    forecast_day = forecast_datetime_est.strftime("%A")  # Get the day of the week
    return local_time, forecast_day

# -----------------------------
# FIND MOST RECENT RUN WITH VALID DATA
# -----------------------------
def find_valid_run():
    now = datetime.utcnow()
    run_hour = determine_run_hour()  # Determine the initial run hour
    while True:
        run_date = now.strftime("%Y%m%d")
        test_url = (
            f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{run_date}/{run_hour}/"
        )
        response = requests.head(test_url)
        if response.status_code == 200:
            print(f"Found valid run: {run_date} {run_hour}Z")
            return run_date, run_hour
        # Go back to the previous run
        if run_hour == "18":
            run_hour = "12"
        elif run_hour == "12":
            run_hour = "06"
        elif run_hour == "06":
            run_hour = "00"
        elif run_hour == "00":
            run_hour = "18"
            now -= timedelta(days=1)  # Go back one day if all runs for the day fail

# Get the most recent valid run date and hour before starting downloads
run_date, run_hour = find_valid_run()

# -----------------------------
# FORECAST SETTINGS
# -----------------------------
forecast_steps = list(range(6, 187, 6))  # 0 → 186

# -----------------------------
# COLORMAP
# -----------------------------
prate_levels = [0.1, 0.25, 0.5, 0.75, 1.5, 2, 2.5, 3, 4, 6, 10, 16, 24]
prate_colors = [
    "#b6ffb6", "#54f354", "#19a319", "#016601",
    "#c9c938", "#f5f825", "#ffd700", "#ffa500",
    "#ff7f50", "#ff4500", "#ff1493", "#9400d3"
]
cmap = ListedColormap(prate_colors)
norm = BoundaryNorm(prate_levels, ncolors=cmap.N, clip=False)

# Add MSLP levels and colormap
mslp_levels = np.arange(960, 1060, 4)  # Contour levels for MSLP in hPa
mslp_cmap = plt.cm.viridis  # Use a perceptually uniform colormap for MSLP

# -----------------------------
# URL BASES
# -----------------------------
base_url_gfs = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
base_url_gefs = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p50b.pl"
gefs_members = ["07", "11"]  # members to include

# -----------------------------
# PROCESS FORECAST HOURS
# -----------------------------
for step in forecast_steps:
    step_str = f"{step:03d}"
    forecast_hour = step  # Forecast hour in hours
    ensemble_count = 0
    sum_data = None
    sum_mslp = None
    sum_csnow = None

    # ---- GFS ----
    while True:  # Retry logic for unavailable forecast hours
        gfs_file = f"gfs.t{run_hour}z.pgrb2.0p25.f{step_str}_prate.grib2"
        gfs_mslp_file = f"gfs.t{run_hour}z.pgrb2.0p25.f{step_str}_mslp.grib2"
        gfs_csnow_file = f"gfs.t{run_hour}z.pgrb2.0p25.f{step_str}_csnow.grib2"
        gfs_path = os.path.join(BASE_DIR_AVG, "grib", gfs_file)
        gfs_mslp_path = os.path.join(BASE_DIR_AVG, "grib", gfs_mslp_file)
        gfs_csnow_path = os.path.join(BASE_DIR_AVG, "grib", gfs_csnow_file)
        gfs_url = (
            f"{base_url_gfs}?file=gfs.t{run_hour}z.pgrb2.0p25.f{step_str}"
            f"&var_PRATE=on&lev_surface=on"
            f"&subregion=&leftlon=220&rightlon=300&toplat=55&bottomlat=20"
            f"&dir=%2Fgfs.{run_date}%2F{run_hour}%2Fatmos"
        )
        gfs_mslp_url = (
            f"{base_url_gfs}?file=gfs.t{run_hour}z.pgrb2.0p25.f{step_str}"
            f"&var_MSLET=on&lev_mean_sea_level=on"
            f"&subregion=&leftlon=220&rightlon=300&toplat=55&bottomlat=20"
            f"&dir=%2Fgfs.{run_date}%2F{run_hour}%2Fatmos"
        )
        gfs_csnow_url = (
            f"{base_url_gfs}?file=gfs.t{run_hour}z.pgrb2.0p25.f{step_str}"
            f"&var_CSNOW=on&lev_surface=on"
            f"&subregion=&leftlon=220&rightlon=300&toplat=55&bottomlat=20"
            f"&dir=%2Fgfs.{run_date}%2F{run_hour}%2Fatmos"
        )

        # Download PRATE if not exist
        if not os.path.exists(gfs_path):
            print(f"Downloading GFS FH{step_str} PRATE …")
            r = requests.get(gfs_url, stream=True)
            if r.status_code == 200:
                with open(gfs_path, 'wb') as f:
                    for chunk in r.iter_content(1024*64):
                        if chunk:
                            f.write(chunk)
                print(f"Saved GFS GRIB: {gfs_path}")
                time.sleep(2)
            else:
                print(f"Failed to download GFS {gfs_file}, status code: {r.status_code}")
                run_date, run_hour = find_valid_run()
                continue

        # Download MSLP if not exist
        if not os.path.exists(gfs_mslp_path):
            print(f"Downloading GFS FH{step_str} MSLP …")
            r = requests.get(gfs_mslp_url, stream=True)
            if r.status_code == 200:
                with open(gfs_mslp_path, 'wb') as f:
                    for chunk in r.iter_content(1024*64):
                        if chunk:
                            f.write(chunk)
                print(f"Saved GFS GRIB: {gfs_mslp_path}")
                time.sleep(2)
            else:
                print(f"Failed to download GFS {gfs_mslp_file}, status code: {r.status_code}")
                run_date, run_hour = find_valid_run()
                continue

        if not os.path.exists(gfs_csnow_path):
            print(f"Downloading GFS FH{step_str} CSNOW …")
            r = requests.get(gfs_csnow_url, stream=True)
            if r.status_code == 200:
                with open(gfs_csnow_path, 'wb') as f:
                    for chunk in r.iter_content(1024*64):
                        if chunk:
                            f.write(chunk)
                print(f"Saved GFS CSNOW GRIB: {gfs_csnow_path}")
                time.sleep(2)
            else:
                print(f"Failed to download GFS {gfs_csnow_file}, status code: {r.status_code}")
                run_date, run_hour = find_valid_run()
                continue

        break

    # Open GFS PRATE
    try:
        data_gfs, lats, lons = load_grib_field(
            gfs_path,
            'prate',
            filter_by_keys={'stepType': 'avg'},
            scale=3600,
        )
        lons_plot = np.where(lons > 180, lons - 360, lons)
    except Exception as e:
        print(f"Failed to open GFS GRIB FH{step_str}: {e}")
        continue

    # Open GFS MSLP
    try:
        data_gfs_mslp, _, _ = load_grib_field(
            gfs_mslp_path,
            'mslet',
            filter_by_keys={'typeOfLevel': 'meanSea'},
            scale=0.01,
        )
    except Exception as e:
        print(f"Failed to open GFS MSLP GRIB FH{step_str}: {e}")
        continue

    # Open GFS CSNOW
    try:
        data_gfs_csnow = load_csnow_array(gfs_csnow_path)
    except Exception as e:
        print(f"Failed to open GFS CSNOW GRIB FH{step_str}: {e}")
        continue

    sum_data = data_gfs
    sum_mslp = data_gfs_mslp
    sum_csnow = data_gfs_csnow
    ensemble_count = 1

    # ---- GEFS MEMBERS ----
    for member in gefs_members:
        while True:  # Retry logic for unavailable GEFS members
            gefs_file = f"gep{member}.t{run_hour}z.pgrb2b.0p50.f{step_str}_prate.grib2"
            gefs_mslp_file = f"gep{member}.t{run_hour}z.pgrb2b.0p50.f{step_str}_mslp.grib2"
            gefs_csnow_file = f"gep{member}.t{run_hour}z.pgrb2a.0p50.f{step_str}_csnow.grib2"
            gefs_path = os.path.join(BASE_DIR_AVG, "grib", gefs_file)
            gefs_mslp_path = os.path.join(BASE_DIR_AVG, "grib", gefs_mslp_file)
            gefs_csnow_path = os.path.join(BASE_DIR_AVG, "grib", gefs_csnow_file)
            gefs_url = (
                f"{base_url_gefs}?file=gep{member}.t{run_hour}z.pgrb2b.0p50.f{step_str}"
                f"&var_PRATE=on&lev_surface=on"
                f"&subregion=&leftlon=220&rightlon=300&toplat=55&bottomlat=20"
                f"&dir=%2Fgefs.{run_date}%2F{run_hour}%2Fatmos%2Fpgrb2bp5"
            )
            gefs_mslp_url = (
                f"{base_url_gefs}?file=gep{member}.t{run_hour}z.pgrb2b.0p50.f{step_str}"
                f"&var_MSLET=on&lev_mean_sea_level=on"
                f"&subregion=&leftlon=220&rightlon=300&toplat=55&bottomlat=20"
                f"&dir=%2Fgefs.{run_date}%2F{run_hour}%2Fatmos%2Fpgrb2bp5"
            )
            gefs_csnow_url = (
                f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p50a.pl?file=gep{member}.t{run_hour}z.pgrb2a.0p50.f{step_str}"
                f"&var_CSNOW=on&lev_surface=on"
                f"&subregion=&leftlon=220&rightlon=300&toplat=55&bottomlat=20"
                f"&dir=%2Fgefs.{run_date}%2F{run_hour}%2Fatmos%2Fpgrb2ap5"
            )

            # Download PRATE and MSLP if not exist
            if not os.path.exists(gefs_path):
                print(f"Downloading GEFS member {member}, FH{step_str} PRATE …")
                r = requests.get(gefs_url, stream=True)
                if r.status_code == 200:
                    with open(gefs_path, 'wb') as f:
                        for chunk in r.iter_content(1024*64):
                            if chunk:
                                f.write(chunk)
                    print(f"Saved GEFS GRIB: {gefs_path}")
                    time.sleep(2)
                else:
                    print(f"Failed to download GEFS {gefs_file}, status code: {r.status_code}")
                    run_date, run_hour = find_valid_run()
                    continue

            if not os.path.exists(gefs_mslp_path):
                print(f"Downloading GEFS member {member}, FH{step_str} MSLP …")
                r = requests.get(gefs_mslp_url, stream=True)
                if r.status_code == 200:
                    with open(gefs_mslp_path, 'wb') as f:
                        for chunk in r.iter_content(1024*64):
                            if chunk:
                                f.write(chunk)
                    print(f"Saved GEFS GRIB: {gefs_mslp_path}")
                    time.sleep(2)
                else:
                    print(f"Failed to download GEFS {gefs_mslp_file}, status code: {r.status_code}")
                    run_date, run_hour = find_valid_run()
                    continue

            if not os.path.exists(gefs_csnow_path):
                print(f"Downloading GEFS member {member}, FH{step_str} CSNOW …")
                r = requests.get(gefs_csnow_url, stream=True)
                if r.status_code == 200:
                    with open(gefs_csnow_path, 'wb') as f:
                        for chunk in r.iter_content(1024*64):
                            if chunk:
                                f.write(chunk)
                    print(f"Saved GEFS CSNOW GRIB: {gefs_csnow_path}")
                    time.sleep(2)
                else:
                    print(f"Failed to download GEFS {gefs_csnow_file}, status code: {r.status_code}")
                    run_date, run_hour = find_valid_run()
                    continue

            break

        # Open GEFS PRATE
        try:
            data_gefs, _, _ = load_grib_field(
                gefs_path,
                'prate',
                filter_by_keys={'stepType': 'avg'},
                scale=3600,
            )
        except Exception as e:
            print(f"Failed to open GEFS member {member} PRATE FH{step_str}: {e}")
            continue

        # Open GEFS MSLP
        try:
            data_gefs_mslp, _, _ = load_grib_field(
                gefs_mslp_path,
                'mslet',
                filter_by_keys={'typeOfLevel': 'meanSea'},
                scale=0.01,
            )
        except Exception as e:
            print(f"Failed to open GEFS member {member} MSLP FH{step_str}: {e}")
            continue

        # Open GEFS CSNOW
        try:
            data_gefs_csnow = load_csnow_array(gefs_csnow_path)
        except Exception as e:
            print(f"Failed to open GEFS member {member} CSNOW FH{step_str}: {e}")
            continue

        # Resize GEFS PRATE and MSLP to match GFS grid if needed
        data_gefs_resized = resize_to_match(data_gefs, data_gfs.shape, order=1)
        data_gefs_mslp_resized = resize_to_match(data_gefs_mslp, data_gfs_mslp.shape, order=1)
        data_gefs_csnow_resized = resize_to_match(data_gefs_csnow, data_gfs_csnow.shape, order=0)

        sum_data += data_gefs_resized
        sum_mslp += data_gefs_mslp_resized
        sum_csnow += data_gefs_csnow_resized
        ensemble_count += 1

        del data_gefs, data_gefs_mslp, data_gefs_csnow
        del data_gefs_resized, data_gefs_mslp_resized, data_gefs_csnow_resized

    # ---- COMPUTE FINAL AVERAGES ----
    avg_data = sum_data / ensemble_count
    avg_mslp = sum_mslp / ensemble_count
    avg_csnow = sum_csnow / ensemble_count

    # -----------------------------
    # NEW: compute accuracy vs previous run for same VALID TIME
    # -----------------------------
    # compute run datetime (UTC) and valid datetime for this forecast
    run_datetime_utc = datetime.strptime(f"{run_date} {run_hour}", "%Y%m%d %H")
    valid_dt = run_datetime_utc + timedelta(hours=forecast_hour)
    valid_tag = valid_dt.strftime("%Y%m%d_%H")  # e.g. 20240203_18 (UTC valid time)
    prev_file = os.path.join(prev_dir, f"avg_prate_valid_{valid_tag}.npy")

    accuracy_pct = None
    try:
        if os.path.exists(prev_file):
            prev_data = np.load(prev_file)
            # resize previous to current if needed
            if prev_data.shape != avg_data.shape:
                zoom_factors = (avg_data.shape[0] / prev_data.shape[0], avg_data.shape[1] / prev_data.shape[1])
                prev_resized = zoom(prev_data, zoom_factors, order=1)
            else:
                prev_resized = prev_data
            # mask NaNs and compute MAE relative to previous magnitude
            mask = ~np.isnan(avg_data) & ~np.isnan(prev_resized)
            if np.any(mask):
                mae = np.mean(np.abs(avg_data[mask] - prev_resized[mask]))
                mean_prev = np.mean(np.abs(prev_resized[mask]))
                accuracy_pct = max(0.0, min(100.0, 100.0 * (1.0 - mae / (mean_prev + 1e-6))))
            else:
                accuracy_pct = None
        else:
            accuracy_pct = None
    except Exception as e:
        print(f"Warning computing accuracy for FH{step_str}: {e}")
        accuracy_pct = None

    # save this run's avg for next run comparison (overwrite)
    try:
        np.save(prev_file, avg_data)
    except Exception as e:
        print(f"Warning saving prev file {prev_file}: {e}")

    # ---- PLOT PRATE AND MSLP ----
    fig = plt.figure(figsize=(10, 7), dpi=300, facecolor='white')
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-130, -65, 20, 54], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3)
    ax.add_feature(cfeature.RIVERS, linewidth=0.4, edgecolor='blue')
    ax.add_feature(cfeature.LAKES, facecolor='lightblue', edgecolor='blue', linewidth=0.3)

    if lats.ndim == 1 and lons.ndim == 1:
        Lon2d, Lat2d = np.meshgrid(lons_plot, lats)
    else:
        Lon2d, Lat2d = lons_plot, lats

    # Plot PRATE
    mesh = ax.contourf(
        Lon2d, Lat2d, avg_data,
        levels=prate_levels,
        cmap=cmap,
        norm=norm,
        extend='max',
        transform=ccrs.PlateCarree()
    )

    # Plot MSLP
    mslp_contours = ax.contour(
        Lon2d, Lat2d, avg_mslp,
        levels=mslp_levels,
        colors='black',
        linewidths=0.5,
        transform=ccrs.PlateCarree()
    )
    ax.clabel(mslp_contours, fmt='%d', fontsize=6)

    # Use PRATE intensity only where the ensemble mean categorical snow mask is present.
    snow_levels = [0.10, 0.25, 0.5, 1, 2, 4, 8, 16]
    snow_colors = [
        "#e3f2fd",  # 0.10 very light blue
        "#bbdefb",  # 0.25 light blue
        "#90caf9",  # 0.5 blue
        "#42a5f5",  # 1 medium blue
        "#1e88e5",  # 2 deeper blue
        "#1565c0",  # 4 dark blue
        "#0d47a1",  # 8 very dark blue
        "#002171",  # 16 almost navy
    ]
    snow_cmap = LinearSegmentedColormap.from_list("snow_cbar", snow_colors, N=len(snow_colors))
    snow_norm = BoundaryNorm(snow_levels, snow_cmap.N)

    # CSNOW is categorical, so keep nearest-neighbor-resized values and threshold the mean mask.
    snow_mask = avg_csnow >= 0.5
    snow_prate_masked = np.ma.masked_where(~snow_mask, avg_data)

    # draw snow overlay on top, solid blue shades (intensity from PRATE but using snow_levels)
    snow_mesh = ax.contourf(
        Lon2d, Lat2d, snow_prate_masked,
        levels=snow_levels,
        cmap=snow_cmap,
        norm=snow_norm,
        extend='max',
        transform=ccrs.PlateCarree(),
        alpha=0.85,
        zorder=(mesh.get_zorder() if 'mesh' in locals() else 4) + 1
    )

    # redraw key map features and MSLP contours on top so they remain visible over snow
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=200)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=200)
    ax.add_feature(cfeature.STATES, linewidth=0.3, zorder=200)
    ax.add_feature(cfeature.RIVERS, linewidth=0.4, edgecolor='blue', zorder=200)
    ax.add_feature(cfeature.LAKES, facecolor='lightblue', edgecolor='blue', linewidth=0.3, zorder=200)
    # re-plot MSLP contours on top of the snow overlay so they remain visible
    mslp_contours_top = ax.contour(
        Lon2d, Lat2d, avg_mslp,
        levels=mslp_levels,
        colors='black',
        linewidths=0.6,
        transform=ccrs.PlateCarree(),
        zorder=210
    )
    ax.clabel(mslp_contours_top, fmt='%d', fontsize=6)

    # Plot lows and highs
    def plot_lows_highs(ax, Lon2d, Lat2d, data, extent, min_distance=5, edge_buffer=2):
        """
        Identify and plot exactly 2 lows (minima) and 2 highs (maxima) on the map.
        Ensure no high or low is near another, and none are near the map edges.
        Ensure no two lows or highs are within the same contour level.
        """
        # Smooth the data to reduce noise
        smoothed_data = gaussian_filter(data, sigma=3)

        # Find local minima (lows) and maxima (highs)
        local_min = minimum_filter(smoothed_data, size=10) == smoothed_data
        local_max = maximum_filter(smoothed_data, size=10) == smoothed_data

        # Extract coordinates and values for lows and highs
        points = []
        lon_min, lon_max, lat_min, lat_max = extent
        for lon, lat, value, is_low in zip(
            Lon2d[local_min | local_max].flatten(),
            Lat2d[local_min | local_max].flatten(),
            data[local_min | local_max].flatten(),
            local_min[local_min | local_max].flatten()
        ):
            # Skip points near the edges of the map extent
            if not (lon_min + edge_buffer <= lon <= lon_max - edge_buffer and
                    lat_min + edge_buffer <= lat <= lat_max - edge_buffer):
                continue
            points.append((lon, lat, value, "L" if is_low else "H"))

        # Sort lows and highs by their prominence (pressure value)
        lows = sorted([p for p in points if p[3] == "L"], key=lambda x: x[2])  # Lowest pressure
        highs = sorted([p for p in points if p[3] == "H"], key=lambda x: -x[2])  # Highest pressure

        # Filter points to ensure no two highs or lows are too close to each other
        def filter_points(points, min_distance):
            filtered = []
            for p in points:
                if all(np.sqrt((p[0] - fp[0])**2 + (p[1] - fp[1])**2) > min_distance for fp in filtered):
                    filtered.append(p)
            return filtered

        # Ensure lows and highs are spaced apart from each other
        def filter_lows_highs(lows, highs, min_distance):
            filtered_lows = []
            filtered_highs = []
            for low in lows:
                if all(np.sqrt((low[0] - high[0])**2 + (low[1] - high[1])**2) > min_distance for high in highs):
                    filtered_lows.append(low)
            for high in highs:
                if all(np.sqrt((high[0] - low[0])**2 + (high[1] - low[1])**2) > min_distance for low in filtered_lows):
                    filtered_highs.append(high)
            return filtered_lows, filtered_highs

        # Ensure no two lows or highs are in the same contour level
        def filter_same_contour(points, levels):
            filtered = []
            for p in points:
                if all(abs(p[2] - fp[2]) > levels for fp in filtered):
                    filtered.append(p)
            return filtered

        lows = filter_points(lows, min_distance)
        highs = filter_points(highs, min_distance)
        lows, highs = filter_lows_highs(lows, highs, min_distance)

        # Filter lows and highs to ensure no two are in the same contour level
        contour_interval = mslp_levels[1] - mslp_levels[0]
        lows = filter_same_contour(lows, contour_interval)
        highs = filter_same_contour(highs, contour_interval)

        # FALLBACKS: pick global min/max restricted to the plotting extent (respect edge_buffer)
        valid_mask = (
            (Lon2d >= lon_min + edge_buffer) & (Lon2d <= lon_max - edge_buffer) &
            (Lat2d >= lat_min + edge_buffer) & (Lat2d <= lat_max - edge_buffer)
        )

        if len(lows) == 0:
            try:
                masked = np.where(valid_mask, smoothed_data, np.nan)
                if not np.all(np.isnan(masked)):
                    idx = np.unravel_index(np.nanargmin(masked), masked.shape)
                    lon_pt = float(Lon2d[idx]); lat_pt = float(Lat2d[idx]); val = float(data[idx])
                    lows.append((lon_pt, lat_pt, val, "L"))
            except Exception:
                pass

        if len(highs) == 0:
            try:
                masked = np.where(valid_mask, smoothed_data, np.nan)
                if not np.all(np.isnan(masked)):
                    idx = np.unravel_index(np.nanargmax(masked), masked.shape)
                    lon_pt = float(Lon2d[idx]); lat_pt = float(Lat2d[idx]); val = float(data[idx])
                    highs.append((lon_pt, lat_pt, val, "H"))
            except Exception:
                pass

        # Ensure there are exactly 2 lows and 2 highs (duplicate only if at least one exists)
        if len(lows) == 1:
            lows += [lows[0]]
        if len(highs) == 1:
            highs += [highs[0]]
        if len(lows) == 0:
            lows = []  # nothing to plot
        if len(highs) == 0:
            highs = []

        lows = lows[:2]
        highs = highs[:2]

        # Plot lows and highs with pressure values (ensure they are topmost)
        for lon, lat, value, label in lows + highs:
            color = "red" if label == "L" else "blue"
            # main label (L/H)
            txt_main = ax.text(
                lon, lat, label, color=color, fontsize=12, fontweight="bold",
                ha="center", va="center", transform=ccrs.PlateCarree(),
                zorder=400, clip_on=False
            )
            # add white stroke so text remains readable over contours/overlays
            txt_main.set_path_effects([mpe.withStroke(linewidth=2.0, foreground="white")])

            # numeric pressure below the label
            txt_val = ax.text(
                lon, lat - 1, f"{value:.0f}", color=color, fontsize=6, fontweight="bold",
                ha="center", va="center", transform=ccrs.PlateCarree(),
                zorder=400, clip_on=False
            )
            txt_val.set_path_effects([mpe.withStroke(linewidth=2.0, foreground="white")])

    plot_lows_highs(ax, Lon2d, Lat2d, avg_mslp, extent=[-130, -65, 20, 54])

    # place both horizontal colorbars way up near the top of the figure
    # [left, bottom, width, height] in figure fraction
    cbar_left = 0.12  # shifted right from 0.06 -> 0.12
    cbar_bottom = 0.10   # move bars up near the top
    cbar_width = 0.38
    cbar_height = 0.018  # compact height
    cbar_ax_prate = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cbar_ax_snow = fig.add_axes([cbar_left + cbar_width + 0.02, cbar_bottom, cbar_width, cbar_height])
    cb1 = fig.colorbar(mesh, cax=cbar_ax_prate, orientation='horizontal')
    cb1.set_label("Average Surface PRATE (GFS + GEFS3 + GEFS12 + GEFS18) mm/hr", fontsize=7)
    cb1.ax.tick_params(labelsize=6)
    cb2 = fig.colorbar(snow_mesh, cax=cbar_ax_snow, orientation='horizontal')
    cb2.set_label("Snow (avg PRATE where ensemble CSNOW is true) mm/hr", fontsize=7)
    cb2.ax.tick_params(labelsize=6)

    # place the main title above the plot
    forecast_local_time, forecast_day = format_local_time(run_date, run_hour, forecast_hour)
    accuracy_str = f" | Accuracy: {accuracy_pct:.1f}%" if accuracy_pct is not None else " | Accuracy: N/A"
    fig.suptitle(
        f"plot4 estimated precip/prate & mslp — Run: {run_date} {run_hour}Z | Forecast: {step_str} ({forecast_local_time} EST, {forecast_day}){accuracy_str}",
        fontsize=8,
        fontweight='bold',
        y=0.80
    )
    # adjust axes so the map has room above/below the title
    plt.subplots_adjust(top=0.84)

    png_path = os.path.join(BASE_DIR_AVG, "png", f"7_11_prate_mslp_avg_{step_str}.png")
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved final AVG PNG FH{step_str}: {png_path}")

    # mark this step as processed
    save_processed_step(step_str)

    # remove any grib files for this forecast hour to avoid accumulation
    try:
        deleted = []
        for fname in os.listdir(os.path.join(BASE_DIR_AVG, "grib")):
            if f"f{step_str}" in fname:
                p = os.path.join(BASE_DIR_AVG, "grib", fname)
                try:
                    os.remove(p)
                    deleted.append(p)
                except Exception as ex:
                    print(f"Failed to delete {p}: {ex}")
        if deleted:
            for d in deleted:
                print(f"Deleted GRIB: {d}")
        else:
            print(f"No GRIB files found for FH{step_str} to delete.")
    except Exception as e:
        print(f"Error scanning/deleting grib files for FH{step_str}: {e}")

    del avg_data, avg_mslp, avg_csnow
    del sum_data, sum_mslp, sum_csnow
    del data_gfs, data_gfs_mslp, data_gfs_csnow

    gc.collect()
    time.sleep(1)

print("All GFS + GEFS4 + GEFS8 + GEFS15 averages downloaded and plotted in AVG_PRATE!")
