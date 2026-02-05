import os
import requests
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time
import gc
from scipy.ndimage import zoom
from datetime import datetime, timedelta
import pytz  # Add this import for timezone handling


current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR_AVG = os.path.join(current_dir, '5_6_10_GFS_OUTPUT')

# Make sure AVG subfolders exist
for sub in ['grib', 'png']:
    os.makedirs(os.path.join(BASE_DIR_AVG, sub), exist_ok=True)

# -----------------------------
# CLEAR ONLY AVG PNGs
# -----------------------------
avg_png_dir = os.path.join(BASE_DIR_AVG, "png")
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
forecast_steps = list(range(0, 187, 6))  # 0 → 186

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

# -----------------------------
# URL BASES
# -----------------------------
base_url_gfs = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
base_url_gefs = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p50b.pl"
gefs_members = ["05", "06", "10"]  # members to include

# -----------------------------
# PROCESS FORECAST HOURS
# -----------------------------
for step in forecast_steps:
    step_str = f"{step:03d}"

    gefs_data_list = []

    # ---- GFS ----
    while True:  # Retry logic for unavailable forecast hours
        gfs_file = f"gfs.t{run_hour}z.pgrb2.0p25.f{step_str}_prate.grib2"
        gfs_path = os.path.join(BASE_DIR_AVG, "grib", gfs_file)
        gfs_url = (
            f"{base_url_gfs}?file=gfs.t{run_hour}z.pgrb2.0p25.f{step_str}"
            f"&var_PRATE=on&lev_surface=on"
            f"&subregion=&leftlon=220&rightlon=300&toplat=55&bottomlat=20"
            f"&dir=%2Fgfs.{run_date}%2F{run_hour}%2Fatmos"
        )

        # Download only if not exist
        if not os.path.exists(gfs_path):
            print(f"Downloading GFS FH{step_str} …")
            r = requests.get(gfs_url, stream=True)
            if r.status_code == 200:
                with open(gfs_path, 'wb') as f:
                    for chunk in r.iter_content(1024*64):
                        if chunk:
                            f.write(chunk)
                print(f"Saved GFS GRIB: {gfs_path}")
                break  # Exit the retry loop if download is successful
            else:
                print(f"Failed to download GFS {gfs_file}, status code: {r.status_code}")
                # Retry with the previous run if download fails
                run_date, run_hour = find_valid_run()
                continue
        else:
            print(f"GFS FH{step_str} already exists, skipping download.")
            break

    # Open GFS
    try:
        ds_gfs = xr.open_dataset(gfs_path, engine="cfgrib", filter_by_keys={'stepType':'avg'})
        data_gfs = ds_gfs['prate'].values * 3600  # mm/hr
        lats = ds_gfs['latitude'].values
        lons = ds_gfs['longitude'].values
        lons_plot = np.where(lons > 180, lons - 360, lons)
    except Exception as e:
        print(f"Failed to open GFS GRIB FH{step_str}: {e}")
        continue

    gefs_data_list.append(data_gfs.squeeze())

    # ---- GEFS MEMBERS ----
    for member in gefs_members:
        while True:  # Retry logic for unavailable GEFS members
            gefs_file = f"gep{member}.t{run_hour}z.pgrb2b.0p50.f{step_str}_prate.grib2"
            gefs_path = os.path.join(BASE_DIR_AVG, "grib", gefs_file)
            gefs_url = (
                f"{base_url_gefs}?file=gep{member}.t{run_hour}z.pgrb2b.0p50.f{step_str}"
                f"&var_PRATE=on&lev_surface=on"
                f"&subregion=&leftlon=220&rightlon=300&toplat=55&bottomlat=20"
                f"&dir=%2Fgefs.{run_date}%2F{run_hour}%2Fatmos%2Fpgrb2bp5"
            )

            # Download only if not exist
            if not os.path.exists(gefs_path):
                print(f"Downloading GEFS member {member}, FH{step_str} …")
                r = requests.get(gefs_url, stream=True)
                if r.status_code == 200:
                    with open(gefs_path, 'wb') as f:
                        for chunk in r.iter_content(1024*64):
                            if chunk:
                                f.write(chunk)
                    print(f"Saved GEFS GRIB: {gefs_path}")
                    break  # Exit the retry loop if download is successful
                else:
                    print(f"Failed to download GEFS {gefs_file}, status code: {r.status_code}")
                    # Retry with the previous run if download fails
                    run_date, run_hour = find_valid_run()
                    continue
            else:
                print(f"GEFS member {member} FH{step_str} already exists, skipping download.")
                break

        # Open GEFS
        try:
            ds_gefs = xr.open_dataset(gefs_path, engine="cfgrib", filter_by_keys={'stepType':'avg'})
            data_gefs = ds_gefs['prate'].values * 3600
        except Exception as e:
            print(f"Failed to open GEFS member {member}, FH{step_str}: {e}")
            continue

        # Resize GEFS to match GFS grid if needed
        if data_gefs.shape != data_gfs.shape:
            zoom_factors = (data_gfs.shape[0]/data_gefs.shape[0], data_gfs.shape[1]/data_gefs.shape[1])
            data_gefs_resized = zoom(data_gefs.squeeze(), zoom_factors, order=1)
        else:
            data_gefs_resized = data_gefs.squeeze()

        gefs_data_list.append(data_gefs_resized)

    # ---- COMPUTE FINAL AVERAGE ----
    avg_data = np.mean(gefs_data_list, axis=0)

    # ---- PLOT ONLY AVERAGE ----
    fig = plt.figure(figsize=(10,7), dpi=300, facecolor='white')
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

    mesh = ax.contourf(
        Lon2d, Lat2d, avg_data,
        levels=prate_levels,
        cmap=cmap,
        norm=norm,
        extend='max',
        transform=ccrs.PlateCarree()
    )

    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.01,
                        aspect=25, shrink=0.65)
    cbar.set_label("Average Surface PRATE (GFS + GEFS5 + GEFS6 + GEFS10) mm/hr", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Include run information in the title
    plt.title(
        f"Average GFS + GEFS5 + GEFS6 + GEFS10 PRATE FH {step_str}\nRun: {run_date} {run_hour}Z",
        fontsize=12,
        fontweight='bold'
    )

    png_path = os.path.join(BASE_DIR_AVG, "png", f"avg_prate_all_{step_str}.png")
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved final AVG PNG FH{step_str}: {png_path}")

    gc.collect()
    time.sleep(1)

print("All GFS + GEFS5 + GEFS6 + GEFS10 averages downloaded and plotted in AVG_PRATE!")
