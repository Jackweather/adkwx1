from flask import Flask, render_template, send_from_directory, abort, jsonify, url_for
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import os
import subprocess
import threading
import traceback
import getpass
import re
from pathlib import Path

from main_models_run.region_config import ACTIVE_REGION_NAMES, DEFAULT_REGION, REGION_LABELS

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

BASE_DIR = Path("/var/data") if Path("/var/data").exists() else Path(__file__).resolve().parent


def resolve_png_dir(path_spec):
    if isinstance(path_spec, Path):
        return path_spec

    for candidate in path_spec:
        if candidate.is_dir():
            return candidate

    for candidate in path_spec:
        return candidate

    return None


def run_scripts(scripts, max_workers):
    print("Flask is running as user:", getpass.getuser())  # Print user for debugging

    def run_script(script, cwd):
        try:
            result = subprocess.run(
                ["python", script],
                check=True, cwd=cwd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            return script, result, None
        except subprocess.CalledProcessError as e:
            return script, e, traceback.format_exc()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        script_iter = iter(scripts)
        active_futures = {}

        for _ in range(max_workers):
            try:
                script, cwd = next(script_iter)
            except StopIteration:
                break
            active_futures[executor.submit(run_script, script, cwd)] = script

        while active_futures:
            done, _ = wait(active_futures, return_when=FIRST_COMPLETED)
            for future in done:
                script, result, error_trace = future.result()
                active_futures.pop(future, None)

                if error_trace:
                    print(f"Error running {os.path.basename(script)}:\n{error_trace}")
                    print("STDOUT:", result.stdout)
                    print("STDERR:", result.stderr)
                else:
                    print(f"{os.path.basename(script)} ran successfully!")
                    print("STDOUT:", result.stdout)
                    print("STDERR:", result.stderr)

                try:
                    next_script, next_cwd = next(script_iter)
                except StopIteration:
                    continue

                active_futures[executor.submit(run_script, next_script, next_cwd)] = next_script

    print("All queued scripts finished. Background task ending.")

MAIN_PNG_DIRS = {
    "main_NWP": BASE_DIR / "EURO_GFS_PRATE_OUTPUT" / "png",
    "total_precip": BASE_DIR / "EURO_GFS_TOTAL_PRECIP_OUTPUT" / "png",
    "total_snow": BASE_DIR / "EURO_GFS_TOTAL_SNOW_OUTPUT" / "png",
    "temp_2m": BASE_DIR / "EURO_GFS_T2M_OUTPUT" / "png",
}

MAIN_GROUP_LABELS = {
    "main_NWP": "Main NWP",
    "total_precip": "Total Precip",
    "total_snow": "Total Snow",
    "temp_2m": "2m Temp",
}

MAIN_DEFAULT_PANEL_GROUPS = ["main_NWP"]

GEFS_PNG_DIRS = {
    "3_12_18": (
        BASE_DIR / "3_12_18_GFS_OUTPUT" / "png",
        BASE_DIR / "gefs_gfs" / "3_12_18_GFS_OUTPUT" / "png",
    ),
    "4_8_15": (
        BASE_DIR / "4_8_15_GFS_OUTPUT" / "png",
        BASE_DIR / "gefs_gfs" / "4_8_15_GFS_OUTPUT" / "png",
    ),
    "5_6_10": (
        BASE_DIR / "5_6_10_GFS_OUTPUT" / "png",
        BASE_DIR / "gefs_gfs" / "5_6_10_GFS_OUTPUT" / "png",
    ),
    "7_11": (
        BASE_DIR / "7_11_GFS_OUTPUT" / "png",
        BASE_DIR / "gefs_gfs" / "7_11_GFS_OUTPUT" / "png",
    ),
}

GEFS_GROUP_LABELS = {
    "3_12_18": "3 / 12 / 18",
    "4_8_15": "4 / 8 / 15",
    "5_6_10": "5 / 6 / 10",
    "7_11": "7 / 11",
}

GEFS_DEFAULT_PANEL_GROUPS = ["3_12_18"]

VIEWER_CONFIGS = {
    "main": {
        "title": "Main Blend",
        "endpoint": "index",
        "png_dirs": MAIN_PNG_DIRS,
        "group_labels": MAIN_GROUP_LABELS,
        "default_panel_groups": MAIN_DEFAULT_PANEL_GROUPS,
        "base_regions": ACTIVE_REGION_NAMES,
    },
    "gefs": {
        "title": "GEFS Viewer",
        "endpoint": "gefs_view",
        "png_dirs": GEFS_PNG_DIRS,
        "group_labels": GEFS_GROUP_LABELS,
        "default_panel_groups": GEFS_DEFAULT_PANEL_GROUPS,
        "base_regions": [DEFAULT_REGION],
    },
}


def build_image_entry(file_path):
    stat = file_path.stat()
    version = f"{stat.st_mtime_ns}-{stat.st_size}"
    match = re.search(r'_(\d{3})\.png$', file_path.name)
    key = match.group(1) if match else f"idx_{file_path.stem}"
    return key, {
        "filename": file_path.name,
        "version": version,
    }


def build_slide_payload(png_dirs, group_labels, default_panel_groups, base_regions=None):
    group_files = {}
    discovered_regions = []

    for group, path_spec in png_dirs.items():
        group_files[group] = {}
        path = resolve_png_dir(path_spec)
        if path is None:
            continue
        if not path.is_dir():
            continue

        for file_path in sorted(path.glob("*.png")):
            key, image_entry = build_image_entry(file_path)
            group_files[group].setdefault(DEFAULT_REGION, {})[key] = image_entry

        for region_dir in sorted(child for child in path.iterdir() if child.is_dir()):
            region_name = region_dir.name
            if region_name not in discovered_regions:
                discovered_regions.append(region_name)

            for file_path in sorted(region_dir.glob("*.png")):
                key, image_entry = build_image_entry(file_path)
                group_files[group].setdefault(region_name, {})[key] = image_entry

    all_keys = set()
    for region_map in group_files.values():
        for images_by_step in region_map.values():
            all_keys.update(images_by_step.keys())

    def key_sort(k):
        return (0, int(k)) if re.fullmatch(r'\d{3}', k) else (1, k)

    ordered_keys = sorted(all_keys, key=key_sort)
    groups = list(png_dirs.keys())
    regions = list(base_regions or [])
    if DEFAULT_REGION not in regions:
        regions.insert(0, DEFAULT_REGION)
    for region_name in discovered_regions:
        if region_name not in regions:
            regions.append(region_name)

    slides = []
    for k in ordered_keys:
        slides.append({
            "index": k,
            "images": {
                group: {region: group_files.get(group, {}).get(region, {}).get(k) for region in regions}
                for group in groups
            }
        })

    return {
        "groups": groups,
        "regions": regions,
        "slides": slides,
        "default_panel_groups": default_panel_groups,
        "default_region": DEFAULT_REGION,
        "group_labels": group_labels,
        "region_labels": {region: REGION_LABELS.get(region, region.replace('_', ' ').title()) for region in regions},
        "compare_group": default_panel_groups[0] if default_panel_groups else (groups[0] if groups else ""),
    }


def render_viewer(viewer_key):
    config = VIEWER_CONFIGS[viewer_key]
    payload = build_slide_payload(
        config["png_dirs"],
        config["group_labels"],
        config["default_panel_groups"],
        config.get("base_regions"),
    )
    viewer_links = [
        {
            "key": key,
            "title": viewer_config["title"],
            "url": f"/{'' if key == 'main' else key}",
        }
        for key, viewer_config in VIEWER_CONFIGS.items()
    ]
    return render_template(
        'index.html',
        slides=payload["slides"],
        groups=payload["groups"],
        regions=payload["regions"],
        default_panel_groups=payload["default_panel_groups"],
        default_region=payload["default_region"],
        group_labels=payload["group_labels"],
        region_labels=payload["region_labels"],
        compare_group=payload["compare_group"],
        page_title=config["title"],
        viewer_links=viewer_links,
        current_viewer=viewer_key,
        slide_data_url=url_for('slide_data_view', viewer_key=viewer_key),
    )


@app.route('/')
def index():
    return render_viewer("main")


@app.route('/gefs')
def gefs_view():
    return render_viewer("gefs")


@app.route('/slide-data')
def slide_data():
    return slide_data_view('main')


@app.route('/slide-data/<viewer_key>')
def slide_data_view(viewer_key):
    if viewer_key not in VIEWER_CONFIGS:
        abort(404)

    config = VIEWER_CONFIGS[viewer_key]
    payload = build_slide_payload(
        config["png_dirs"],
        config["group_labels"],
        config["default_panel_groups"],
        config.get("base_regions"),
    )
    response = jsonify(payload)
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/pngs/<group>/<region>/<filename>')
def serve_region_png(group, region, filename):
    dir_path = resolve_png_dir(MAIN_PNG_DIRS.get(group) or GEFS_PNG_DIRS.get(group))
    if not dir_path or not dir_path.is_dir():
        abort(404)

    region_dir = dir_path / region
    if region_dir.is_dir():
        response = send_from_directory(region_dir, filename, max_age=0)
    elif region == DEFAULT_REGION:
        response = send_from_directory(dir_path, filename, max_age=0)
    else:
        abort(404)

    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/pngs/<group>/<filename>')
def serve_png(group, filename):
    return serve_region_png(group, DEFAULT_REGION, filename)

@app.route("/run-task1")
def run_task1():
    scripts = [
        ("/opt/render/project/src/gefs_gfs/5_6_10_GFS_prate.py", "/opt/render/project/src/gefs_gfs"),
        ("/opt/render/project/src/gefs_gfs/4_8_15_GFS_prate.py", "/opt/render/project/src/gefs_gfs"),
        ("/opt/render/project/src/gefs_gfs/7_11_GFS_prate.py", "/opt/render/project/src/gefs_gfs"),
        ("/opt/render/project/src/gefs_gfs/3_12_18_GFS_prate.py", "/opt/render/project/src/gefs_gfs"),
    ]
    threading.Thread(target=lambda: run_scripts(scripts, 2)).start()
    return "Task started in background! Check logs folder for output.", 200


@app.route("/run-task2")
def run_task2():
    scripts = [
        ("/opt/render/project/src/main_models_run/prate_type_with_gfs_euro.py", "/opt/render/project/src/main_models_run"),
        ("/opt/render/project/src/main_models_run/total_precip_type_with_gfs_euro.py", "/opt/render/project/src/main_models_run"),
        ("/opt/render/project/src/main_models_run/total_snowfall_type_with_gfs_euro.py", "/opt/render/project/src/main_models_run"),
        ("/opt/render/project/src/main_models_run/temp_2m_type_with_gfs_euro.py", "/opt/render/project/src/main_models_run"),
    ]
    threading.Thread(target=lambda: run_scripts(scripts, 2)).start()
    return "Task started in background! Check logs folder for output.", 200

if __name__ == '__main__':
    app.run(debug=True)
