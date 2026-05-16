from flask import Flask, render_template, send_from_directory, abort
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import os
import subprocess
import threading
import traceback
import getpass
import re
from pathlib import Path

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

BASE_DIR = Path("/var/data")


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

# Add new groups for 3_12_18 and 7_11 outputs
PNG_DIRS = {
    "5_6_10": BASE_DIR / "5_6_10_GFS_OUTPUT" / "png",
    "4_8_15": BASE_DIR / "4_8_15_GFS_OUTPUT" / "png",
    "3_12_18": BASE_DIR / "3_12_18_GFS_OUTPUT" / "png",
    "7_11": BASE_DIR / "7_11_GFS_OUTPUT" / "png",
}

@app.route('/')
def index():
    # build per-group filename map keyed by 3-digit forecast index (e.g. "006")
    group_files = {}
    for group, path in PNG_DIRS.items():
        group_files[group] = {}
        if os.path.isdir(path):
            for f in sorted([x for x in os.listdir(path) if x.endswith('.png')]):
                file_path = os.path.join(path, f)
                version = str(int(os.path.getmtime(file_path)))
                m = re.search(r'_(\d{3})\.png$', f)
                if m:
                    key = m.group(1)
                    group_files[group][key] = {
                        "filename": f,
                        "version": version,
                    }
                else:
                    # fall back to numeric ordering if no index found
                    idx = f"idx_{len(group_files[group])}"
                    group_files[group][idx] = {
                        "filename": f,
                        "version": version,
                    }

    # determine ordered union of indices (prefer numeric 3-digit keys)
    all_keys = set()
    for gmap in group_files.values():
        all_keys.update(gmap.keys())

    # try to sort numeric 3-digit keys first, then non-numeric
    def key_sort(k):
        return (0, int(k)) if re.fullmatch(r'\d{3}', k) else (1, k)
    ordered_keys = sorted(all_keys, key=key_sort)

    # create slides pairing four groups (use groups order from PNG_DIRS)
    groups = list(PNG_DIRS.keys())
    slides = []
    for k in ordered_keys:
        slide = {
            "index": k,
            "images": {group: group_files.get(group, {}).get(k) for group in groups}
        }
        slides.append(slide)

    return render_template('index.html', slides=slides, groups=groups)

@app.route('/pngs/<group>/<filename>')
def serve_png(group, filename):
    dir_path = PNG_DIRS.get(group)
    if not dir_path or not os.path.isdir(dir_path):
        abort(404)
    response = send_from_directory(dir_path, filename, max_age=0)
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

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

if __name__ == '__main__':
    app.run(debug=True)
