from flask import Flask, render_template, send_from_directory, abort
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import subprocess
import threading
import traceback
import getpass
import re

app = Flask(__name__)

# Add new groups for 3_12_18 and 7_11 outputs
PNG_DIRS = {
    "5_6_10": os.path.join(os.getcwd(), "gefs_gfs", "5_6_10_GFS_OUTPUT", "png"),
    "4_8_15": os.path.join(os.getcwd(), "gefs_gfs", "4_8_15_GFS_OUTPUT", "png"),
    "3_12_18": os.path.join(os.getcwd(), "gefs_gfs", "3_12_18_GFS_OUTPUT", "png"),
    "7_11": os.path.join(os.getcwd(), "gefs_gfs", "7_11_GFS_OUTPUT", "png"),
}

@app.route('/')
def index():
    # build per-group filename map keyed by 3-digit forecast index (e.g. "006")
    group_files = {}
    for group, path in PNG_DIRS.items():
        group_files[group] = {}
        if os.path.isdir(path):
            for f in sorted([x for x in os.listdir(path) if x.endswith('.png')]):
                m = re.search(r'_(\d{3})\.png$', f)
                if m:
                    key = m.group(1)
                    group_files[group][key] = f
                else:
                    # fall back to numeric ordering if no index found
                    idx = f"idx_{len(group_files[group])}"
                    group_files[group][idx] = f

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
    return send_from_directory(dir_path, filename)

@app.route("/run-task1")
def run_task1():
    def run_all_scripts():
        print("Flask is running as user:", getpass.getuser())  # Print user for debugging
        scripts = [
            ("/opt/render/project/src/gefs_gfs/5_6_10_GFS_prate.py", "/opt/render/project/src/gefs_gfs"),
            ("/opt/render/project/src/gefs_gfs/4_8_15_GFS_prate.py", "/opt/render/project/src/gefs_gfs"),
            ("/opt/render/project/src/gefs_gfs/7_11_GFS_prate.py", "/opt/render/project/src/gefs_gfs"),
            ("/opt/render/project/src/gefs_gfs/3_12_18_GFS_prate.py", "/opt/render/project/src/gefs_gfs"),
        ]

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

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(run_script, script, cwd) for script, cwd in scripts]
            for future in as_completed(futures):
                script, result, error_trace = future.result()
                if error_trace:
                    print(f"Error running {os.path.basename(script)}:\n{error_trace}")
                    print("STDOUT:", result.stdout)
                    print("STDERR:", result.stderr)
                else:
                    print(f"{os.path.basename(script)} ran successfully!")
                    print("STDOUT:", result.stdout)
                    print("STDERR:", result.stderr)

    # Run the task in a separate thread
    threading.Thread(target=run_all_scripts).start()
    return "Task started in background! Check logs folder for output.", 200

if __name__ == '__main__':
    app.run(debug=True)
