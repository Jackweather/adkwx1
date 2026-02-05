from flask import Flask, render_template, send_from_directory
import os
import subprocess
import threading
import traceback
import getpass

app = Flask(__name__)

PNG_DIR = os.path.join(os.getcwd(), "GEFS_GFS_OUTPUT", "static", "png")

@app.route('/')
def index():
    # Get a sorted list of PNG files in the directory
    images = sorted([file for file in os.listdir(PNG_DIR) if file.endswith('.png')])
    return render_template('index.html', images=images)

@app.route('/pngs/<filename>')
def serve_png(filename):
    # Serve PNG files directly from the existing directory
    return send_from_directory(PNG_DIR, filename)

@app.route("/run-task1")
def run_task1():
    def run_all_scripts():
        print("Flask is running as user:", getpass.getuser())  # Print user for debugging
        scripts = [
            ("/opt/render/project/src/gefs_gfs/5_6_10_GFS_prate.py", "/opt/render/project/src/gefs_gfs"),
        ]
        for script, cwd in scripts:
            try:
                result = subprocess.run(
                    ["python", script],
                    check=True, cwd=cwd,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                print(f"{os.path.basename(script)} ran successfully!")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
            except subprocess.CalledProcessError as e:
                error_trace = traceback.format_exc()
                print(f"Error running {os.path.basename(script)}:\n{error_trace}")
                print("STDOUT:", e.stdout)
                print("STDERR:", e.stderr)

    # Run the task in a separate thread
    threading.Thread(target=run_all_scripts).start()
    return "Task started in background! Check logs folder for output.", 200

if __name__ == '__main__':
    app.run(debug=True)
