import os
import subprocess

log_dir = "logs/m08"
for folder in os.listdir(log_dir):
    folder_path = os.path.join(log_dir, folder)
    if os.path.isdir(folder_path):
        print(f"Processing: {folder_path}")
        subprocess.run([
            "python",
            "output_metric_calc.py",
            "--log_folder_path", folder_path
        ])