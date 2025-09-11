# /// script
# requires-python = ">=3.9"
# dependencies = [ "jatic_dota[all]" ]
# [tool.uv.sources]
# jatic_dota = { git = "https://github.com/nenb/jatic-dota.git", rev = "v0.2.0" }
# ///

import subprocess
import sys

IMG_PATH = "path/to/dota/image.png"
MODEL_NAME = "bbav"  # or "dafne"

if __name__ == "__main__":
    try:
        subprocess.run([sys.executable, "-m", "jatic_dota", "--image_path", IMG_PATH, "--model", MODEL_NAME], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running jatic_dota: {e}")