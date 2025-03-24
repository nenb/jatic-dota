# /// script
# requires-python = ">=3.9"
# dependencies = [ "jatic_dota" ]
# [tool.uv.sources]
# jatic_dota = { git = "https://github.com/nenb/jatic-dota.git", rev = "v0.1.0" }
# ///

import subprocess
import sys

IMG_PATH = "path/to/dota/image.png"

if __name__ == "__main__":
    try:
        subprocess.run([sys.executable, "-m", "jatic_dota", "--image_path", IMG_PATH], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running jatic_dota: {e}")