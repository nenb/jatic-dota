import argparse
from jatic_dota.inference import dota_inference
from jatic_dota.plot import plot_bounding_boxes
import os
from jatic_dota.log import logger
from PIL import Image
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Run DOTA inference and plot results.")
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to the image file. If not provided, a default image will be used.",
    )

    args = parser.parse_args()
    
    try:
        img = Image.open(args.image_path)
        kwargs = {"img_arr": np.array(img)}
    except Exception:
        logger.warning(f"Error opening image path. Using default image instead...")
        kwargs = {}

    img, results = dota_inference(**kwargs)
    plot_bounding_boxes(img, results)

if __name__ == "__main__":
    main()
