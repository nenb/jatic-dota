import argparse
from jatic_dota.inference import bbav_inference, dafne_inference
from jatic_dota.plot import plot_bounding_boxes
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
    parser.add_argument(
        "--model",
        type=str,
        default="bbav",
        choices=["bbav", "dafne"],
        help="Model to use for inference. Choose from 'bbav' or 'dafne'. Default is 'bbav'.",
    )

    args = parser.parse_args()

    try:
        img = Image.open(args.image_path)
        kwargs = {"img_arr": np.array(img)}
    except Exception:
        logger.warning(f"Error opening image path. Using default image instead...")
        kwargs = {}

    if args.model == "bbav":
        img, results = bbav_inference(**kwargs)
    elif args.model == "dafne":
        img, results = dafne_inference(**kwargs)
    else:
        raise Exception("Invalid model selected for inference.")
    
    plot_bounding_boxes(img, results)


if __name__ == "__main__":
    main()
