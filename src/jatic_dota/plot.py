import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .log import logger
import tempfile
import os
import hashlib

def get_save_path(img_hash):
    tmp_dir = tempfile.gettempdir()
    filename = f"{img_hash}.png"
    file_path = os.path.join(tmp_dir, filename)
    return file_path


def plot_bounding_boxes(img, detections, save=True):
    logger.info("Generating plot of predictions ...")    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)

    category_colors = {
        0: "red",
        1: "blue",
        2: "green",
        3: "purple",
        4: "orange",
        5: "cyan",
        6: "magenta",
        7: "yellow",
        8: "brown",
        9: "pink",
        10: "gray",
        11: "olive",
        12: "lime",
        13: "teal",
        14: "navy",
    }

    for detection in detections:
        category = detection.category
        bbox = detection.polygon

        polygon_points = [(bbox[i], bbox[i + 1]) for i in range(0, len(bbox), 2)]

        color = category_colors.get(category, "black")

        poly = patches.Polygon(
            polygon_points, closed=True, edgecolor=color, facecolor="none"
        )
        ax.add_patch(poly)
    
    if save:
        array_hash = hashlib.md5(img.tobytes()).hexdigest()
        save_path = get_save_path(array_hash)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Plot containing model predictions saved to {save_path}!")
    
    else:
        return fig, ax
