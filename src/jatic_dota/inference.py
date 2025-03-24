import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
import math
from tqdm import tqdm

from .decoder import DecDecoder
from .geometry_utils import Detection, apply_nms_patch, nms_obb
from .models import ctrbox_net
from .download import download_pickle_to_file
from .log import logger

DOWN_RATIO = 4

CATEGORIES = {
    0: "plane",
    1: "baseball-diamond",
    2: "bridge",
    3: "ground-track-field",
    4: "small-vehicle",
    5: "large-vehicle",
    6: "ship",
    7: "tennis-court",
    8: "basketball-court",
    9: "storage-tank",
    10: "soccer-ball-field",
    11: "roundabout",
    12: "harbor",
    13: "swimming-pool",
    14: "helicopter",
}


@dataclass
class Patch:
    """
    Represents a square 600x600 region extracted from an image.

    This dataclass stores the image data of the patch and the coordinates
    of its bottom-left corner within the original image from which it was extracted.
    """

    image: torch.Tensor
    x_offset: int
    y_offset: int


def prepare_patch_for_model(
    image_patch: npt.NDArray, device: torch.device
) -> torch.Tensor:
    """
    Transforms an image patch into the format expected by a model.

    This function performs the following transformations:
    1. Normalizes the image patch to the range [-0.5, 0.5].
    2. Rearranges the dimensions from Height x Width x Channels (HWC) to Channels x Height x Width (CHW).
    3. Adds a batch dimension of size 1 at the beginning, resulting in a shape of (1, C, H, W).
    4. Converts the NumPy array to a PyTorch tensor and moves it to the specified device.

    Args:
        image_patch: The image patch to prepare (HWC format).
        patch_size: The height and width of the patch.
        device: The device to move the tensor to.

    Returns:
        The prepared image patch tensor (1, C, H, W).
    """
    image_patch = image_patch.astype(np.float32) / 255.0 - 0.5
    image_patch = image_patch.transpose(2, 0, 1)  # HWC to CHW
    image_patch = np.expand_dims(image_patch, axis=0)
    return torch.from_numpy(image_patch).to(device)


def preprocess_image(img_arr: npt.NDArray, device: torch.device) -> list[Patch]:
    """
    Splits an image into patches and preprocesses them for model input.

    This function divides an input image into overlapping patches of size 600x600.
    If the image's height or width is larger than the patch size, the image is split
    into multiple overlapping patches. Otherwise, the entire image is treated as a
    single patch.

    Args:
        img_arr: The input image as a NumPy array (HWC format).
        device: The PyTorch device to move the preprocessed patches to.

    Returns:
        A list of Patch objects, where each Patch contains the preprocessed image
        patch and its x and y offsets within the original image.
    """

    H, W, _ = img_arr.shape
    patches = []
    patch_size = 600
    overlap = 100
    stride = patch_size - overlap

    if H > patch_size or W > patch_size:
        x_positions = list(range(0, W - patch_size + 1, stride))
        if x_positions[-1] + patch_size < W:
            x_positions.append(W - patch_size)

        y_positions = list(range(0, H - patch_size + 1, stride))
        if y_positions[-1] + patch_size < H:
            y_positions.append(H - patch_size)

        logger.info(
            f"Image is too large. Splitting into {len(y_positions) * len((x_positions))} patches ..."
        )

        for y in y_positions:
            for x in x_positions:
                img = prepare_patch_for_model(
                    img_arr[y : y + patch_size, x : x + patch_size], device
                )
                patches.append(Patch(image=img, x_offset=x, y_offset=y))

    else:
        img = prepare_patch_for_model(img_arr, device)
        patches.append(Patch(image=img, x_offset=0, y_offset=0))

    return patches


def extract_detections(
    predictions: npt.NDArray, patch: Patch
) -> dict[int, list[Detection]]:
    """
    Extracts detection information from model predictions and organizes it by category.

    This function processes model predictions, converting them into Detection objects
    containing polygon coordinates, confidence scores, and category IDs. The detections
    are adjusted based on the downsampling ratio and the patch's offset in the original
    image. The extracted detections are then organized into a dictionary where keys are
    category IDs and values are lists of Detection objects.

    Args:
        predictions: The model's prediction output as a NumPy array. Each prediction is
            expected to contain center coordinates, bounding box offsets, a score, and
            a class ID.
        patch: A Patch object containing the patch's offset within the original image.

    Returns:
        A dictionary where keys are category IDs and values are lists of Detection
        objects.
    """
    detections_by_category: dict[int, list[Detection]] = {id_: [] for id_ in CATEGORIES}

    for pred in predictions:
        cx, cy, tx, ty, rx, ry, bx, by, lx, ly, score, clse = pred

        center = np.array([cx, cy], dtype=np.float64)
        top = np.array([tx, ty], dtype=np.float64)
        right = np.array([rx, ry], dtype=np.float64)
        bottom = np.array([bx, by], dtype=np.float64)
        left = np.array([lx, ly], dtype=np.float64)

        top_left = top + left - center
        bottom_left = bottom + left - center
        top_right = top + right - center
        bottom_right = bottom + right - center

        points = np.array(
            [top_right, bottom_right, bottom_left, top_left], dtype=np.float64
        )
        points *= DOWN_RATIO

        bbox = [
            points[0, 0].item() + patch.x_offset,
            points[0, 1].item() + patch.y_offset,
            points[1, 0].item() + patch.x_offset,
            points[1, 1].item() + patch.y_offset,
            points[2, 0].item() + patch.x_offset,
            points[2, 1].item() + patch.y_offset,
            points[3, 0].item() + patch.x_offset,
            points[3, 1].item() + patch.y_offset,
        ]

        category_id = int(clse)
        detections_by_category[category_id].append(
            Detection(polygon=bbox, score=score, category=category_id)
        )

    return detections_by_category


def postprocess_predictions(predictions: npt.NDArray, patch: Patch) -> list[Detection]:
    """
    Postprocesses model predictions by extracting detections and applying Non-Maximum Suppression (NMS).

    Args:
        predictions: The model's prediction output.
        patch: A Patch object representing the image patch.

    Returns:
        A list of Detection objects after NMS.
    """
    detections_by_category = extract_detections(predictions, patch)
    return apply_nms_patch(detections_by_category)


def get_device() -> torch.device:
    """Returns the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def initialize_model(
    device: torch.device = get_device(),
    weights_dir: Path = Path("~/.cache/dota").expanduser().resolve(),
    model_name: str = "model_50",
) -> ctrbox_net.CTRBOX:
    """
    Initializes the CTRBOX model with pretrained weights and sets it to evaluation mode.

    Checks if the specified `weights_dir` exists. If not, it downloads and saves the
    pretrained weights.

    See https://arxiv.org/pdf/2008.07043 for details on the CTRBOX model.

    Args:
        device: The device to load the model onto. Defaults to the best available device.
        weights_dir: Path to the model weights.

    Returns:
        The initialized CTRBOX model.
    """
    filepath = Path(f"{weights_dir}/{model_name}.pth")
    if not os.path.exists(filepath):
        download_pickle_to_file(filepath=filepath, model_name=model_name)

    logger.info(f"Loading DOTA model onto {device} ...")
    model = ctrbox_net.CTRBOX(down_ratio=DOWN_RATIO)
    checkpoint = torch.load(filepath, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    logger.info(f"Loaded DOTA model onto {device}!")
    return model


def _get_random_image():
    """
    Retrieves a random image from the specified resources directory.
    """

    image_files = [f for f in os.listdir("resources") if f.lower().endswith((".png",))]

    random_image_file = random.choice(image_files)
    image_path = os.path.join("resources", random_image_file)

    img = Image.open(image_path)
    return np.array(img)


def dota_inference(
    img_arr: npt.NDArray = _get_random_image(),
    model: ctrbox_net.CTRBOX = initialize_model(),
    batch_size: int = 4,
    num_keypoints: int = 500,
    conf_thresh: float = 0.1,
) -> tuple[npt.NDArray, list[Detection]]:
    """
    Performs inference on an image from the DOTA dataset using the CTRBOX model. If
    the image is too large, it is processed in patches.

    This function preprocesses an input image, divides it into patches, performs
    inference using the model, decodes the predictions, and postprocesses the
    results.

    Args:
        img_arr: The input image as a NumPy array.
        model: The CTRBOX model to use for inference. Defaults to initialize_model().
        batch_size: The batch size for processing patches. Defaults to 4.
        num_keypoints: The number of keypoints for decoding predictions. Defaults to 500.
        conf_thresh: The confidence threshold for detections. Defaults to 0.1.

    Returns:
        A list of Detection objects
    """
    # TODO: how important is this specific seed?
    torch.manual_seed(317)

    device = next(model.parameters()).device

    patches = preprocess_image(img_arr, device)

    patch_prediction_pairs: list[tuple[Patch, npt.NDArray]] = []
    num_batches = math.ceil(len(patches) / batch_size)
    for i in tqdm(
        range(0, len(patches), batch_size),
        total=num_batches,
        desc="Processing image patches",
        unit="batch",
    ):
        batch_patches = patches[i : i + batch_size]
        with torch.no_grad():
            centrenet_encoded_preds = model(
                torch.cat([patch.image for patch in batch_patches], dim=0)
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        decoded_predictions = DecDecoder(
            K=num_keypoints, conf_thresh=conf_thresh
        ).ctdet_decode(centrenet_encoded_preds)

        patch_prediction_pairs.extend(
            [
                (patch, bbox_pred)
                for patch, bbox_pred in zip(batch_patches, decoded_predictions)
            ]
        )

    results = []
    for patch, prediction in patch_prediction_pairs:
        results.extend(postprocess_predictions(prediction, patch))

    # this final NMS is required as patches overlap and this can lead to overlapping
    # bounding boxes that come from separate patches
    results_nms = nms_obb(results)

    return img_arr, results_nms
