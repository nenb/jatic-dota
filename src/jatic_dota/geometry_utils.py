from shapely.geometry import Polygon
import numpy as np
from dataclasses import dataclass


@dataclass
class Detection:
    """
    Represents an oriented object detection.

    This dataclass stores the oriented bounding box polygon coordinates,
    confidence score, and category of a detected object.
    """

    polygon: list[float]  # (x1, y1, x2, y2, x3, y3, x4, y4)
    score: float
    category: int


def iou_poly_shapely(p: list[float], q: list[float]) -> float:
    """
    Calculates the Intersection over Union (IoU) of two polygons using shapely.

    Args:
        p: A list of floats representing the vertices of the first polygon [x1, y1, x2, y2, ...].
        q: A list of floats representing the vertices of the second polygon [x1, y1, x2, y2, ...].

    Returns:
        The IoU value.
    """
    poly1 = Polygon([(p[i], p[i + 1]) for i in range(0, len(p), 2)])
    poly2 = Polygon([(q[i], q[i + 1]) for i in range(0, len(q), 2)])

    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area

    return intersection_area / union_area if union_area else 0.0

def nms_obb_fast(detections: list[Detection], thresh: float = 0.1) -> list[Detection]:
    from dataclasses import dataclass
    from shapely.geometry import Polygon
    from shapely.prepared import prep
    polys = [
        Polygon(list(zip(det.polygon[0::2], det.polygon[1::2])))
        for det in detections
    ]
    areas = np.array([p.area for p in polys], dtype=float)
    bounds = np.array([p.bounds for p in polys], dtype=float)  # (minx, miny, maxx, maxy)
    scores = np.array([d.score for d in detections], dtype=float)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        others = order[1:]
        if others.size == 0:
            break

        b = bounds[i]
        cand_mask = (
            (bounds[others, 0] <= b[2]) & (bounds[others, 2] >= b[0]) &
            (bounds[others, 1] <= b[3]) & (bounds[others, 3] >= b[1])
        )
        if not cand_mask.any():
            order = others
            continue

        candidates = others[cand_mask]

        prep_i = prep(polys[i])
        inters_mask = np.array([prep_i.intersects(polys[j]) for j in candidates])
        if not inters_mask.any():
            order = others
            continue

        near = candidates[inters_mask]

        suppress = set()
        ai = areas[i]
        for j in near:
            inter = polys[i].intersection(polys[j]).area
            if inter <= 0.0:
                continue
            iou = inter / (ai + areas[j] - inter)
            if iou > thresh:
                suppress.add(j)

        keep_mask = np.array([j not in suppress for j in others], dtype=bool)
        order = others[keep_mask]

    return [detections[k] for k in keep]


def nms_obb(detections: list[Detection], thresh: float = 0.1) -> list[Detection]:
    """
    Performs Non-Maximum Suppression (NMS) on oriented bounding box detections.

    NMS keeps the bounding box with the maximum confidence and suppresses all
    other lower-confidence boxes that overlap with it according to an Intersection
    over Union (IoU) threshold. NMS process proceeds iteratively until all boxes
    have been processed, resulting in a set of non-overlapping, maximum-confidence
    bouning box detections.

    Args:
        detections: A list of Detection objects.
        thresh: The IoU threshold for NMS. Detections with IoU above this threshold
                will be suppressed.

    Returns:
        A list of Detection objects that were kept after NMS.
    """
    polys = [d.polygon for d in detections]
    scores = np.array([d.score for d in detections])
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        best_index = order[0]
        keep.append(best_index)

        ious = np.array(
            [
                iou_poly_shapely(polys[best_index], polys[other_index])
                for other_index in order[1:]
            ]
        )

        order = order[1:][ious <= thresh]

    return [detections[i] for i in keep]


def apply_nms_patch(
    detections_by_category: dict[int, list[Detection]],
) -> list[Detection]:
    """
    Applies Non-Maximum Suppression (NMS) across all categories on an individual patch.

    This function first applies NMS within each category and then applies NMS
    across all resulting detections to remove any remaining overlaps.

    Args:
        detections_by_category: A dictionary where keys are category IDs and
                                values are lists of Detection objects.

    Returns:
        A list of Detection objects that were kept after NMS.
    """
    all_detections_after_category_nms: list[Detection] = []

    for detections in detections_by_category.values():
        all_detections_after_category_nms.extend(nms_obb_fast(detections))

    return nms_obb_fast(all_detections_after_category_nms)
