import sys
try:
    import detectron2  # user-provided install
except Exception:
    from jatic_dota._vendor import detectron2 as _vendored_d2
    sys.modules.setdefault("detectron2", _vendored_d2)

from jatic_dota.inference import bbav_inference, dafne_inference, initialize_bbav_model, initialize_dafne_model
from jatic_dota.plot import plot_bounding_boxes

__all__ = ["bbav_inference", "dafne_inference", "initialize_dafne_model", "initialize_bbav_model", "plot_bounding_boxes"]
