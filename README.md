## `jatic_dota`

A standalone distribution of [BBAVectors-Oriented-Object-Detection](https://github.com/yijingru/BBAVectors-Oriented-Object-Detection/tree/11dad9aefab65f68f9330824980ee0c08bdb7234) and [DAFNe](https://github.com/braun-steven/DAFNe) for inference on the DOTAv1 dataset.

## Quickstart

1. Install `uv`: https://docs.astral.sh/uv/getting-started/installation/.

2. Run `uv run script.py`.

## Release

```shell
python -m venv .venv
source .venv/bin/activate
python -m pip install build wheel setuptools torch==2.4.1 torchvision==0.19.1 # detectron2 is funky
python -m build --wheel
```