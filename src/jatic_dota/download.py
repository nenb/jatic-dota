from pathlib import Path
import httpx
from tqdm import tqdm

from .log import logger  # Assuming you have a logger module

MODEL_URLS = {
    "model_50": "https://github.com/nenb/jatic-dota/releases/download/v0.0.1/model_50.pth",
}


def download_pickle_to_file(filepath: Path, model_name: str):
    """
    Downloads a pickle file from the given URL and saves it to the specified filepath.
    """
    url = MODEL_URLS[model_name]

    with httpx.stream("GET", url, follow_redirects=True) as response:
        response.raise_for_status()
        total_length = int(response.headers.get("Content-Length", 0))
        chunk_size = 1024

        progress_bar = tqdm(
            total=total_length, unit="iB", unit_scale=True, desc="Downloading DOTA model ..."
        )

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            for chunk in response.iter_bytes(chunk_size=chunk_size):
                f.write(chunk)
                progress_bar.update(len(chunk))

        progress_bar.close()

    logger.info(f"Pickle file downloaded and saved to {filepath} .")
