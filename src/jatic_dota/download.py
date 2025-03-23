import io
import zipfile
from pathlib import Path

import httpx
from tqdm import tqdm

from .log import logger

MODEL_URLS = {
    "model_50": "https://www.dropbox.com/scl/fo/jo1mfnz7dbocep5vusdgb/ANY07_g56yYA3UukilSgIcs/model_50.zip?rlkey=45mxubsaoayymtjah9c51q0ov&dl=1",
    "model_43": "https://www.dropbox.com/scl/fo/jo1mfnz7dbocep5vusdgb/AEansoxtiwm65LjtflrQAxU/model_43.zip?rlkey=45mxubsaoayymtjah9c51q0ov&dl=1",
}


def download_and_unzip_in_memory(extract_dir: Path, model_name: str):
    """
    Streams a ZIP file from the given URL into memory and extracts
    it directly to 'extract_dir' without writing a ZIP file to disk.
    """
    url = MODEL_URLS[model_name]
    with httpx.stream("GET", url, follow_redirects=True) as response:
        total_length = int(response.headers.get("Content-Length", 0))
        chunk_size = 1024

        progress_bar = tqdm(
            total=total_length, unit="iB", unit_scale=True, desc="Downloading DOTA model ..."
        )

        in_memory_zip = io.BytesIO()

        for chunk in response.iter_bytes(chunk_size=chunk_size):
            in_memory_zip.write(chunk)
            progress_bar.update(len(chunk))

        progress_bar.close()

    in_memory_zip.seek(0)

    with zipfile.ZipFile(in_memory_zip, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    logger.info(f"Files extracted to {extract_dir} .")
