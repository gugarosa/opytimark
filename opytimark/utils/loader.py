"""Auxiliary files loader.
"""

import os
import tarfile
import urllib.request

import numpy as np

import opytimark.utils.constants as c


def download_file(url: str, output_path: str) -> None:
    """Downloads a file given its URL and the output path to be saved.

    Args:
        url: URL to download the file.
        output_path: Path to save the downloaded file.

    """

    file_exists = os.path.exists(output_path)

    if not file_exists:
        folder_exists = os.path.exists(c.DATA_FOLDER)

        if not folder_exists:
            os.mkdir(c.DATA_FOLDER)

        urllib.request.urlretrieve(url, output_path)


def untar_file(file_path: str) -> str:
    """De-compress a file with .tar.gz.

    Args:
        file_path: Path of the file to be de-compressed.

    Returns:
        (str): The folder that has been de-compressed.

    """

    with tarfile.open(file_path, "r:gz") as tar:
        folder_path = file_path.split(".tar.gz")[0]
        folder_path_exists = os.path.exists(folder_path)

        if not folder_path_exists:
            tar.extractall(path=folder_path)

    return folder_path


def load_cec_auxiliary(name: str, year: str) -> np.ndarray:
    """Loads auxiliary data for CEC-based benchmarking functions.

    Args:
        name: Name of function to be loaded.
        year: Year of function to be loaded.

    Returns:
        (np.ndarray): Auxiliary data.

    """

    # Defines some common-use variables
    base_url = "http://recogna.tech/files/opytimark/"
    tar_name = f"{year}.tar.gz"
    tar_path = f"data/{tar_name}"

    # Downloads the file
    download_file(base_url + tar_name, tar_path)

    # De-compresses the file
    folder_path = untar_file(tar_path)

    # Loads the auxiliary data
    data = np.loadtxt(f"{folder_path}/{name}.txt")

    return data
