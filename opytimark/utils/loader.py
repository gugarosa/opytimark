"""Auxiliary files loader.
"""

import os
import tarfile
import urllib.request

import numpy as np

import opytimark.utils.constants as c


def download_file(url, output_path):
    """Downloads a file given its URL and the output path to be saved.

    Args:
        url (str): URL to download the file.
        output_path (str): Path to save the downloaded file.

    """

    file_exists = os.path.exists(output_path)

    if not file_exists:
        folder_exists = os.path.exists(c.DATA_FOLDER)

        if not folder_exists:
            os.mkdir(c.DATA_FOLDER)

        urllib.request.urlretrieve(url, output_path)


def untar_file(file_path):
    """De-compress a file with .tar.gz.

    Args:
        file_path (str): Path of the file to be de-compressed.

    Returns:
        The folder that has been de-compressed.

    """

    with tarfile.open(file_path, "r:gz") as tar:
        folder_path = file_path.split(".tar.gz")[0]
        folder_path_exists = os.path.exists(folder_path)

        if not folder_path_exists:
            tar.extractall(path=folder_path)

    return folder_path


def load_cec_auxiliary(name, year):
    """Loads auxiliary data for CEC-based benchmarking functions.

    Args:
        name (str): Name of function to be loaded.
        year (str): Year of function to be loaded.

    Returns:
        An array holding the auxiliary data.

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
