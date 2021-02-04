"""Auxiliary files loader.
"""

import os
import urllib.request
import tarfile


def download_file(url, output_path):
    """Downloads a file given its URL and the output path to be saved.

    Args:
        url (str): URL to download the file.
        output_path (str): Path to save the downloaded file.

    """

    # Checks if file exists
    file_exists = os.path.exists(output_path)

    # If file does not exist
    if not file_exists:
        # Downloads the file
        urllib.request.urlretrieve(url, output_path)


def untar_file(file_name):
    """De-compress a file with .tar.gz.

    Args:
        file_name (str): Name of the file to be de-compressed.

    """

    # Opens a .tar.gz file with `file_name`
    with tarfile.open(f'{file_name}.tar.gz', "r:gz") as tar:
        # Extracts all files
        tar.extractall(path=file_name)


def load_cec_auxiliary(idx, year):
    """
    """

    #
    base_url = 'http://recogna.tech/files/opytimark/'
    file_name = year + '.tar.gz'
    output_path = 'data/' + file_name

    #
    download_file(base_url + file_name, output_path)

    #
    untar_file(output_path)

    #


