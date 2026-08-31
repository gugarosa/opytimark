"""CEC auxiliary-data loading."""

import os
import pkgutil
import tarfile
import urllib.request
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np

from opytimark.utils.constants import DATA_FOLDER

_BASE_URL = "http://recogna.tech/files/opytimark/"


def download_file(url, output_path):
    """Download a file unless it already exists."""

    output = Path(output_path)
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(output))


def untar_file(file_path):
    """Extract a ``.tar.gz`` archive once and return its folder."""

    archive = Path(file_path)
    folder = Path(str(archive)[: -len(".tar.gz")])
    if not folder.exists():
        folder.mkdir(parents=True)
        with tarfile.open(str(archive), "r:gz") as tar:
            root = os.path.abspath(str(folder))
            for member in tar.getmembers():
                target = os.path.abspath(os.path.join(root, member.name))
                if os.path.commonpath((root, target)) != root:
                    raise ValueError(f"Unsafe archive member: {member.name}")
            tar.extractall(str(folder))
    return str(folder)


@lru_cache(maxsize=None)
def _load_bundled(name, year):
    try:
        archive = pkgutil.get_data("opytimark.data", f"{year}.tar.gz")
    except OSError:
        return None
    if archive is None:
        return None

    with tarfile.open(fileobj=BytesIO(archive), mode="r:gz") as tar:
        member = tar.extractfile(f"{name}.txt")
        return None if member is None else np.loadtxt(member)


def load_cec_auxiliary(name, year):
    """Load CEC data from local overrides, bundled archives, or the remote source."""

    archive = Path(DATA_FOLDER) / f"{year}.tar.gz"
    extracted = Path(DATA_FOLDER) / year / f"{name}.txt"

    if extracted.exists():
        return np.loadtxt(str(extracted))
    if archive.exists():
        return np.loadtxt(str(Path(untar_file(str(archive))) / f"{name}.txt"))

    bundled = _load_bundled(name, year)
    if bundled is not None:
        return bundled.copy()

    download_file(f"{_BASE_URL}{year}.tar.gz", str(archive))
    return np.loadtxt(str(Path(untar_file(str(archive))) / f"{name}.txt"))
