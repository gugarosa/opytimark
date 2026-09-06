# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import os
import tarfile
import urllib.request
from functools import lru_cache
from importlib.resources import files
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from numpy.typing import NDArray

from opytimark.utils.constants import DATA_FOLDER

_BASE_URL = "http://recogna.tech/files/opytimark/"


def download_file(url: str, output_path: str | os.PathLike[str]) -> None:
    """Download a file unless it already exists, publishing only complete data.

    Args:
        url: Source URL accepted by urllib.request.urlretrieve.
        output_path: Destination filename whose parent directories are created as needed.

    Raises:
        OSError: The download or filesystem operation fails.

    Notes:
        Existing destinations are left untouched.
        New downloads are staged in a sibling directory and published only after retrieval succeeds.
        Failed downloads propagate their error and leave no partial destination.

    """

    output = Path(output_path)
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output.parent) as temporary:
            downloaded = Path(temporary) / output.name
            urllib.request.urlretrieve(url, str(downloaded))
            downloaded.replace(output)


def untar_file(file_path: str | os.PathLike[str]) -> str:
    """Extract a trusted .tar.gz archive once and return its completed folder.

    Args:
        file_path: Trusted archive whose suffix is removed to obtain the destination folder.

    Returns:
        Extracted folder path, reusing an existing folder without modifying its contents.

    Raises:
        ValueError: An archive member's path would escape the extraction directory.
        tarfile.TarError: The archive cannot be read or extracted.
        OSError: A filesystem operation fails.

    Notes:
        Extraction is staged in a sibling directory and published only after completion.
        Failed extractions propagate their error and leave no partial destination.
        The path check is not a substitute for trusting the archive and its link targets.

    """

    archive = Path(file_path)
    folder = Path(str(archive)[: -len(".tar.gz")])

    if not folder.exists():
        with TemporaryDirectory(dir=folder.parent) as temporary:
            extracted = Path(temporary) / folder.name
            extracted.mkdir()

            with tarfile.open(str(archive), "r:gz") as tar:
                root = os.path.abspath(str(extracted))
                for member in tar.getmembers():
                    target = os.path.abspath(os.path.join(root, member.name))
                    if os.path.commonpath((root, target)) != root:
                        raise ValueError(f"`archive member={member.name}` would extract outside {root!r}.")

                tar.extractall(str(extracted))

            extracted.rename(folder)

    return str(folder)


@lru_cache(maxsize=None)
def _load_bundled(name: str, year: str) -> NDArray[np.float64] | None:
    try:
        archive = files("opytimark.data").joinpath(f"{year}.tar.gz").read_bytes()
    except FileNotFoundError:
        return None

    with tarfile.open(fileobj=BytesIO(archive), mode="r:gz") as tar:
        member = tar.extractfile(f"{name}.txt")
        return None if member is None else np.loadtxt(member)


def load_cec_auxiliary(name: str, year: str) -> NDArray[np.float64]:
    """Load CEC data from local overrides, bundled archives, or the remote source.

    Args:
        name: Auxiliary-data basename without the .txt suffix.
        year: CEC edition identifying the data directory and archive.

    Returns:
        NumPy array loaded with loadtxt's default dtype and shape, independently owned by the caller.

    Raises:
        OSError: Auxiliary data cannot be read or downloaded.
        tarfile.TarError: An auxiliary-data archive cannot be read or extracted.
        KeyError: A bundled archive does not contain the requested member.
        ValueError: Numeric data is invalid or an archive member would escape the extraction directory.

    Notes:
        Local extracted data takes precedence over local archives, then bundled data, then a remote download.
        Bundled resources work from filesystem and ZIP installations, with cached arrays copied for each call.
        Only a missing bundled archive or absent extractable data permits remote fallback.
        Read failures propagate rather than hiding corruption or permissions problems.

    """

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
