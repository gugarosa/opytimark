"""Bundled CEC auxiliary-data loader."""

import tarfile
from functools import cache
from importlib.resources import files

import numpy as np


@cache
def _load_cec_auxiliary(name, year):
    archive = files("opytimark.data").joinpath(f"{year}.tar.gz")
    with archive.open("rb") as stream, tarfile.open(fileobj=stream, mode="r:gz") as tar:
        member = tar.extractfile(f"{name}.txt")
        if member is None:
            raise FileNotFoundError(f"{name}.txt is missing from {archive.name}")
        return np.loadtxt(member)


def load_cec_auxiliary(name, year):
    """Load a copy of an auxiliary array bundled with the package."""

    return _load_cec_auxiliary(name, year).copy()
