import io
import tarfile
from pathlib import Path

import numpy as np

from opytimark.utils import loader


def test_download_and_untar_file(tmp_path):
    source = tmp_path / "source.tar.gz"
    payload = b"1 2 3\n"
    with tarfile.open(str(source), "w:gz") as archive:
        info = tarfile.TarInfo("values.txt")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    downloaded = tmp_path / "downloaded.tar.gz"
    loader.download_file(source.as_uri(), str(downloaded))
    folder = Path(loader.untar_file(str(downloaded)))

    assert (folder / "values.txt").read_bytes() == payload


def test_load_cec_auxiliary_returns_independent_arrays():
    first = loader.load_cec_auxiliary("F1_o", "2005")
    second = loader.load_cec_auxiliary("F1_o", "2005")

    first[0] = 0

    assert second.shape == (100,)
    assert second[0] != 0
    assert np.isfinite(second).all()


def test_load_cec_auxiliary_prefers_local_data(tmp_path, monkeypatch):
    folder = tmp_path / "custom"
    folder.mkdir()
    np.savetxt(str(folder / "F1_o.txt"), np.array([1, 2, 3]))
    monkeypatch.setattr(loader, "DATA_FOLDER", str(tmp_path))

    assert np.array_equal(loader.load_cec_auxiliary("F1_o", "custom"), [1, 2, 3])
