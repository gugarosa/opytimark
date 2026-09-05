import io
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

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


def test_failed_download_is_retryable(tmp_path, monkeypatch):
    output = tmp_path / "download.tar.gz"

    def incomplete_download(url, destination):
        Path(destination).write_bytes(b"partial")
        raise OSError("connection interrupted")

    monkeypatch.setattr(loader.urllib.request, "urlretrieve", incomplete_download)
    with pytest.raises(OSError, match="connection interrupted"):
        loader.download_file("https://example.invalid/data", output)

    assert not output.exists()
    assert list(tmp_path.iterdir()) == []

    source = tmp_path / "source"
    source.write_bytes(b"complete")
    monkeypatch.undo()
    loader.download_file(source.as_uri(), output)
    assert output.read_bytes() == b"complete"


@pytest.mark.parametrize("failure", ["unreadable", "during-extraction"])
def test_failed_extraction_is_retryable(tmp_path, monkeypatch, failure):
    source = tmp_path / "source.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        info = tarfile.TarInfo("values.txt")
        info.size = 2
        archive.addfile(info, io.BytesIO(b"1\n"))
    valid_archive = source.read_bytes()

    if failure == "unreadable":
        source.write_bytes(b"incomplete archive")
    else:

        def incomplete_extraction(archive, path, *args, **kwargs):
            Path(path, "partial.txt").write_text("partial")
            raise OSError("extraction interrupted")

        monkeypatch.setattr(tarfile.TarFile, "extractall", incomplete_extraction)

    with pytest.raises((tarfile.ReadError, OSError)):
        loader.untar_file(source)

    assert not (tmp_path / "source").exists()
    assert list(tmp_path.iterdir()) == [source]

    source.write_bytes(valid_archive)
    monkeypatch.undo()
    folder = Path(loader.untar_file(source))
    assert (folder / "values.txt").read_bytes() == b"1\n"


def test_existing_download_and_extraction_are_preserved(tmp_path):
    output = tmp_path / "cached.tar.gz"
    output.write_bytes(b"existing download")
    folder = tmp_path / "cached"
    folder.mkdir()
    (folder / "values.txt").write_text("local override")

    loader.download_file("https://example.invalid/data", output)

    assert output.read_bytes() == b"existing download"
    assert Path(loader.untar_file(output)) == folder
    assert (folder / "values.txt").read_text() == "local override"


def test_bundled_io_errors_do_not_fall_back_to_download(tmp_path, monkeypatch):
    resource = Mock()
    resource.joinpath.return_value.read_bytes.side_effect = PermissionError(
        "bundled data is not readable"
    )

    def unexpected_download(*args):
        pytest.fail("an unreadable bundled resource must not trigger a download")

    monkeypatch.setattr(loader, "DATA_FOLDER", str(tmp_path))
    monkeypatch.setattr(loader, "files", lambda package: resource)
    monkeypatch.setattr(loader, "download_file", unexpected_download)

    with pytest.raises(PermissionError, match="bundled data is not readable"):
        loader.load_cec_auxiliary("unreadable", "test-permissions")


def test_missing_bundled_archive_uses_fallback(tmp_path, monkeypatch):
    archive_data = io.BytesIO()
    with tarfile.open(fileobj=archive_data, mode="w:gz") as archive:
        info = tarfile.TarInfo("values.txt")
        info.size = 6
        archive.addfile(info, io.BytesIO(b"1 2 3\n"))

    def download(url, destination):
        assert url == f"{loader._BASE_URL}missing.tar.gz"
        Path(destination).write_bytes(archive_data.getvalue())

    monkeypatch.setattr(loader, "DATA_FOLDER", str(tmp_path))
    monkeypatch.setattr(loader, "files", lambda package: tmp_path)
    monkeypatch.setattr(loader.urllib.request, "urlretrieve", download)

    np.testing.assert_array_equal(
        loader.load_cec_auxiliary("values", "missing"), [1, 2, 3]
    )


def test_zip_import_handles_present_and_missing_resources(tmp_path):
    package = Path(loader.__file__).parents[1]
    archive_path = tmp_path / "opytimark.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for parts in [
            ("__init__.py",),
            ("utils", "__init__.py"),
            ("utils", "constants.py"),
            ("utils", "loader.py"),
            ("data", "__init__.py"),
            ("data", "2005.tar.gz"),
        ]:
            archive.write(package.joinpath(*parts), "/".join(("opytimark", *parts)))

    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import sys
sys.path.insert(0, sys.argv[1])
from opytimark.utils import loader

assert '.zip' in loader.__file__
assert loader.load_cec_auxiliary('F1_o', '2005').shape == (100,)
assert loader._load_bundled('values', 'missing') is None
""",
            str(archive_path),
        ],
        cwd=tmp_path,
        check=True,
    )
