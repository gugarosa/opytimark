import os

from opytimark.utils import loader


def test_download_file():
    url = "http://recogna.tech/files/opytimark/"
    tar_name = "2005.tar.gz"
    tar_path = f"data/{tar_name}"

    loader.download_file(url + tar_name, tar_path)

    assert os.path.exists(tar_path)


def test_untar_file():
    tar_name = "2005.tar.gz"
    tar_path = f"data/{tar_name}"

    folder_path = loader.untar_file(tar_path)

    assert os.path.exists(folder_path)


def test_load_cec_auxiliary():
    data = loader.load_cec_auxiliary("F1_o", "2005")

    assert data.shape[0] == 100
