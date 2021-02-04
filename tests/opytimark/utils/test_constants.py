import sys

from opytimark.utils import constants


def test_constants():
    assert constants.DATA_FOLDER == 'data/'
    
    assert constants.EPSILON == 1e-32

    assert constants.FLOAT_MAX == sys.float_info.max
