import opytimark.utils.loader as l
from opytimark.markers.cec import year_2013


def test_F4():
    f = year_2013.F4()

    x = l.load_cec_auxiliary('F4_o', '2013')

    y = f(x)

    assert y == 0
