import numpy as np

from opytimark.markers.cec import year_2010


def test_F1():
    f = year_2010.F1()

    x = np.array([-36.8842894, -49.4144157, 52.3625561, -62.4563044, 21.9347535])

    y = f(x)

    assert y == 0


def test_F2():
    f = year_2010.F2()

    x = np.array([-2.11826934, 3.63430157, -2.45105877, -3.40291962, 1.55897077])

    y = f(x)

    assert y == 0


def test_F3():
    f = year_2010.F3()

    x = np.array([22.9398580, -26.2158471, 3.04427070, -11.6837655, -2.84278344])

    y = f(x)

    assert y == 0
