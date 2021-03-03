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


def test_F4():
    f = year_2010.F4(group_size=3)

    x = np.array([75.2782785, -75.5448350, -3.32647349, 24.4725725, -97.8481755])

    y = f(x)

    assert y == 0


def test_F5():
    f = year_2010.F5(group_size=3)

    x = np.array([-4.11740908, 4.88998416, 3.66810078, 1.38579175, -3.05399570])

    y = f(x)

    assert y == 0

def test_F6():
    f = year_2010.F6(group_size=3)

    x = np.array([19.7711343, 24.0654493, -3.35055413, 0.896206766, 25.9506031])

    y = f(x)

    assert y == 0

def test_F7():
    f = year_2010.F7(group_size=3)

    x = np.array([38.4601100, -10.8592936, -93.9407186, 97.5466996, 5.69178553])

    y = f(x)

    assert y == 0

def test_F8():
    f = year_2010.F8(group_size=3)

    x = np.array([56.8153286, 77.6415951, -62.6235968, -13.9831963, 91.5341041])

    y = f(x)

    assert y == 0

def test_F9():
    f = year_2010.F9(group_size=3)

    x = np.array([89.2431653, 1.83310145, 88.8133159, 53.3132770, -82.4244899])

    y = f(x)

    assert y == 0
