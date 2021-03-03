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

def test_F10():
    f = year_2010.F10(group_size=3)

    x = np.array([4.01137594, -0.408007193, 1.64479052, -2.44287696, 1.58452018])

    y = f(x)

    assert y == 0

def test_F11():
    f = year_2010.F11(group_size=3)

    x = np.array([21.0218123, -1.42266583, 7.74318110, -28.6969265, 11.6738134])

    y = f(x)

    assert y == 0

def test_F12():
    f = year_2010.F12(group_size=3)

    x = np.array([-12.9233824, 46.7684469, -43.1863610, 37.2031847, 35.9039872])

    y = f(x)

    assert y == 0

def test_F13():
    f = year_2010.F13(group_size=3)

    x = np.array([-16.8399898, -98.5634041, -96.8428816, 92.9324128, -98.2147851])

    y = f(x)

    assert y == 0

def test_F14():
    f = year_2010.F14(group_size=3)

    x = np.array([60.3990593, 82.6543534, 89.8018632, -40.2622723, -35.1271793])

    y = f(x)

    assert y == 0

def test_F15():
    f = year_2010.F15(group_size=3)

    x = np.array([-0.605280722, -0.0431802822, 2.38813967, 0.995560478, -1.98047109])

    y = f(x)

    assert y == 0

def test_F16():
    f = year_2010.F16(group_size=3)

    x = np.array([-5.44104284, 3.58829303, 4.04146483, 3.73257391, -1.46393263])

    y = f(x)

    assert y == 0

def test_F17():
    f = year_2010.F17(group_size=3)

    x = np.array([-54.2952020, -35.8624243, 76.9180144, -72.7802232, -90.9895313])

    y = f(x)

    assert y == 0

def test_F18():
    f = year_2010.F18(group_size=3)

    x = np.array([-82.1766825, -43.6583360, 67.4227545, 3.04409842, -99.6584289])

    y = f(x)

    assert y == 0
