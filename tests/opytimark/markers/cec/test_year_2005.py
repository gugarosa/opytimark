import opytimark.utils.loader as l
from opytimark.markers.cec import year_2005


def test_F1():
    f = year_2005.F1()

    x = l.load_cec_auxiliary('F1_o', '2005')

    y = f(x)

    assert y == -450


def test_F2():
    f = year_2005.F2()

    x = l.load_cec_auxiliary('F2_o', '2005')

    y = f(x)

    assert y == -450


def test_F3():
    f = year_2005.F3()

    x = l.load_cec_auxiliary('F3_o', '2005')
    x = x[:50]

    y = f(x)

    assert y == -450


def test_F4():
    f = year_2005.F4()

    x = l.load_cec_auxiliary('F4_o', '2005')

    y = f(x)

    assert y == -450


def test_F5():
    f = year_2005.F5()

    x = l.load_cec_auxiliary('F5_o', '2005')
    x[:int(x.shape[0]/4)] = -100
    x[int(3*x.shape[0]/4):] = 100

    y = f(x)

    assert y == -310


def test_F6():
    f = year_2005.F6()

    x = l.load_cec_auxiliary('F6_o', '2005')
    x += 1

    y = f(x)

    assert y == 390


def test_F7():
    f = year_2005.F7()

    x = l.load_cec_auxiliary('F7_o', '2005')
    x = x[:50]

    y = f(x)

    assert y == -180


def test_F8():
    f = year_2005.F8()

    x = l.load_cec_auxiliary('F8_o', '2005')
    x = x[:50]

    for j in range(int(x.shape[0]/2)):
        x[2*j] = -32 * x[2*j+1]

    y = f(x)

    assert y == -140


def test_F9():
    f = year_2005.F9()

    x = l.load_cec_auxiliary('F9_o', '2005')

    y = f(x)

    assert y == -330


def test_F10():
    f = year_2005.F10()

    x = l.load_cec_auxiliary('F10_o', '2005')
    x = x[:50]

    y = f(x)

    assert y == -330


def test_F11():
    f = year_2005.F11()

    x = l.load_cec_auxiliary('F11_o', '2005')
    x = x[:50]

    y = f(x)

    assert y == 90


def test_F12():
    f = year_2005.F12()

    x = l.load_cec_auxiliary('F12_alpha', '2005')

    y = f(x)

    assert y == -460


def test_F13():
    f = year_2005.F13()

    x = l.load_cec_auxiliary('F13_o', '2005')

    y = f(x)

    assert y == -130


def test_F14():
    f = year_2005.F14()

    x = l.load_cec_auxiliary('F14_o', '2005')
    x = x[:50]

    y = f(x)

    assert y == -300


def test_F15():
    f = year_2005.F15()

    x = l.load_cec_auxiliary('F15_o', '2005')
    x = x[0][:50]

    y = f(x)

    assert y == 120


def test_F16():
    f = year_2005.F16()

    x = l.load_cec_auxiliary('F16_o', '2005')
    x = x[0][:50]

    y = f(x)

    assert y == 120


def test_F17():
    f = year_2005.F17()

    x = l.load_cec_auxiliary('F17_o', '2005')
    x = x[0][:50]

    y = f(x)

    assert y == 120


def test_F18():
    f = year_2005.F18()

    x = l.load_cec_auxiliary('F18_o', '2005')
    x = x[0][:50]

    y = f(x)

    assert y == 10


def test_F19():
    f = year_2005.F19()

    x = l.load_cec_auxiliary('F19_o', '2005')
    x = x[0][:50]

    y = f(x)

    assert y == 10


def test_F20():
    f = year_2005.F20()

    x = l.load_cec_auxiliary('F20_o', '2005')
    x = x[0][:50]

    for j in range(int(x.shape[0]/2)):
        x[2*j+1] = 5

    y = f(x)

    assert y == 10


def test_F21():
    f = year_2005.F21()

    x = l.load_cec_auxiliary('F21_o', '2005')
    x = x[0][:50]

    y = f(x)

    assert y == 360


def test_F22():
    f = year_2005.F22()

    x = l.load_cec_auxiliary('F22_o', '2005')
    x = x[0][:50]

    y = f(x)

    assert y == 360


def test_F23():
    f = year_2005.F23()

    x = l.load_cec_auxiliary('F23_o', '2005')
    x = x[0][:50]

    y = f(x)

    assert y == 360


def test_F24():
    f = year_2005.F24()

    x = l.load_cec_auxiliary('F24_o', '2005')
    x = x[0][:50]

    y = f(x)

    assert y == 260


def test_F25():
    f = year_2005.F25()

    x = l.load_cec_auxiliary('F25_o', '2005')
    x = x[0][:50]

    y = f(x)

    assert y == 260
