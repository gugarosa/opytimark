import numpy as np

from opytimark.markers import two_dimensional


def test_ackley2():
    f = two_dimensional.Ackley2()

    x = np.array([0.5, 0.5])

    y = f(x)

    assert np.round(y, 2) == -173.62


def test_ackley3():
    f = two_dimensional.Ackley3()

    x = np.array([0.5, 0.5])

    y = f(x)

    assert np.round(y, 2) == -159.07


def test_adjiman():
    f = two_dimensional.Adjiman()

    x = np.array([0.5, 0.5])

    y = f(x)

    assert np.round(y, 2) == 0.02


def test_bartels_conn():
    f = two_dimensional.BartelsConn()

    x = np.array([0, 0])

    y = f(x)

    assert y == 1


def test_beale():
    f = two_dimensional.Beale()

    x = np.array([3, 0.5])

    y = f(x)

    assert y == 0


def test_biggs_exponential2():
    f = two_dimensional.BiggsExponential2()

    x = np.array([1, 10])

    y = f(x)

    assert y == 0


def test_bird():
    f = two_dimensional.Bird()

    x = np.array([4.70104, 3.15294])

    y = f(x)

    assert y == -106.76453674760198


def test_bohachevsky1():
    f = two_dimensional.Bohachevsky1()

    x = np.array([0, 0])

    y = f(x)

    assert y == 0


def test_bohachevsky2():
    f = two_dimensional.Bohachevsky2()

    x = np.array([0, 0])

    y = f(x)

    assert y == 0


def test_bohachevsky3():
    f = two_dimensional.Bohachevsky3()

    x = np.array([0, 0])

    y = f(x)

    assert y == 0


def test_booth():
    f = two_dimensional.Booth()

    x = np.array([1, 3])

    y = f(x)

    assert y == 0


def test_branin_hoo():
    f = two_dimensional.BraninHoo()

    x = np.array([9.42478, 2.475])

    y = f(x)

    assert y == 0.39788735775266204


def test_brent():
    f = two_dimensional.Brent()

    x = np.array([-10, -10])

    y = f(x)

    assert y == np.exp(-200)


def test_bukin2():
    f = two_dimensional.Bukin2()

    x = np.array([-10, 0])

    y = f(x)

    assert y == 0


def test_bukin4():
    f = two_dimensional.Bukin4()

    x = np.array([-10, 0])

    y = f(x)

    assert y == 0


def test_bukin6():
    f = two_dimensional.Bukin6()

    x = np.array([-10, 1])

    y = f(x)

    assert y == 0


def test_camel3():
    f = two_dimensional.Camel3()

    x = np.array([0, 0])

    y = f(x)

    assert y == 0


def test_camel6():
    f = two_dimensional.Camel6()

    x = np.array([0.0898, -0.7126])

    y = f(x)

    assert y == -1.0316284229280819


def test_chen_bird():
    f = two_dimensional.ChenBird()

    x = np.array([0.5, 0.5])

    y = f(x)

    assert y == -2000.0039999840003


def test_chen_v():
    f = two_dimensional.ChenV()

    x = np.array([0.388888888888889, 0.722222222222222])

    y = f(x)

    assert y == 2000.0000000000002


def test_chichinadze():
    f = two_dimensional.Chichinadze()

    x = np.array([6.189866586965680, 0.5])

    y = f(x)

    assert y == -42.94438701899099


def test_cross_tray():
    f = two_dimensional.CrossTray()

    x = np.array([1.349406685353340, 1.349406608602084])

    y = f(x)

    assert y == -2.062611870822739


def test_cube():
    f = two_dimensional.Cube()

    x = np.array([1, 1])

    y = f(x)

    assert y == 0


def test_damavandi():
    f = two_dimensional.Damavandi()

    x = np.array([2.00000001, 2.00000001])

    y = f(x)

    assert np.round(y) == 0


def test_deckkers_aarts():
    f = two_dimensional.DeckkersAarts()

    x = np.array([0, 15])

    y = f(x)

    assert y == -24771.093749999996


def test_drop_wave():
    f = two_dimensional.DropWave()

    x = np.array([0, 0])

    y = f(x)

    assert y == -1


def test_easom():
    f = two_dimensional.Easom()

    x = np.array([np.pi, np.pi])

    y = f(x)

    assert y == -1


def test_el_attar_vidyasagar_dutta():
    f = two_dimensional.ElAttarVidyasagarDutta()

    x = np.array([3.4091868222, -2.1714330361])

    y = f(x)

    assert y == 1.7127803548622027


def test_egg_crate():
    f = two_dimensional.EggCrate()

    x = np.array([0, 0])

    y = f(x)

    assert y == 0


def test_egg_holder():
    f = two_dimensional.EggHolder()

    x = np.array([512, 404.2319])

    y = f(x)

    assert y == -959.6406627106155


def test_freudenstein_roth():
    f = two_dimensional.FreudensteinRoth()

    x = np.array([5, 4])

    y = f(x)

    assert y == 0


def test_giunta():
    f = two_dimensional.Giunta()

    x = np.array([0.4673200277395354, 0.4673200169591304])

    y = f(x)

    assert y == 0.06447042053690571


def test_goldenstein_price():
    f = two_dimensional.GoldsteinPrice()

    x = np.array([0, -1])

    y = f(x)

    assert y == 3


def test_himmelblau():
    f = two_dimensional.Himmelblau()

    x = np.array([3, 2])

    y = f(x)

    assert y == 0


def test_holder_table():
    f = two_dimensional.HolderTable()

    x = np.array([8.05502, 9.66459])

    y = f(x)

    assert y == -19.208502567767606


def test_hosaki():
    f = two_dimensional.Hosaki()

    x = np.array([4, 2])

    y = f(x)

    assert y == -2.345811576101292


def test_jennrich_sampson():
    f = two_dimensional.JennrichSampson()

    x = np.array([0.257825, 0.257825])

    y = f(x)

    assert y == 124.36218236181409


def test_keane():
    f = two_dimensional.Keane()

    x = np.array([1.393249070031784, 0])

    y = f(x)

    assert y == 0.6736675211468548


def test_leon():
    f = two_dimensional.Leon()

    x = np.array([1, 1])

    y = f(x)

    assert y == 0


def test_levy13():
    f = two_dimensional.Levy13()

    x = np.array([1, 1])

    y = f(x)

    assert np.round(y) == 0


def test_matyas():
    f = two_dimensional.Matyas()

    x = np.array([0, 0])

    y = f(x)

    assert y == 0


def test_mc_cormick():
    f = two_dimensional.McCormick()

    x = np.array([-0.547, -1.547])

    y = f(x)

    assert y == -1.9132228873800594


def test_mishra3():
    f = two_dimensional.Mishra3()

    x = np.array([-8.466613775046579, -9.998521308999999])

    y = f(x)

    assert y == -0.18465133334298883


def test_mishra4():
    f = two_dimensional.Mishra4()

    x = np.array([-9.941127263635860, -9.999571661999983])

    y = f(x)

    assert y == -0.1994069700888328


def test_mishra5():
    f = two_dimensional.Mishra5()

    x = np.array([-1.986820662153768, -10])

    y = f(x)

    assert y == -1.019829519930943


def test_mishra6():
    f = two_dimensional.Mishra6()

    x = np.array([2.886307215440481, 1.823260331422321])

    y = f(x)

    assert y == -2.2839498384747587


def test_mishra8():
    f = two_dimensional.Mishra8()

    x = np.array([2, -3])

    y = f(x)

    assert y == 0


def test_parsopoulos():
    f = two_dimensional.Parsopoulos()

    x = np.array([np.pi / 2, np.pi])

    y = f(x)

    assert np.round(y) == 0


def test_pen_holder():
    f = two_dimensional.PenHolder()

    x = np.array([-9.646167671043401, -9.646167671043401])

    y = f(x)

    assert y == -0.9635348327265058


def test_periodic():
    f = two_dimensional.Periodic()

    x = np.array([0, 0])

    y = f(x)

    assert y == 0.9


def test_price1():
    f = two_dimensional.Price1()

    x = np.array([-5, -5])

    y = f(x)

    assert y == 0


def test_price2():
    f = two_dimensional.Price2()

    x = np.array([0, 0])

    y = f(x)

    assert y == 0.9


def test_price3():
    f = two_dimensional.Price3()

    x = np.array([0.341307503353524, 0.116490811845416])

    y = f(x)

    assert np.round(y) == 0


def test_price4():
    f = two_dimensional.Price4()

    x = np.array([2, 4])

    y = f(x)

    assert y == 0


def test_quadratic():
    f = two_dimensional.Quadratic()

    x = np.array([0.19388, 0.48513])

    y = f(x)

    assert y == -3873.7241821830326


def test_rotated_ellipse1():
    f = two_dimensional.RotatedEllipse1()

    x = np.array([0, 0])

    y = f(x)

    assert y == 0


def test_rotated_ellipse2():
    f = two_dimensional.RotatedEllipse2()

    x = np.array([0, 0])

    y = f(x)

    assert y == 0


def test_rump():
    f = two_dimensional.Rump()

    x = np.array([0, 0])

    y = f(x)

    assert y == 0


def test_schaffer1():
    f = two_dimensional.Schaffer1()

    x = np.array([0, 0])

    y = f(x)

    assert y == 0


def test_schaffer2():
    f = two_dimensional.Schaffer2()

    x = np.array([0, 0])

    y = f(x)

    assert y == 0


def test_schaffer3():
    f = two_dimensional.Schaffer3()

    x = np.array([0, 1.253115])

    y = f(x)

    assert y == 0.0015668545260126288


def test_schaffer4():
    f = two_dimensional.Schaffer4()

    x = np.array([0, 1.253115])

    y = f(x)

    assert y == 0.29243850703298857


def test_schwefel26():
    f = two_dimensional.Schwefel26()

    x = np.array([1, 3])

    y = f(x)

    assert y == 0


def test_schwefel236():
    f = two_dimensional.Schwefel236()

    x = np.array([12, 12])

    y = f(x)

    assert y == -3456


def test_table1():
    f = two_dimensional.Table1()

    x = np.array([9.646168, 9.646168])

    y = f(x)

    assert y == -26.920335555515848


def test_table2():
    f = two_dimensional.Table2()

    x = np.array([8.055023472141116, 9.664590028909654])

    y = f(x)

    assert y == -19.20850256788675


def test_table3():
    f = two_dimensional.Table3()

    x = np.array([9.646157266348881, 9.646134286497169])

    y = f(x)

    assert y == -24.156815516506533


def test_testtube_holder():
    f = two_dimensional.TesttubeHolder()

    x = np.array([np.pi / 2, 0])

    y = f(x)

    assert y == -10.872299901558


def test_trecani():
    f = two_dimensional.Trecani()

    x = np.array([0, 0])

    y = f(x)

    assert y == 0


def test_trefethen():
    f = two_dimensional.Trefethen()

    x = np.array([-0.024403, 0.210612])

    y = f(x)

    assert y == -3.3068686465567008


def test_venter_sobiezcczanski_sobieski():
    f = two_dimensional.VenterSobiezcczanskiSobieski()

    x = np.array([0, 0])

    y = f(x)

    assert y == -400


def test_wayburn_seader1():
    f = two_dimensional.WayburnSeader1()

    x = np.array([1, 2])

    y = f(x)

    assert y == 0


def test_wayburn_seader2():
    f = two_dimensional.WayburnSeader2()

    x = np.array([0.200138974728779, 1])

    y = f(x)

    assert np.round(y) == 0


def test_wayburn_seader3():
    f = two_dimensional.WayburnSeader3()

    x = np.array([5.146896745324582, 6.839589743000071])

    y = f(x)

    assert y == 19.105879794568


def test_zettl():
    f = two_dimensional.Zettl()

    x = np.array([-0.0299, 0])

    y = f(x)

    assert y == -0.0037912371501199


def test_zirilli():
    f = two_dimensional.Zirilli()

    x = np.array([-1.0465, 0])

    y = f(x)

    assert y == -0.3523860365437344
