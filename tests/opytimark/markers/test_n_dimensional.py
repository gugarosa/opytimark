import numpy as np

from opytimark.markers import n_dimensional


def test_ackley1():
    f = n_dimensional.Ackley1()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_ackley4():
    f = n_dimensional.Ackley4()

    x = np.array([-1.51, -0.755])

    y = f(x)

    assert y == -4.5901006651507235


def test_alpine1():
    f = n_dimensional.Alpine1()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_alpine2():
    f = n_dimensional.Alpine2()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_brown():
    f = n_dimensional.Brown()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_chung_reynolds():
    f = n_dimensional.ChungReynolds()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_cosine_mixture():
    f = n_dimensional.CosineMixture()

    x = np.zeros(50)

    y = f(x)

    assert y == -5.0


def test_csendes():
    f = n_dimensional.Csendes()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_deb1():
    f = n_dimensional.Deb1()

    x = np.array([-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9])

    y = f(x)

    assert np.round(y, 1) == -1.0


def test_deb3():
    f = n_dimensional.Deb3()

    x = np.zeros(50)

    y = f(x)

    assert np.round(y, 3) == -0.125


def test_dixon_price():
    f = n_dimensional.DixonPrice()

    x = np.array([1, np.sqrt(0.5)])

    y = f(x)

    assert np.round(y) == 0


def test_exponential():
    f = n_dimensional.Exponential()

    x = np.zeros(50)

    y = f(x)

    assert y == 1


def test_f8f2():
    f = n_dimensional.F8F2()

    x = np.ones(50)

    y = f(x)

    assert y == 0


def test_griewank():
    f = n_dimensional.Griewank()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_happy_cat():
    f = n_dimensional.HappyCat()

    x = np.full(50, -1)

    y = f(x)

    assert y == 0


def test_high_conditioned_elliptic():
    f = n_dimensional.HighConditionedElliptic()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_levy():
    f = n_dimensional.Levy()

    x = np.ones(50)

    y = f(x)

    assert np.round(y) == 0


def test_michalewicz():
    f = n_dimensional.Michalewicz()

    x = np.array([2.20, 1.57])

    y = f(x)

    assert np.round(y, 4) == -1.8011


def test_non_continous_expanded_scaffer_f6():
    f = n_dimensional.NonContinuousExpandedScafferF6()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_non_continous_rastrigin():
    f = n_dimensional.NonContinuousRastrigin()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_pathological():
    f = n_dimensional.Pathological()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_periodic():
    f = n_dimensional.Periodic()

    x = np.zeros(50)

    y = f(x)

    assert y == 0.9


def test_perm_0_d_beta():
    f = n_dimensional.Perm0DBeta()

    x = np.array([1, 1 / 2, 1 / 3, 1 / 4, 1 / 5])

    y = f(x)

    assert np.round(y) == 0


def test_perm_d_beta():
    f = n_dimensional.PermDBeta()

    x = np.array([1, 2, 3, 4, 5])

    y = f(x)

    assert np.round(y) == 0


def test_powell_singular2():
    f = n_dimensional.PowellSingular2()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_powell_sum():
    f = n_dimensional.PowellSum()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_qing():
    f = n_dimensional.Qing()

    x = np.array([np.sqrt(1), np.sqrt(2), np.sqrt(3)])

    y = f(x)

    assert np.round(y) == 0


def test_quartic():
    f = n_dimensional.Quartic()

    x = np.zeros(50)

    y = f(x)

    assert y <= 1


def test_quintic():
    f = n_dimensional.Quintic()

    x = np.full(50, -1)

    y = f(x)

    assert y == 0

    x = np.full(50, 2)

    y = f(x)

    assert y == 0


def test_rana():
    f = n_dimensional.Rana()

    x = np.full(50, -500)

    y = f(x)

    assert y == -22285.14852971478


def test_rastrigin():
    f = n_dimensional.Rastrigin()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_ridge():
    f = n_dimensional.Ridge()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_rosenbrock():
    f = n_dimensional.Rosenbrock()

    x = np.ones(50)

    y = f(x)

    assert y == 0


def test_rotated_expanded_scaffer_f6():
    f = n_dimensional.RotatedExpandedScafferF6()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_rotated_hyper_ellipsoid():
    f = n_dimensional.RotatedHyperEllipsoid()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_salomon():
    f = n_dimensional.Salomon()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_schumer_steiglitz():
    f = n_dimensional.SchumerSteiglitz()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_schwefel():
    f = n_dimensional.Schwefel()

    x = np.full(50, 420.9687)

    y = f(x)

    assert np.round(y) == 0


def test_schwefel220():
    f = n_dimensional.Schwefel220()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_schwefel221():
    f = n_dimensional.Schwefel221()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_schwefel222():
    f = n_dimensional.Schwefel222()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_schwefel223():
    f = n_dimensional.Schwefel223()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_schwefel225():
    f = n_dimensional.Schwefel225()

    x = np.ones(50)

    y = f(x)

    assert y == 0


def test_schwefel226():
    f = n_dimensional.Schwefel226()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_shubert():
    f = n_dimensional.Shubert()

    x = np.zeros(50)

    y = f(x)

    assert np.round(y) == 0


def test_shubert3():
    f = n_dimensional.Shubert3()

    x = np.zeros(50)

    y = f(x)

    assert np.round(y) == -237


def test_shubert4():
    f = n_dimensional.Shubert4()

    x = np.zeros(50)

    y = f(x)

    assert np.round(y) == -223


def test_schaffer_f6():
    f = n_dimensional.SchafferF6()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_sphere():
    f = n_dimensional.Sphere()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_sphere_with_noise():
    f = n_dimensional.SphereWithNoise()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_step():
    f = n_dimensional.Step()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_step2():
    f = n_dimensional.Step2()

    x = np.full(50, -0.5)

    y = f(x)

    assert y == 0


def test_step3():
    f = n_dimensional.Step3()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_streched_v_sine_wave():
    f = n_dimensional.StrechedVSineWave()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_styblinski_tang():
    f = n_dimensional.StyblinskiTang()

    x = np.array([-2.903534, -2.903534])

    y = f(x)

    assert np.round(y, 4) == -78.3323


def test_sum_different_powers():
    f = n_dimensional.SumDifferentPowers()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_sum_squares():
    f = n_dimensional.SumSquares()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_trid():
    f = n_dimensional.Trid()

    x = np.array([2, 2])

    y = f(x)

    assert y == -2


def test_trigonometric1():
    f = n_dimensional.Trigonometric1()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_trigonometric2():
    f = n_dimensional.Trigonometric2()

    x = np.full(50, 0.9)

    y = f(x)

    assert y == 1


def test_wavy():
    f = n_dimensional.Wavy()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_weierstrass():
    f = n_dimensional.Weierstrass()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_xin_she_yang():
    f = n_dimensional.XinSheYang()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_xin_she_yan2():
    f = n_dimensional.XinSheYang2()

    x = np.zeros(50)

    y = f(x)

    assert y == 0


def test_xin_she_yang3():
    f = n_dimensional.XinSheYang3()

    x = np.zeros(50)

    y = f(x)

    assert y == -1


def test_xin_she_yang4():
    f = n_dimensional.XinSheYang4()

    x = np.zeros(50)

    y = f(x)

    assert y == -1


def test_zakharov():
    f = n_dimensional.Zakharov()

    x = np.zeros(50)

    y = f(x)

    assert y == 0
