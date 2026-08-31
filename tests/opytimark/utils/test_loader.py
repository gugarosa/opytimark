from opytimark.utils.loader import load_cec_auxiliary


def test_load_cec_auxiliary_returns_independent_arrays():
    first = load_cec_auxiliary("F1_o", "2005")
    second = load_cec_auxiliary("F1_o", "2005")

    first[0] = 0

    assert second.shape == (100,)
    assert second[0] != 0
