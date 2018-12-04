from mvpa2.testing import *
from mvpa2.misc.fx import single_gamma_hrf
from mvpa2.misc.fx import double_gamma_hrf


def test_negative_values_single_gamma_hrf():

    # generate something with negative and/or positive values to convolve
    x_neg = np.random.randint(-10, -5, (10))
    x_pos = np.random.randint(5, 10, (10))
    x_pn = np.random.randint(-5, 5, (10))
    # define lambda function for convolution:
    hrf_gen=lambda t: single_gamma_hrf(t)
    hrf_double_gen=lambda t: double_gamma_hrf(t)
    hrf_xntest = hrf_gen(x_neg)
    hrf_xptest = hrf_gen(x_pos)
    hrf_xpntest = hrf_gen(x_pn)
    hrf_double_xntest = hrf_double_gen(x_neg)
    hrf_double_xptest = hrf_double_gen(x_pos)
    hrf_double_xpntest = hrf_double_gen(x_pn)

    # this is a bit convoluted. It checks whether any element of the result is nan
    assert_true(~[np.isnan(i) for i in hrf_xntest][0].any())
    assert_true(~[np.isnan(i) for i in hrf_xptest][0].any())
    assert_true(~[np.isnan(i) for i in hrf_xpntest][0].any())
    assert_true(~[np.isnan(i) for i in hrf_double_xntest][0].any())
    assert_true(~[np.isnan(i) for i in hrf_double_xptest][0].any())
    assert_true(~[np.isnan(i) for i in hrf_double_xpntest][0].any())
