import numpy as np
from pysixtrack.mathlibs import MathlibDefault
from pysixtrack.be_beamfields.qgauss import QGauss


def test_qgauss_gauss_compare():
    def gauss_distr(x, sigma, mu=0.0):
        sigma_squ = sigma * sigma
        arg = (x - mu) / sigma
        return np.exp(-arg * arg / 2.0) / np.sqrt(2 * np.pi * sigma_squ)

    q = 1.0
    sigma1 = 1.0
    abscissa = np.linspace(-4 * sigma1, 4 * sigma1, 101)
    cmp_gauss1 = gauss_distr(abscissa, sigma1)

    EPS = float(1e-16)
    distr = QGauss(q)
    assert distr.q == 1.0
    assert np.allclose(distr.cq, np.sqrt(np.pi), EPS, EPS)

    gauss1 = distr.eval(abscissa, QGauss.sqrt_beta(sigma1))
    assert np.allclose(cmp_gauss1, gauss1, EPS, EPS)

    sigma2 = 2.37
    abscissa = np.linspace(-4 * sigma2, 4 * sigma2, 101)
    cmp_gauss2 = gauss_distr(abscissa, sigma2)
    gauss2 = distr.eval(abscissa, QGauss.sqrt_beta(sigma2))
    assert np.allclose(cmp_gauss2, gauss2, EPS, EPS)


if __name__ == "__main__":
    test_qgauss_gauss_compare()
