import numpy as np
import pysixtrack
from pysixtrack.mathlibs import MathlibDefault
from pysixtrack.be_beamfields.gaussian_fields import (
    _get_transv_field_gauss_ellip,
)


def test_track_spacecharge():
    x_co = 0.1
    y_co = -0.5
    sigma_x = 0.5
    sigma_y = 0.1
    el1 = pysixtrack.elements.SCQGaussProfile(
        number_of_particles=1e11,
        bunchlength_rms=0.22,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        length=2.0,
        x_co=x_co,
        y_co=y_co,
    )
    el2 = pysixtrack.elements.SCCoasting(
        number_of_particles=el1.number_of_particles,
        circumference=el1.bunchlength_rms * np.sqrt(2 * np.pi),
        sigma_x=el1.sigma_x,
        sigma_y=el1.sigma_y,
        length=el1.length,
        x_co=el1.x_co,
        y_co=el1.y_co,
    )

    # test absolute kick for sigma_x > sigma_y
    x_offset = 0.2
    y_offset = -0.5
    p1 = pysixtrack.Particles()
    p1.x = x_co + x_offset
    p1.y = y_co + y_offset
    p2 = p1.copy()
    el1.track(p1)
    el2.track(p2)
    assert np.isclose(p1.px, 1.8329795395186613e-07, atol=1e-15)
    assert np.isclose(p1.py, -8.540420459001383e-07, atol=1e-15)
    assert p1.compare(p2, abs_tol=1e-15)

    # test absolute kick for sigma_y > sigma_x
    el1.sigma_x = sigma_y
    el1.sigma_y = sigma_x
    el2.sigma_x = sigma_y
    el2.sigma_y = sigma_x
    p1 = pysixtrack.Particles()
    p1.x = x_co + y_offset
    p1.y = y_co + x_offset
    p2 = p1.copy()
    el1.track(p1)
    el2.track(p2)
    assert np.isclose(p1.px, -8.540420459001383e-07, atol=1e-15)
    assert np.isclose(p1.py, 1.8329795395186613e-07, atol=1e-15)
    assert p1.compare(p2, abs_tol=1e-15)

    # test no kick for particle on closed orbit
    p1 = pysixtrack.Particles()
    p1.x = el1.x_co
    p1.y = el1.y_co
    p2 = p1.copy()
    el1.track(p1)
    el2.track(p2)
    assert np.isclose(p1.px, 0.0, atol=1e-15)
    assert np.isclose(p1.py, 0.0, atol=1e-15)
    assert p1.compare(p2, abs_tol=1e-15)

    # test round beam
    p1 = pysixtrack.Particles()
    p1.x = el1.x_co + 0.5
    p1.y = el1.y_co + 0.1
    p2 = p1.copy()
    el1.sigma_y = el1.sigma_x
    el2.sigma_y = el2.sigma_x
    el1.track(p1)
    el2.track(p2)
    assert np.isclose(p1.px, 1.2895332740238447e-06, atol=1e-15)
    assert np.isclose(p1.py, 2.579066548047689e-07, atol=1e-15)
    assert p1.compare(p2, abs_tol=1e-15)


def test_get_transv_field_gauss_ellip():
    try:
        _get_transv_field_gauss_ellip(
            sigmax=1.0,
            sigmay=1.0,
            Delta_x=0.0,
            Delta_y=1.0,
            x=0.5,
            y=0.1,
            mathlib=MathlibDefault,
        )
    except ZeroDivisionError:
        pass  # test passed


if __name__ == "__main__":
    test_track_spacecharge()
