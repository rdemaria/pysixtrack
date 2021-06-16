import numpy as np

import pysixtrack

element_list = [
    pysixtrack.elements.Drift,
    pysixtrack.elements.DriftExact,
    pysixtrack.elements.Multipole,
    pysixtrack.elements.Cavity,
    pysixtrack.elements.SawtoothCavity,
    pysixtrack.elements.XYShift,
    pysixtrack.elements.SRotation,
    pysixtrack.elements.RFMultipole,
    pysixtrack.elements.BeamMonitor,
    pysixtrack.elements.DipoleEdge,
    pysixtrack.elements.Line,
    pysixtrack.elements.LimitRect,
    pysixtrack.elements.LimitEllipse,
    pysixtrack.elements.LimitRectEllipse,
    pysixtrack.elements.BeamBeam4D,
    pysixtrack.elements.BeamBeam6D,
    pysixtrack.elements.SCCoasting,
    pysixtrack.elements.SCQGaussProfile,
]


def test_track_all():
    for el in element_list:
        p = pysixtrack.Particles()
        el().track(p)


def test_track_rfmultipole():
    p1 = pysixtrack.Particles()
    p1.x = 1
    p1.y = 1
    p2 = p1.copy()

    el1 = pysixtrack.elements.RFMultipole(knl=[0.5, 2, 0.2], ksl=[0.5, 3, 0.1])
    el2 = pysixtrack.elements.Multipole(knl=el1.knl, ksl=el1.ksl)

    el1.track(p1)
    el2.track(p2)

    assert p1.compare(p2, abs_tol=1e-15)


def test_track_LimitRect():
    min_x = -0.1
    max_x = 0.3
    min_y = -0.5
    max_y = 0.1
    el = pysixtrack.elements.LimitRect(
        min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y
    )

    p1 = pysixtrack.Particles()
    p1.x = 1
    p1.y = 1
    ret = el.track(p1)
    assert ret == "Particle lost"

    arr = np.arange(0, 1, 0.001)
    p2 = pysixtrack.Particles(x=arr, y=arr)
    survive = np.where(
        (p2.x >= min_x) & (p2.x <= max_x) & (p2.y >= min_y) & (p2.y <= max_y)
    )
    ret = el.track(p2)
    assert len(p2.state) == len(survive[0])

    p2.x += max_x + 1e-6
    ret = el.track(p2)
    assert ret == "All particles lost"


def test_track_LimitEllipse():
    limit_a = 0.1
    limit_b = 0.2
    el = pysixtrack.elements.LimitEllipse(a=limit_a, b=limit_b)

    p1 = pysixtrack.Particles()
    p1.x = 1
    p1.y = 1
    ret = el.track(p1)
    assert ret == "Particle lost"

    arr = np.arange(0, 1, 0.001)
    p2 = pysixtrack.Particles(x=arr, y=arr)
    survive = np.where(
        (p2.x ** 2 / limit_a ** 2 + p2.y ** 2 / limit_b ** 2 <= 1.0)
    )
    ret = el.track(p2)
    assert len(p2.state) == len(survive[0])

    p2.x += limit_a + 1e-6
    ret = el.track(p2)
    assert ret == "All particles lost"


def test_track_LimitRectEllipse():
    limit_a = 0.1
    limit_b = 0.2
    max_x = 0.1
    max_y = 0.05
    el = pysixtrack.elements.LimitRectEllipse(
        max_x=max_x, max_y=max_y, a=limit_a, b=limit_b
    )

    p1 = pysixtrack.Particles()
    p1.x = 1
    p1.y = 1
    ret = el.track(p1)
    assert ret == "Particle lost"

    arr = np.arange(0, 1, 0.001)
    p2 = pysixtrack.Particles(x=arr, y=arr)
    survive = np.where(
        (p2.x ** 2 / limit_a ** 2 + p2.y ** 2 / limit_b ** 2 <= 1.0)
        & (p2.x >= -max_x)
        & (p2.x <= max_x)
        & (p2.y >= -max_y)
        & (p2.y <= max_y)
    )
    ret = el.track(p2)
    assert len(p2.state) == len(survive[0])

    p2.x += limit_a + 1e-6
    ret = el.track(p2)
    assert ret == "All particles lost"
