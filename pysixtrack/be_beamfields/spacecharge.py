from pysixtrack.base_classes import Element
from .gaussian_fields import get_Ex_Ey_Gx_Gy_gauss


class SpaceChargeCoasting(Element):
    """Space charge for a coasting beam"""

    _description = [
        ("line_density", "1/m", "Number of particles per unit length", 0.0),
        ("sigma_x", "m", "Horizontal size of the beam (r.m.s.)", 1.0),
        ("sigma_y", "m", "Vertical size of the beam (r.m.s.)", 1.0),
        ("length", "m", "Integration length of space charge kick", 0.0),
        ("x_co", "m", "Horizontal closed orbit offset", 0.0),
        ("y_co", "m", "Vertical closed orbit offset", 0),
    ]
    _extra = [
        ("min_sigma_diff", "m", "Threshold to detect round beam", 1e-30),
        ("enabled", "", "Switch to disable space charge effect", True),
    ]

    def track(self, p):
        if self.enabled:
            length = self.length
            sigma_x = self.sigma_x
            sigma_y = self.sigma_y

            charge = p.q0 * p.echarge
            x = p.x - self.x_co
            px = p.px
            y = p.y - self.y_co
            py = p.py

            chi = p.chi

            beta = p.beta0 / p.rvv
            p0c = p.p0c * p.echarge

            Ex, Ey = get_Ex_Ey_Gx_Gy_gauss(
                x,
                y,
                sigma_x,
                sigma_y,
                min_sigma_diff=1e-10,
                skip_Gs=True,
                mathlib=p._m,
            )

            fact_kick = (
                chi
                * self.line_density
                * (charge * p.qratio)
                * charge
                * (1 - p.beta0 * beta)
                / (p0c * beta)
                * length
            )

            px += fact_kick * Ex
            py += fact_kick * Ey

            p.px = px
            p.py = py


class SpaceChargeBunched(Element):
    _description = [
        ("number_of_particles", "", "Number of particles in the bunch", 0.0),
        ("bunchlength_rms", "m", "Length of the bunch (r.m.s.)", 1.0),
        ("sigma_x", "m", "Horizontal size of the beam (r.m.s.)", 1.0),
        ("sigma_y", "m", "Vertical size of the beam (r.m.s.)", 1.0),
        ("length", "m", "Integration length of space charge kick", 0.0),
        ("x_co", "m", "Horizontal closed orbit offset", 0.0),
        ("y_co", "m", "Vertical closed orbit offset", 0),
    ]
    _extra = [
        ("min_sigma_diff", "m", "Threshold to detect round beam", 1e-30),
        ("enabled", "", "Switch to disable space charge effect", True),
    ]

    def track(self, p):
        if self.enabled:
            pi = p._m.pi
            exp = p._m.exp
            sqrt = p._m.sqrt
            bunchlength_rms = self.bunchlength_rms
            length = self.length
            sigma_x = self.sigma_x
            sigma_y = self.sigma_y

            charge = p.q0 * p.echarge
            x = p.x - self.x_co
            px = p.px
            y = p.y - self.y_co
            py = p.py
            sigma = p.sigma

            chi = p.chi

            beta = p.beta0 / p.rvv
            p0c = p.p0c * p.echarge

            Ex, Ey = get_Ex_Ey_Gx_Gy_gauss(
                x,
                y,
                sigma_x,
                sigma_y,
                min_sigma_diff=1e-10,
                skip_Gs=True,
                mathlib=p._m,
            )

            fact_kick = (
                chi
                * (charge * p.qratio)
                * charge
                * (1 - p.beta0 * beta)
                / (p0c * beta)
                * length
            )

            fact_kick *= (
                self.number_of_particles
                / (bunchlength_rms * sqrt(2 * pi))
                * exp(
                    -0.5
                    * (sigma / bunchlength_rms)
                    * (sigma / bunchlength_rms)
                )
            )

            px += fact_kick * Ex
            py += fact_kick * Ey

            p.px = px
            p.py = py
