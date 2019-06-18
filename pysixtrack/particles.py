import numpy as np

# attaching faddeeva to np
from scipy.special import wofz


def wfun(z_re, z_im):
    w = wofz(z_re + 1j * z_im)
    return w.real, w.imag


np.wfun = wfun


def count_not_none(*lst):
    return len(lst) - sum(p is None for p in lst)


class Particles(object):
    clight = 299792458
    pi = 3.141592653589793238
    echarge = 1.602176565e-19
    emass = 0.510998928e6
    # Was: 938.272046e6;
    # correct value acc. to PDG 2018 938.2720813(58)e6 MeV/c²
    pmass = 938.272081e6
    epsilon0 = 8.854187817e-12
    mu0 = 4e-7 * pi
    eradius = echarge**2 / (4 * pi * epsilon0 * emass * clight**2)
    pradius = echarge**2 / (4 * pi * epsilon0 * pmass * clight**2)
    anumber = 6.02214129e23
    kboltz = 1.3806488e-23

    def _g1(self, mass0, p0c, energy0):
        beta0 = p0c / energy0
        gamma0 = energy0 / mass0
        return mass0, beta0, gamma0, p0c, energy0

    def _g2(self, mass0, beta0, gamma0):
        energy0 = mass0 * gamma0
        p0c = energy0 * beta0
        return mass0, beta0, gamma0, p0c, energy0

    def _f1(self, mass0, p0c):
        sqrt = self._m.sqrt
        energy0 = sqrt(p0c**2 + mass0**2)
        return self._g1(mass0, p0c, energy0)

    def _f2(self, mass0, energy0):
        sqrt = self._m.sqrt
        p0c = sqrt(energy0**2 - mass0**2)
        return self._g1(mass0, p0c, energy0)

    def _f3(self, mass0, beta0):
        sqrt = self._m.sqrt
        gamma0 = 1 / sqrt(1 - beta0**2)
        return self._g2(mass0, beta0, gamma0)

    def _f4(self, mass0, gamma0):
        sqrt = self._m.sqrt
        beta0 = sqrt(1 - 1 / gamma0**2)
        return self._g2(mass0, beta0, gamma0)

    def copy(self):
        p = Particles()
        for k, v in list(self.__dict__.items()):
            if type(v) in [np.ndarray, dict]:
                v = v.copy()
            p.__dict__[k] = v
        return p

    def __init__ref(self, p0c, energy0, gamma0, beta0):
        not_none = count_not_none(beta0, gamma0, p0c, energy0)
        if not_none == 0:
            p0c = 1e9
            not_none = 1
            #raise ValueError("Particles defined without energy reference")
        if not_none == 1:
            if p0c is not None:
                new = self._f1(self.mass0, p0c)
                self._update_ref(*new)
            elif energy0 is not None:
                new = self._f2(self.mass0, energy0)
                self._update_ref(*new)
            elif gamma0 is not None:
                new = self._f4(self.mass0, gamma0)
                self._update_ref(*new)
            elif beta0 is not None:
                new = self._f3(self.mass0, beta0)
                self._update_ref(*new)
        else:
            raise ValueError(f"""\
            Particles defined with multiple energy references:
            p0c    = {p0c},
            energy0     = {energy0},
            gamma0 = {gamma0},
            beta0  = {beta0}""")

    def __init__delta(self, delta, ptau, psigma):
        not_none = count_not_none(delta, ptau, psigma)
        if not_none == 0:
            self.delta = 0.
        elif not_none == 1:
            if delta is not None:
                self.delta = delta
            elif ptau is not None:
                self.ptau = ptau
            elif psigma is not None:
                self.psigma = psigma
        else:
            raise ValueError(f"""
            Particles defined with multiple energy deviations:
            delta  = {delta},
            ptau     = {ptau},
            psigma = {psigma}""")

    def __init__zeta(self, zeta, tau, sigma):
        not_none = count_not_none(zeta, tau, sigma)
        if not_none == 0:
            self.zeta = 0.
        elif not_none == 1:
            if zeta is not None:
                self.zeta = zeta
            elif tau is not None:
                self.tau = tau
            elif sigma is not None:
                self.sigma = sigma
        else:
            raise ValueError(f"""\
            Particles defined with multiple time deviations:
            zeta  = {zeta},
            tau   = {tau},
            sigma = {sigma}""")

    def __init__chi(self, mratio, qratio, chi):
        not_none = count_not_none(mratio, qratio, chi)
        if not_none == 0:
            self._chi = 1.
            self._mratio = 1.
            self._qratio = 1.
        elif not_none == 1:
            raise ValueError(f"""\
            Particles defined with insufficient mass/charge information:
            chi    = {chi},
            mratio = {mratio},
            qratio = {qratio}""")
        elif not_none == 2:
            if chi is None:
                self._mratio = mratio
                self.qratio = qratio
            elif mratio is None:
                self._chi = chi
                self.qratio = qratio
            elif qratio is None:
                self._chi = chi
                self.mratio = mratio
        else:
            raise ValueError(f"""
            Particles defined with multiple mass/charge information:
            chi    = {chi},
            mratio = {mratio},
            qratio = {qratio}""")

    def __init__(self,
                 s=0., x=0., px=0., y=0., py=0.,
                 delta=None, ptau=None, psigma=None, rvv=None,
                 zeta=None, tau=None, sigma=None,
                 mass0=pmass, q0=1.,
                 p0c=None, energy0=None, gamma0=None, beta0=None,
                 chi=None, mratio=None, qratio=None,
                 partid=None, turn=None, state=None, elemid=None,
                 mathlib=np, **args):

        self._m = mathlib
        self.s = s
        self.x = x
        self.px = px
        self.y = y
        self.py = py
        self.zeta = zeta
        self._mass0 = mass0
        self.q0 = q0
        self._update_coordinates = False
        self.__init__ref(p0c, energy0, gamma0, beta0)
        self.__init__delta(delta, ptau, psigma)
        self.__init__zeta(zeta, tau, sigma)
        self.__init__chi(chi, mratio, qratio)
        self._update_coordinates = True
        self.partid = partid
        self.turn = turn
        self.state = state

    Px = property(lambda p: p.px * p.p0c * p.mratio)
    Py = property(lambda p: p.py * p.p0c * p.mratio)
    Energy = property(lambda p: (p.ptau * p.p0c + p.energy0) * p.mratio)
    Pc = property(lambda p: (p.delta * p.p0c + p.p0c) * p.mratio)
    mass = property(lambda p: p.mass0 * p.mratio)
    beta = property(lambda p: (1 + p.delta) / (1 / p.beta0 + p.ptau))
    # rvv = property(lambda self: self.beta/self.beta0)
    # rpp = property(lambda self: 1/(1+self.delta))

    rvv = property(lambda self: self._rvv)
    rpp = property(lambda self: self._rpp)

    def add_to_energy(self, energy):
        sqrt = self._m.sqrt
        oldrvv = self._rvv
        deltabeta0 = self.delta * self.beta0
        ptaubeta0 = sqrt(deltabeta0**2 + 2 * deltabeta0 * self.beta0 + 1) - 1
        ptaubeta0 += energy / self.energy0
        ptau = ptaubeta0 / self.beta0
        self._delta = sqrt(ptau**2 + 2 * ptau / self.beta0 + 1) - 1
        self._rvv = (1 + self.delta) / (1 + ptaubeta0)
        self._rpp = 1 / (1 + self.delta)
        self.zeta *= self._rvv / oldrvv

    delta = property(lambda self: self._delta)

    @delta.setter
    def delta(self, delta):
        sqrt = self._m.sqrt
        self._delta = delta
        deltabeta0 = delta * self.beta0
        ptaubeta0 = sqrt(deltabeta0**2 + 2 * deltabeta0 * self.beta0 + 1) - 1
        self._rvv = (1 + self.delta) / (1 + ptaubeta0)
        self._rpp = 1 / (1 + self.delta)

    psigma = property(lambda self: self.ptau / self.beta0)

    @psigma.setter
    def psigma(self, psigma):
        self.ptau = psigma * self.beta0

    tau = property(lambda self: self.zeta / self.beta)

    @tau.setter
    def tau(self, tau):
        self.zeta = self.beta * tau

    sigma = property(lambda self: (self.beta0 / self.beta) * self.zeta)

    @sigma.setter
    def sigma(self, sigma):
        self.zeta = self.beta / self.beta0 * sigma

    @property
    def ptau(self):
        sqrt = self._m.sqrt
        return sqrt(self.delta**2 + 2 * self.delta +
                    1 / self.beta0**2) - 1 / self.beta0

    @ptau.setter
    def ptau(self, ptau):
        sqrt = self._m.sqrt
        self.delta = sqrt(ptau**2 + 2 * ptau / self.beta0 + 1) - 1

    mass0 = property(lambda self: self._mass0)

    @mass0.setter
    def mass0(self, mass0):
        new = self._f1(mass0, self.p0c)
        _abs = self._get_absolute()
        self._update_ref(*new)
        self._update_particles_from_absolute(*_abs)

    beta0 = property(lambda self: self._beta0)

    @beta0.setter
    def beta0(self, beta0):
        new = self._f3(self.mass0, beta0)
        _abs = self._get_absolute()
        self._update_ref(*new)
        self._update_particles_from_absolute(*_abs)

    gamma0 = property(lambda self: self._gamma0)

    @gamma0.setter
    def gamma0(self, gamma0):
        new = self._f4(self.mass0, gamma0)
        _abs = self._get_absolute()
        self._update_ref(*new)
        self._update_particles_from_absolute(*_abs)

    p0c = property(lambda self: self._p0c)

    @p0c.setter
    def p0c(self, p0c):
        new = self._f1(self.mass0, p0c)
        _abs = self._get_absolute()
        self._update_ref(*new)
        self._update_particles_from_absolute(*_abs)

    energy0 = property(lambda self: self._energy0)

    @energy0.setter
    def energy0(self, energy0):
        new = self._f2(self.mass0, energy0)
        _abs = self._get_absolute()
        self._update_ref(*new)
        self._update_particles_from_absolute(*_abs)

    mratio = property(lambda self: self._mratio)

    @mratio.setter
    def mratio(self, mratio):
        self._mratio = mratio
        self._chi = self._qratio / self._mratio

    qratio = property(lambda self: self._qratio)

    @qratio.setter
    def qratio(self, qratio):
        self._qratio = qratio
        self._chi = qratio / self._mratio

    chi = property(lambda self: self._chi)

    @chi.setter
    def chi(self, chi):
        self._qratio = self._chi * self._mratio
        self._chi = chi

    def _get_absolute(self):
        return self.Px, self.Py, self.Pc, self.Energy

    def _update_ref(self, mass0, beta0, gamma0, p0c, energy0):
        self._mass0 = mass0
        self._beta0 = beta0
        self._gamma0 = gamma0
        self._p0c = p0c
        self._energy0 = energy0

    def _update_particles_from_absolute(self, Px, Py, Pc, Energy):
        if self._update_coordinates:
            mratio = self.mass / self.mass0
            norm = mratio*self.p0c
            self._mratio = mratio
            self._chi = self._qratio / mratio
            self._ptau = Energy / norm - 1
            self._delta = Pc / norm - 1
            self.px = Px / norm
            self.py = Py / norm

    def __repr__(self):
        out = f"""\
        mass0   = {self.mass0}
        p0c     = {self.p0c}
        energy0 = {self.energy0}
        beta0   = {self.beta0}
        gamma0  = {self.gamma0}
        s       = {self.s}
        x       = {self.x}
        px      = {self.px}
        y       = {self.y}
        py      = {self.py}
        zeta    = {self.zeta}
        delta   = {self.delta}
        ptau    = {self.ptau}
        mratio  = {self.mratio}
        qratio  = {self.qratio}
        chi     = {self.chi}"""
        return out

    _dict_vars = 's x px y py delta zeta'.split() +\
        'mass0 q0 p0c chi mratio'.split() +\
        'partid turn state'.split()

    def to_dict(self):
        return {kk: getattr(self, kk) for kk in self._dict_vars}

    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)
