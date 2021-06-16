import numpy as np

from . import elements as pysixtrack_elements


def iter_from_madx_sequence(
    sequence,
    classes=pysixtrack_elements,
    ignored_madtypes=[],
    exact_drift=False,
    drift_threshold=1e-6,
    install_apertures=False,
):

    if exact_drift:
        myDrift = classes.DriftExact
    else:
        myDrift = classes.Drift
    seq = sequence

    elements = seq.elements
    ele_pos = seq.element_positions()

    old_pp = 0.0
    i_drift = 0
    for ee, pp in zip(elements, ele_pos):

        if pp > old_pp + drift_threshold:
            yield "drift_%d" % i_drift, myDrift(length=(pp - old_pp))
            old_pp = pp
            i_drift += 1

        eename = ee.name
        mad_etype = ee.base_type.name

        # if ee.length > 0:
        #    raise ValueError(f"Sequence {seq} contains {eename} with length>0")

        if mad_etype in [
            "marker",
            "monitor",
            "hmonitor",
            "vmonitor",
            "collimator",
            "rcollimator",
            "elseparator",
            "instrument",
            "solenoid",
            "drift",
        ]:
            newele = myDrift(length=ee.l)
            old_pp += ee.l

        elif mad_etype in ignored_madtypes:
            pass

        elif mad_etype == "multipole":
            knl = ee.knl if hasattr(ee, "knl") else [0]
            ksl = ee.ksl if hasattr(ee, "ksl") else [0]
            newele = classes.Multipole(
                knl=list(knl),
                ksl=list(ksl),
                hxl=knl[0],
                hyl=ksl[0],
                length=ee.lrad,
            )

        elif mad_etype == "tkicker" or mad_etype == "kicker":
            hkick = [-ee.hkick] if hasattr(ee, "hkick") else []
            vkick = [ee.vkick] if hasattr(ee, "vkick") else []
            newele = classes.Multipole(
                knl=hkick, ksl=vkick, length=ee.lrad, hxl=0, hyl=0
            )

        elif mad_etype == "vkicker":
            newele = classes.Multipole(
                knl=[], ksl=[ee.kick], length=ee.lrad, hxl=0, hyl=0
            )

        elif mad_etype == "hkicker":
            newele = classes.Multipole(
                knl=[-ee.kick], ksl=[], length=ee.lrad, hxl=0, hyl=0
            )
        elif mad_etype == "dipedge":
            newele = classes.DipoleEdge(
                h=ee.h, e1=ee.e1, hgap=ee.hgap, fint=ee.fint
            )

        elif mad_etype == "rfcavity":
            newele = classes.Cavity(
                voltage=ee.volt * 1e6,
                frequency=ee.freq * 1e6,
                lag=ee.lag * 360,
            )

        elif mad_etype == "rfmultipole":
            newele = classes.RFMultipole(
                voltage=ee.volt * 1e6,
                frequency=ee.freq * 1e6,
                lag=ee.lag * 360,
                knl=ee.knl[:],
                ksl=ee.ksl[:],
                pn=[v * 360 for v in ee.pnl],
                ps=[v * 360 for v in ee.psl],
            )

        elif mad_etype == "crabcavity":
            newele = classes.RFMultipole(
                frequency=ee.freq * 1e6,
                knl=[ee.volt / sequence.beam.pc],
                pn=[ee.lag * 360 - 90],
            )

        elif mad_etype == "beambeam":
            if ee.slot_id == 6 or ee.slot_id == 60:
                # BB interaction is 6D
                newele = classes.BeamBeam6D(
                    phi=0.0,
                    alpha=0.0,
                    x_bb_co=0.0,
                    y_bb_co=0.0,
                    charge_slices=[0.0],
                    zeta_slices=[0.0],
                    sigma_11=1.0,
                    sigma_12=0.0,
                    sigma_13=0.0,
                    sigma_14=0.0,
                    sigma_22=1.0,
                    sigma_23=0.0,
                    sigma_24=0.0,
                    sigma_33=0.0,
                    sigma_34=0.0,
                    sigma_44=0.0,
                    x_co=0.0,
                    px_co=0.0,
                    y_co=0.0,
                    py_co=0.0,
                    zeta_co=0.0,
                    delta_co=0.0,
                    d_x=0.0,
                    d_px=0.0,
                    d_y=0.0,
                    d_py=0.0,
                    d_zeta=0.0,
                    d_delta=0.0,
                )
            else:
                # BB interaction is 4D
                newele = classes.BeamBeam4D(
                    charge=0.0,
                    sigma_x=1.0,
                    sigma_y=1.0,
                    beta_r=1.0,
                    x_bb=0.0,
                    y_bb=0.0,
                    d_px=0.0,
                    d_py=0.0,
                )
        elif mad_etype == "placeholder":
            if ee.slot_id == 1:
                newele = classes.SCCoasting()
            elif ee.slot_id == 2:
                newele = classes.SCQGaussProfile()
            elif ee.slot_id == 3:
                newele = classes.SCInterpolatedProfile()
            else:
                newele = myDrift(length=ee.l)
                old_pp += ee.l
        else:
            raise ValueError(f'MAD element "{mad_etype}" not recognized')

        yield eename, newele

        if (
            install_apertures
            and hasattr(ee, "aperture")
            and (min(ee.aperture) > 0)
        ):
            if ee.apertype == "rectangle":
                newaperture = pysixtrack_elements.LimitRect(
                    min_x=-ee.aperture[0],
                    max_x=ee.aperture[0],
                    min_y=-ee.aperture[1],
                    max_y=ee.aperture[1],
                )
            elif ee.apertype == "ellipse":
                newaperture = pysixtrack_elements.LimitEllipse(
                    a=ee.aperture[0], b=ee.aperture[1]
                )
            elif ee.apertype == "circle":
                newaperture = pysixtrack_elements.LimitEllipse(
                    a=ee.aperture[0], b=ee.aperture[0]
                )
            elif ee.apertype == "rectellipse":
                newaperture = pysixtrack_elements.LimitRectEllipse(
                    max_x=ee.aperture[0],
                    max_y=ee.aperture[1],
                    a=ee.aperture[2],
                    b=ee.aperture[3],
                )
            else:
                raise ValueError("Aperture type not recognized")

            yield eename + "_aperture", newaperture

    if hasattr(seq, "length") and seq.length > old_pp:
        yield "drift_%d" % i_drift, myDrift(length=(seq.length - old_pp))


class MadPoint(object):
    @classmethod
    def from_survey(cls, name, mad):
        return cls(name, mad, use_twiss=False, use_survey=True)

    @classmethod
    def from_twiss(cls, name, mad):
        return cls(name, mad, use_twiss=True, use_survey=False)

    def __init__(self, name, mad, use_twiss=True, use_survey=True):

        self.use_twiss = use_twiss
        self.use_survey = use_survey

        if not (use_survey) and not (use_twiss):
            raise ValueError(
                "use_survey and use_twiss cannot be False at the same time"
            )

        self.name = name
        if use_twiss:
            twiss = mad.table.twiss
            names = twiss.name
        if use_survey:
            survey = mad.table.survey
            names = survey.name

        idx = np.where(names == name)[0][0]

        if use_twiss:
            self.tx = twiss.x[idx]
            self.ty = twiss.y[idx]
            self.tpx = twiss.px[idx]
            self.tpy = twiss.py[idx]
        else:
            self.tx = None
            self.ty = None
            self.tpx = None
            self.tpy = None

        if use_survey:
            self.sx = survey.x[idx]
            self.sy = survey.y[idx]
            self.sz = survey.z[idx]
            self.sp = np.array([self.sx, self.sy, self.sz])
            theta = survey.theta[idx]
            phi = survey.phi[idx]
            psi = survey.psi[idx]
        else:
            self.sx = None
            self.sy = None
            self.sz = None
            self.sp = None
            theta = 0.0
            phi = 0.0
            psi = 0.0

        thetam = np.array(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ]
        )
        phim = np.array(
            [
                [1, 0, 0],
                [0, np.cos(phi), np.sin(phi)],
                [0, -np.sin(phi), np.cos(phi)],
            ]
        )
        psim = np.array(
            [
                [np.cos(psi), -np.sin(psi), 0],
                [np.sin(psi), np.cos(psi), 0],
                [0, 0, 1],
            ]
        )
        wm = np.dot(thetam, np.dot(phim, psim))
        self.ex = np.dot(wm, np.array([1, 0, 0]))
        self.ey = np.dot(wm, np.array([0, 1, 0]))
        self.ez = np.dot(wm, np.array([0, 0, 1]))

        self.p = np.array([0.0, 0.0, 0.0])

        if use_twiss:
            self.p += self.ex * self.tx + self.ey * self.ty

        if use_survey:
            self.p += self.sp

    def dist(self, other):
        return np.sqrt(np.sum((self.p - other.p) ** 2))

    def distxy(self, other):
        dd = self.p - other.p
        return np.dot(dd, self.ex), np.dot(dd, self.ey)


def mad_benchmark(mtype, attrs, pc=0.2, x=0, px=0, y=0, py=0, t=0, pt=0):
    import pysixtrack
    from cpymad.madx import Madx

    mad = Madx(stdout=False)
    madtype = mad.command[mtype]
    mad.beam(particle="proton", pc=pc)
    madtype.clone("mm", **attrs)
    mad.input("bench: line=(mm)")
    mad.use(sequence="bench")
    mad.track(onepass=True, dump=False)
    mad.start(x=x, px=px, y=y, py=py, t=t, pt=pt)
    mad.run()
    mad.endtrack()
    p_mad = pysixtrack.Particles.from_madx_track(mad)
    p_six = p_mad.copy(0)
    line = pysixtrack.Line.from_madx_sequence(
        mad.sequence.bench, exact_drift=True
    )
    line.track(p_six)
    p_mad.copy(-1).compare(p_six, rel_tol=0)
    return mad, line, p_mad, p_six
