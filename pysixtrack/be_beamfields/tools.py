import numpy as np
import pysixtrack
from pysixtrack import MadPoint

_sigma_names = [11, 12, 13, 14, 22, 23, 24, 33, 34, 44]
_beta_names = ["betx", "bety"]


def norm(v):
    return np.sqrt(np.sum(v ** 2))


def get_points_twissdata_for_elements(
    ele_names, mad, seq_name, use_survey=True, use_twiss=True
):

    mad.use(sequence=seq_name)
    mad.twiss()

    if use_survey:
        mad.survey()

    bb_xyz_points = []
    bb_twissdata = {
        kk: []
        for kk in _sigma_names
        + _beta_names
        + "dispersion_x dispersion_y x y".split()
    }
    for eename in ele_names:
        bb_xyz_points.append(
            MadPoint(
                eename + ":1", mad, use_twiss=use_twiss, use_survey=use_survey
            )
        )

        i_twiss = np.where(mad.table.twiss.name == (eename + ":1"))[0][0]

        for sn in _sigma_names:
            bb_twissdata[sn].append(
                getattr(mad.table.twiss, "sig%d" % sn)[i_twiss]
            )

        for kk in ["betx", "bety"]:
            bb_twissdata[kk].append(mad.table.twiss[kk][i_twiss])
        gamma = mad.table.twiss.summary.gamma
        beta = np.sqrt(1.0 - 1.0 / (gamma * gamma))
        for pp in ["x", "y"]:
            bb_twissdata["dispersion_" + pp].append(
                mad.table.twiss["d" + pp][i_twiss] * beta
            )
            bb_twissdata[pp].append(mad.table.twiss[pp][i_twiss])
        # , 'dx', 'dy']:

    return bb_xyz_points, bb_twissdata


def get_elements(seq, ele_type=None, slot_id=None):

    elements = []
    element_names = []
    for ee in seq.elements:

        if ele_type is not None:
            if ee.base_type.name != ele_type:
                continue

        if slot_id is not None:
            if ee.slot_id != slot_id:
                continue

        elements.append(ee)
        element_names.append(ee.name)

    return elements, element_names


def get_points_twissdata_for_element_type(
    mad, seq_name, ele_type=None, slot_id=None, use_survey=True, use_twiss=True
):

    elements, element_names = get_elements(
        seq=mad.sequence[seq_name], ele_type=ele_type, slot_id=slot_id
    )

    points, twissdata = get_points_twissdata_for_elements(
        element_names,
        mad,
        seq_name,
        use_survey=use_survey,
        use_twiss=use_twiss,
    )

    return elements, element_names, points, twissdata


###############################
# beam beam related functions #
###############################
def find_alpha_and_phi(dpx, dpy):

    phi = np.sqrt(dpx ** 2 + dpy ** 2) / 2.0
    if phi < 1e-20:
        alpha = 0.0
    elif np.abs(dpx) >= np.abs(dpy):
        alpha = np.arctan(dpy / dpx)
        if dpx < 0:
            phi = -phi
    else:
        alpha = np.sign(dpy) * (np.pi / 2 - np.abs(np.arctan(dpx / dpy)))
        if dpy < 0:
            phi = -phi

    return alpha, phi


def get_bb_names_madpoints_sigmas(
    mad, seq_name, use_survey=True, use_twiss=True
):
    (
        _,
        element_names,
        points,
        twissdata,
    ) = get_points_twissdata_for_element_type(
        mad,
        seq_name,
        ele_type="beambeam",
        slot_id=None,
        use_survey=use_survey,
        use_twiss=use_twiss,
    )
    sigmas = {kk: twissdata[kk] for kk in _sigma_names}
    return element_names, points, sigmas


def shift_strong_beam_based_on_close_ip(
    points_weak, points_strong, IPs_survey_weak, IPs_survey_strong
):

    for i_bb, _ in enumerate(points_weak):

        pbw = points_weak[i_bb]
        pbs = points_strong[i_bb]

        # Find closest IP
        d_ip = 1e6
        use_ip = 0
        for ip in IPs_survey_weak.keys():
            dd = norm(pbw.p - IPs_survey_weak[ip].p)
            if dd < d_ip:
                use_ip = ip
                d_ip = dd

        # Shift Bs
        shift_ws = IPs_survey_strong[use_ip].p - IPs_survey_weak[use_ip].p
        pbs.p -= shift_ws


def find_bb_separations(points_weak, points_strong, names=None):

    if names is None:
        names = ["bb_%d" % ii for ii in range(len(points_weak))]

    sep_x = []
    sep_y = []
    for i_bb, name_bb in enumerate(names):

        pbw = points_weak[i_bb]
        pbs = points_strong[i_bb]

        # Find vws
        vbb_ws = points_strong[i_bb].p - points_weak[i_bb].p

        # Check that the two reference system are parallel
        try:
            assert norm(pbw.ex - pbs.ex) < 1e-10  # 1e-4 is a reasonable limit
            assert norm(pbw.ey - pbs.ey) < 1e-10  # 1e-4 is a reasonable limit
            assert norm(pbw.ez - pbs.ez) < 1e-10  # 1e-4 is a reasonable limit
        except AssertionError:
            print(name_bb, "Reference systems are not parallel")
            if (
                np.sqrt(
                    norm(pbw.ex - pbs.ex) ** 2
                    + norm(pbw.ey - pbs.ey) ** 2
                    + norm(pbw.ez - pbs.ez) ** 2
                )
                < 5e-3
            ):
                print("Smaller that 5e-3, tolerated.")
            else:
                raise ValueError("Too large! Stopping.")

        # Check that there is no longitudinal separation
        try:
            assert np.abs(np.dot(vbb_ws, pbw.ez)) < 1e-4
        except AssertionError:
            print(name_bb, "The beams are longitudinally shifted")

        # Find separations
        sep_x.append(np.dot(vbb_ws, pbw.ex))
        sep_y.append(np.dot(vbb_ws, pbw.ey))

    return sep_x, sep_y


def setup_beam_beam_in_line(
    line,
    bb_names,
    bb_sigmas_strong,
    bb_points_weak,
    bb_points_strong,
    beta_r_strong,
    bunch_intensity_strong,
    n_slices_6D,
    bb_coupling,
):

    sep_x, sep_y = find_bb_separations(
        points_weak=bb_points_weak,
        points_strong=bb_points_strong,
        names=bb_names,
    )

    i_bb = 0
    assert bb_coupling is False  # Not implemented
    for ee, eename in zip(line.elements, line.element_names):
        if isinstance(ee, pysixtrack.elements.BeamBeam4D):
            assert eename == bb_names[i_bb]

            ee.charge = bunch_intensity_strong
            ee.sigma_x = np.sqrt(bb_sigmas_strong[11][i_bb])
            ee.sigma_y = np.sqrt(bb_sigmas_strong[33][i_bb])
            ee.beta_r = beta_r_strong
            ee.x_bb = sep_x[i_bb]
            ee.y_bb = sep_y[i_bb]

            i_bb += 1
        if isinstance(ee, pysixtrack.elements.BeamBeam6D):
            assert eename == bb_names[i_bb]

            dpx = bb_points_weak[i_bb].tpx - bb_points_strong[i_bb].tpx
            dpy = bb_points_weak[i_bb].tpy - bb_points_strong[i_bb].tpy

            alpha, phi = find_alpha_and_phi(dpx, dpy)

            ee.phi = phi
            ee.alpha = alpha
            ee.x_bb_co = sep_x[i_bb]
            ee.y_bb_co = sep_y[i_bb]
            ee.charge_slices = [bunch_intensity_strong / n_slices_6D]
            ee.zeta_slices = [0.0]
            ee.sigma_11 = bb_sigmas_strong[11][i_bb]
            ee.sigma_12 = bb_sigmas_strong[12][i_bb]
            ee.sigma_13 = bb_sigmas_strong[13][i_bb]
            ee.sigma_14 = bb_sigmas_strong[14][i_bb]
            ee.sigma_22 = bb_sigmas_strong[22][i_bb]
            ee.sigma_23 = bb_sigmas_strong[23][i_bb]
            ee.sigma_24 = bb_sigmas_strong[24][i_bb]
            ee.sigma_33 = bb_sigmas_strong[33][i_bb]
            ee.sigma_34 = bb_sigmas_strong[34][i_bb]
            ee.sigma_44 = bb_sigmas_strong[44][i_bb]

            if not (bb_coupling):
                ee.sigma_13 = 0.0
                ee.sigma_14 = 0.0
                ee.sigma_23 = 0.0
                ee.sigma_24 = 0.0

            i_bb += 1


##################################
# space charge related functions #
##################################
sc_mode_to_slotid = {"Coasting": "1", "Bunched": "2", "Interpolated": "3"}


def determine_sc_locations(line, n_SCkicks, length_fuzzy):
    s_elements = np.array(line.get_s_elements())
    length_target = s_elements[-1] / float(n_SCkicks)
    s_targets = np.arange(0, s_elements[-1], length_target)
    sc_locations = []
    for s in s_targets:
        idx_closest = (np.abs(s_elements - s)).argmin()
        s_closest = s_elements[idx_closest]
        if abs(s - s_closest) < length_fuzzy / 2.0:
            sc_locations.append(s_closest)
        else:
            sc_locations.append(s)
    sc_lengths = np.diff(sc_locations).tolist() + [
        s_elements[-1] - sc_locations[-1]
    ]
    return sc_locations, sc_lengths


def install_sc_placeholders(mad, seq_name, name, s, mode="Bunched"):
    sid = sc_mode_to_slotid[mode]
    mad.input(
        f"""
            seqedit, sequence={seq_name};"""
    )
    for name_, s_ in zip(np.atleast_1d(name), np.atleast_1d(s)):
        mad.input(
            f"""
            {name_} : placeholder, l=0., slot_id={sid};
            install, element={name_}, at={s_:.10e};"""
        )
    mad.input(
        f"""
            flatten;
            endedit;
            use, sequence={seq_name};"""
    )


def get_spacecharge_names_twdata(mad, seq_name, mode):
    _, mad_sc_names, _, twdata = get_points_twissdata_for_element_type(
        mad,
        seq_name,
        ele_type="placeholder",
        slot_id=int(sc_mode_to_slotid[mode]),
        use_survey=False,
        use_twiss=True,
    )
    return mad_sc_names, twdata


def _setup_spacecharge_in_line(
    sc_elements,
    sc_lengths,
    sc_twdata,
    betagamma,
    number_of_particles,
    delta_rms,
    neps_x,
    neps_y,
):

    for ii, ss in enumerate(sc_elements):
        ss.number_of_particles = number_of_particles
        ss.sigma_x = np.sqrt(
            sc_twdata["betx"][ii] * neps_x / betagamma
            + (sc_twdata["dispersion_x"][ii] * delta_rms) ** 2
        )
        ss.sigma_y = np.sqrt(
            sc_twdata["bety"][ii] * neps_y / betagamma
            + (sc_twdata["dispersion_y"][ii] * delta_rms) ** 2
        )
        ss.length = sc_lengths[ii]
        ss.x_co = sc_twdata["x"][ii]
        ss.y_co = sc_twdata["y"][ii]
        ss.enabled = True


def setup_spacecharge_bunched_in_line(
    sc_elements,
    sc_lengths,
    sc_twdata,
    betagamma,
    number_of_particles,
    delta_rms,
    neps_x,
    neps_y,
    bunchlength_rms,
):

    for ii, ss in enumerate(sc_elements):
        ss.bunchlength_rms = bunchlength_rms
    _setup_spacecharge_in_line(
        sc_elements,
        sc_lengths,
        sc_twdata,
        betagamma,
        number_of_particles,
        delta_rms,
        neps_x,
        neps_y,
    )


def setup_spacecharge_coasting_in_line(
    sc_elements,
    sc_lengths,
    sc_twdata,
    betagamma,
    number_of_particles,
    delta_rms,
    neps_x,
    neps_y,
    circumference,
):

    for ii, ss in enumerate(sc_elements):
        ss.circumference = circumference
    _setup_spacecharge_in_line(
        sc_elements,
        sc_lengths,
        sc_twdata,
        betagamma,
        number_of_particles,
        delta_rms,
        neps_x,
        neps_y,
    )


def setup_spacecharge_interpolated_in_line(
    sc_elements,
    sc_lengths,
    sc_twdata,
    betagamma,
    number_of_particles,
    delta_rms,
    neps_x,
    neps_y,
    line_density_profile,
    dz,
    z0,
    method=0,
):
    assert method == 0 or method == 1
    for ii, ss in enumerate(sc_elements):
        ss.line_density_profile = line_density_profile
        ss.dz = dz
        ss.z0 = z0
        ss.method = method
    _setup_spacecharge_in_line(
        sc_elements,
        sc_lengths,
        sc_twdata,
        betagamma,
        number_of_particles,
        delta_rms,
        neps_x,
        neps_y,
    )


def check_spacecharge_consistency(
    sc_elements, sc_names, sc_lengths, mad_sc_names
):
    assert len(sc_elements) == len(mad_sc_names)
    assert len(sc_lengths) == len(mad_sc_names)
    for ii, (ss, nn) in enumerate(zip(sc_elements, sc_names)):
        assert nn == mad_sc_names[ii]
