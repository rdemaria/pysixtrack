import os
import sys
from importlib import util

import pysixtrack
import pysixtrack.be_beamfields.tools as bt


def test_madx_import():
    cpymad_spec = util.find_spec("cpymad")
    if cpymad_spec is None:
        print("cpymad is not available - abort test")
        sys.exit(0)

    from cpymad.madx import Madx

    seq_name = "psb1"
    use_aperture = True

    n_SCkicks = 120
    length_fuzzy = 0.0
    p0c = 0.571e6
    particle = pysixtrack.Particles(p0c=p0c)
    betagamma = particle.beta0 * particle.gamma0
    # mass = pysixtrack.Particles.pmass
    delta_rms = 1e-3
    neps_x = 1.5e-6
    neps_y = 1.5e-6

    # for space charge bunched
    number_of_particles = 1e11
    bunchlength_rms = 1.0

    # for space charge coasting
    line_density = 1e11

    for sc_mode in ["Bunched", "Coasting"]:

        mad = Madx()
        mad.options.echo = False
        mad.options.info = False
        mad.warn = False
        file_path = os.path.realpath(__file__)
        path = os.path.dirname(file_path) + "/psb/"
        mad.call(path + "psb_fb_lhc.madx", chdir=True)

        # Determine space charge locations
        temp_line = pysixtrack.Line.from_madx_sequence(mad.sequence[seq_name])
        sc_locations, sc_lengths = bt.determine_sc_locations(
            temp_line, n_SCkicks, length_fuzzy
        )

        # Install spacecharge place holders
        sc_names = ["sc%d" % number for number in range(len(sc_locations))]
        bt.install_sc_placeholders(
            mad, seq_name, sc_names, sc_locations, mode=sc_mode
        )

        # Generate line with spacecharge
        line = pysixtrack.Line.from_madx_sequence(
            mad.sequence[seq_name], install_apertures=use_aperture
        )

        # Get sc info from optics
        mad_sc_names, sc_twdata = bt.get_spacecharge_names_twdata(
            mad, seq_name, mode=sc_mode
        )

        # Check consistency
        if sc_mode == "Bunched":
            sc_elements, sc_names = line.get_elements_of_type(
                pysixtrack.elements.SpaceChargeBunched
            )
        elif sc_mode == "Coasting":
            sc_elements, sc_names = line.get_elements_of_type(
                pysixtrack.elements.SpaceChargeCoasting
            )
        else:
            raise ValueError("mode not understood")
        bt.check_spacecharge_consistency(
            sc_elements, sc_names, sc_lengths, mad_sc_names
        )

        # Setup spacecharge in the line
        if sc_mode == "Bunched":
            bt.setup_spacecharge_bunched_in_line(
                sc_elements,
                sc_lengths,
                sc_twdata,
                betagamma,
                number_of_particles,
                bunchlength_rms,
                delta_rms,
                neps_x,
                neps_y,
            )
        elif sc_mode == "Coasting":
            bt.setup_spacecharge_coasting_in_line(
                sc_elements,
                sc_lengths,
                sc_twdata,
                betagamma,
                line_density,
                delta_rms,
                neps_x,
                neps_y,
            )
        else:
            raise ValueError("mode not understood")
