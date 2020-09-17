import pickle
import numpy as np
import matplotlib.pylab as plt

from cpymad.madx import Madx

import pysixtrack
import pysixtrack.be_beamfields.tools as bt
import footprint


# sc_mode = 'Coasting'
sc_mode = "Bunched"

particle_type = "protons"
# particle_type = "ions"

if particle_type == "protons":

    # For bunched
    number_of_particles = 2e11
    bunchlength_rms = 0.22

    # For coasting
    line_density = 2e11 / 0.5

    mass = pysixtrack.Particles.pmass
    p0c = 25.92e9
    charge_state = 1.0
    neps_x = 2e-6
    neps_y = 2e-6
    delta_rms = 1.5e-3
    V_RF_MV = 4.5
    lag_RF_deg = 180.0
    n_SCkicks = 250
    length_fuzzy = 1.5

if particle_type == "ions":

    # For bunched
    number_of_particles = 3.5e8
    bunchlength_rms = 0.22

    # For coasting
    line_density = 3.5e8 / 0.5

    mass = 193.7e9
    p0c = 1402.406299e9
    charge_state = 82.0
    neps_x = 1.63e-6 
    neps_y = 0.86e-6 
    delta_rms = 1.0e-3
    V_RF_MV = 3 
    lag_RF_deg = 0.0
    n_SCkicks = 250
    length_fuzzy = 1.5


seq_name = "sps"

mad = Madx()
mad.options.echo = False
mad.options.info = False
mad.warn = False
mad.chdir("madx")
mad.call("sps_thin.madx")
mad.use(seq_name)

# Determine space charge locations
temp_line = pysixtrack.Line.from_madx_sequence(mad.sequence.sps)
sc_locations, sc_lengths = bt.determine_sc_locations(
    temp_line, n_SCkicks, length_fuzzy
)

# Install spacecharge place holders
sc_names = ["sc%d" % number for number in range(len(sc_locations))]
bt.install_sc_placeholders(mad, seq_name, sc_names, sc_locations, mode=sc_mode)

# twiss
twtable = mad.twiss()

# Generate line with spacecharge
line = pysixtrack.Line.from_madx_sequence(mad.sequence.sps)

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

# enable RF
i_cavity = line.element_names.index("acta.31637")
line.elements[i_cavity].voltage = V_RF_MV * 1e6
line.elements[i_cavity].lag = lag_RF_deg

# particle on closed orbit
part_on_CO = line.find_closed_orbit(
    guess=[
        twtable["x"][0],
        twtable["px"][0],
        twtable["y"][0],
        twtable["py"][0],
        0.0,
        0.0,
    ],
    p0c=p0c,
    method="get_guess",
)
part_on_CO.q0 = charge_state
part_on_CO.mass0 = mass

# Save particle on CO
with open("particle_on_CO.pkl", "wb") as fid:
    pickle.dump(part_on_CO.to_dict(), fid)

betagamma = part_on_CO.beta0 * part_on_CO.gamma0

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


with open("line.pkl", "wb") as fid:
    pickle.dump(line.to_dict(keepextra=True), fid)


# Save twiss at start ring
with open("twiss_at_start.pkl", "wb") as fid:
    pickle.dump({"betx": twtable.betx[0], "bety": twtable.bety[0]}, fid)


# prepare particles for tracking
r_max_sigma = 10.0
N_r_footp = 10.0
N_theta_footp = 10.0

with open("line.pkl", "rb") as fid:
    line = pysixtrack.Line.from_dict(pickle.load(fid), keepextra=True)

with open("particle_on_CO.pkl", "rb") as fid:
    part_on_CO = pysixtrack.Particles.from_dict(pickle.load(fid))

part = part_on_CO.copy()  

# get beta functions from twiss table
with open("twiss_at_start.pkl", "rb") as fid:
    twiss_at_start = pickle.load(fid)

beta_x = twiss_at_start["betx"]
beta_y = twiss_at_start["bety"]

sigmax = np.sqrt(beta_x * neps_x / part.beta0 / part.gamma0)
sigmay = np.sqrt(beta_y * neps_y / part.beta0 / part.gamma0)

xy_norm = footprint.initial_xy_polar(
    r_min=5e-2,
    r_max=r_max_sigma,
    r_N=N_r_footp + 1,
    theta_min=np.pi / 100,
    theta_max=np.pi / 2 - np.pi / 100,
    theta_N=N_theta_footp,
)

DpxDpy_wrt_CO = np.zeros_like(xy_norm)

for ii in range(xy_norm.shape[0]):
    for jj in range(xy_norm.shape[1]):

        DpxDpy_wrt_CO[ii, jj, 0] = xy_norm[ii, jj, 0] * np.sqrt(
            neps_x / part.beta0 / part.gamma0 / beta_x
        )
        DpxDpy_wrt_CO[ii, jj, 1] = xy_norm[ii, jj, 1] * np.sqrt(
            neps_y / part.beta0 / part.gamma0 / beta_y
        )

with open("DpxDpy_for_footprint.pkl", "wb") as fid:
    pickle.dump({"DpxDpy_wrt_CO": DpxDpy_wrt_CO, "xy_norm": xy_norm}, fid)
