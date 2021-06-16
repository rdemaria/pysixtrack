import matplotlib.pyplot as plt
import numpy as np
import pickle

from cpymad.madx import Madx
import pysixtrack

from pysixtrack.be_beamfields.tools import get_bb_names_madpoints_sigmas
from pysixtrack.be_beamfields.tools import (
    compute_shift_strong_beam_based_on_close_ip,
)
from pysixtrack.be_beamfields.tools import setup_beam_beam_in_line
from pysixtrack import MadPoint

ip_names = [1, 2, 5, 8]

# Parameters to be cross-checked
n_slices = 11

# Use cpymad to compute the required beam parameters
mad = Madx()
mad.options.echo = False
mad.options.warn = False
mad.options.info = False
# Sequences (b1 and b2) with clean machine to compute beam-beam parameters
mad.call("mad/lhcwbb.seq")

# Disable mad beam-beam kicks (we want unperturbed beams)
mad.globals.on_bb_charge = 0.0

# Get IP locations from the survey
mad.use("lhcb1")
mad.twiss()
mad.survey()
IP_xyz_b1 = {}
for ip in ip_names:
    IP_xyz_b1[ip] = MadPoint.from_survey("ip%d" % ip + ":1", mad)

mad.use("lhcb2")
mad.twiss()
mad.survey()
IP_xyz_b2 = {}
for ip in ip_names:
    IP_xyz_b2[ip] = MadPoint.from_survey("ip%d" % ip + ":1", mad)


# Get locations of the bb encounters (absolute from survey), closed orbit
# and orientation of the local reference system (MadPoint objects)
bb_names_b1, bb_xyz_b1, bb_sigmas_b1 = get_bb_names_madpoints_sigmas(
    mad, seq_name="lhcb1"
)
bb_names_b2, bb_xyz_b2, bb_sigmas_b2 = get_bb_names_madpoints_sigmas(
    mad, seq_name="lhcb2"
)

# Get bunch paramenters
bunch_intensity = mad.sequence.lhcb1.beam.npart
gamma_r = mad.sequence.lhcb1.beam.gamma
beta_r = np.sqrt(1 - 1.0 / gamma_r ** 2)

# Check naming convention
assert len(bb_names_b1) == len(bb_names_b2)
for nbb1, nbb2 in zip(bb_names_b1, bb_names_b2):
    assert nbb1 == nbb2.replace("b2_", "b1_")

# Check number of slices
assert (
    len([nn for nn in bb_names_b1 if nn.startswith("bb_ho.l1")])
    == (n_slices - 1) / 2
)

# Correct for small shifts between surveys of the two beams
strong_shifts = compute_shift_strong_beam_based_on_close_ip(
    points_weak=bb_xyz_b1,
    points_strong=bb_xyz_b2,
    IPs_survey_weak=IP_xyz_b1,
    IPs_survey_strong=IP_xyz_b2,
)

# Use another instance of cpymad to load sequence to be used from tracking
# (dispesion knowb ON, multipole error correction, tunes, chromaticity)
mad_ft = Madx()
mad_ft.options.echo = False
mad_ft.options.warn = False
mad_ft.options.info = False
mad_ft.call("mad/lhcwbb_fortracking.seq")
mad_ft.use("lhcb1")  # without this the sequence does not work properly

# Build pysixtrack line
line_for_tracking = pysixtrack.Line.from_madx_sequence(
    mad_ft.sequence["lhcb1"]
)

# Setup 4D and 6D beam beam lenses
setup_beam_beam_in_line(
    line_for_tracking,
    bb_names_b1,
    bb_sigmas_strong=bb_sigmas_b2,
    bb_points_weak=bb_xyz_b1,
    bb_points_strong=bb_xyz_b2,
    bb_shift_strong=strong_shifts,
    beta_r_strong=beta_r,
    bunch_intensity_strong=bunch_intensity,
    n_slices_6D=n_slices,
    bb_coupling=False,
)

# A check
assert (
    np.abs(line_for_tracking.get_length() - mad.sequence.lhcb1.beam.circ)
    < 1e-6
)

# There is a problem in the mask
# (the RF frequancy is wrong in the machine for tracking
# I patch it here -> to be fixed properly!!!!
line_temp = pysixtrack.Line.from_madx_sequence(mad.sequence.lhcb1)
dct_correct_cavities = dict(
    zip(*line_temp.get_elements_of_type(pysixtrack.elements.Cavity)[::-1])
)
for ii, nn in enumerate(line_for_tracking.element_names):
    if nn in dct_correct_cavities.keys():
        line_for_tracking.elements[ii].frequency = dct_correct_cavities[
            nn
        ].frequency

# Save line
with open("line_from_mad.pkl", "wb") as fid:
    pickle.dump(line_for_tracking.to_dict(keepextra=True), fid)

# Plot positions and local reference system at bb interactions
plt.close("all")
fig1 = plt.figure(1)

plt.plot([pb.p[2] for pb in bb_xyz_b1], [pb.p[0] for pb in bb_xyz_b1], "b.")
plt.quiver(
    np.array([pb.p[2] for pb in bb_xyz_b1]),
    np.array([pb.p[0] for pb in bb_xyz_b1]),
    np.array([pb.ex[2] for pb in bb_xyz_b1]),
    np.array([pb.ex[0] for pb in bb_xyz_b1]),
)
plt.quiver(
    np.array([pb.p[2] for pb in bb_xyz_b1]),
    np.array([pb.p[0] for pb in bb_xyz_b1]),
    np.array([pb.ez[2] for pb in bb_xyz_b1]),
    np.array([pb.ez[0] for pb in bb_xyz_b1]),
)

plt.plot([pb.p[2] for pb in bb_xyz_b2], [pb.p[0] for pb in bb_xyz_b2], "r.")
plt.quiver(
    np.array([pb.p[2] for pb in bb_xyz_b2]),
    np.array([pb.p[0] for pb in bb_xyz_b2]),
    np.array([pb.ex[2] for pb in bb_xyz_b2]),
    np.array([pb.ex[0] for pb in bb_xyz_b2]),
)
plt.quiver(
    np.array([pb.p[2] for pb in bb_xyz_b2]),
    np.array([pb.p[0] for pb in bb_xyz_b2]),
    np.array([pb.ez[2] for pb in bb_xyz_b2]),
    np.array([pb.ez[0] for pb in bb_xyz_b2]),
)
plt.show()
