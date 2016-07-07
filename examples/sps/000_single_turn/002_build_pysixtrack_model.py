import sys, os
BIN = os.path.expanduser("../../../../")#location of pyoptics
sys.path.append(BIN)
BIN = os.path.expanduser("../../../")#location of pysixtrack
sys.path.append(BIN)

n_ele_check_against_mad = 10

import numpy as np

from pyoptics import madlang,optics

#see sps/madx/a001_track_thin.madx
mad_sps=madlang.open('SPS_Q20_thin.seq')

# same long settings that will be used later to track in mad
mad_sps.acta_31637.volt=4.5 
mad_sps.acta_31637.lag=0.5

import pysixtrack

# build seq with info for pysixtrack 
sps_seq,rest=mad_sps.sps.expand_struct(pysixtrack.convert)
if len(rest)>0: raise ValueError('len(rest)>0')

# extract the list of names and elements - for dummies :-) 
sps_elems = []
sps_names = []
for item in sps_seq:
    sps_elems.append(item[1])
    sps_names.append(item[0])
    

# build a particle (actually a bunch of with one particle)
particle=pysixtrack.Bunch(x=1e-3,px=np.zeros(1),
                   y=-0.5e-3,py=np.zeros(1),
                   tau=0.74,pt=np.zeros(1),
                   e0=26.01692438e9, m0=0.93827205e9)
                   
#track the particle
for i_ele, ele in enumerate(sps_elems):
    ele.track(particle)
    
    #~ if 

    


