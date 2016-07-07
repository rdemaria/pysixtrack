import sys, os
BIN = os.path.expanduser("../../../../")#location of pyoptics
sys.path.append(BIN)
BIN = os.path.expanduser("../../../")#location of pysixtrack
sys.path.append(BIN)

madx_exec_path = '../../../../madx'

n_ele_check_against_mad = 200

import numpy as np
import replaceline as rl

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
                   e0=26.01692438e9, m0=0.93827205e9) #same as in MAD file
                   

#prepare lists for coord for storage
class recorded_coords(object):
    def __init__(self, coord_names='x px y py t pt'.split()):
        self.coord_names = coord_names
        for coord in coord_names:
            self.__dict__[coord] = []
    def record(self,part):
        for coord in self.coord_names:
            self.__dict__[coord].append(getattr(part, coord)[0])
    def convert_to_numpy_arrays(self):
        for coord in self.coord_names:
            self.__dict__[coord]=np.array(self.__dict__[coord])

mad_recorded = recorded_coords()
pysixt_recorded = recorded_coords(coord_names='x px y py tau pt'.split())


#track the particle and compare
i_ele_recorded = []; name_ele_recorded = []
for i_ele, ele in enumerate(sps_elems):
    
    if np.mod(i_ele, len(sps_elems)/30)==0:
        print 'Track ele %d/%d'%(i_ele, len(sps_elems))
    
    ele.track(particle)
     
    if np.mod(i_ele, n_ele_check_against_mad)==0:
        name = sps_names[i_ele]
        if name.startswith('drift'):
            continue
        rl.replaceline_and_save(fname='track_thin.madx',
            findln='observe, place =', 
            newline='observe, place = %s;\n'%name )
        
        os.system(madx_exec_path+' < track_thin.madx>trash')
        mad_track_res = optics.open('track.obs0002.p0001')
        
        mad_recorded.record(mad_track_res)
        pysixt_recorded.record(particle)
        
        i_ele_recorded.append(i_ele)
        name_ele_recorded.append(name)

mad_recorded.convert_to_numpy_arrays()
pysixt_recorded.convert_to_numpy_arrays()
pysixt_recorded.t = pysixt_recorded.tau

import pylab as pl    
pl.close('all')    
for i_fig, coord in enumerate(mad_recorded.coord_names):
    pl.figure(i_fig)
    pl.subplot(2,1,1)
    pl.plot(i_ele_recorded, mad_recorded.__dict__[coord], '.-', label='MAD-X')
    pl.plot(i_ele_recorded, pysixt_recorded.__dict__[coord], '.-', label='pysixtrack')
    pl.gca().ticklabel_format(style='sci', scilimits=(0,0),axis='y') 
    pl.ylabel(coord)
    pl.subplot(2,1,2)
    pl.plot(i_ele_recorded, np.abs(mad_recorded.__dict__[coord]-pysixt_recorded.__dict__[coord]), '.-')
    pl.gca().ticklabel_format(style='sci', scilimits=(0,0),axis='y')
    pl.xlabel('# element')
    pl.ylabel('# Error on '+coord)
    

pl.show()
    



