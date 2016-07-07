import sys, os
BIN = os.path.expanduser("../../../../")#location of pyoptics
sys.path.append(BIN)
BIN = os.path.expanduser("../../../")#location of pysixtrack
sys.path.append(BIN)

madx_exec_path = '../../../../madx'

n_turns = 512

import numpy as np
import replaceline as rl

from pyoptics import madlang,optics

#load sequence
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

pysixt_recorded = recorded_coords(coord_names='x px y py tau pt'.split())


# Build one turn block
entire_sps = pysixtrack.Block(sps_elems)

# track with pysixtrack
for i_turn in xrange(n_turns):
    print "Turn %d/%d"%(i_turn, n_turns)
    entire_sps.track(particle)
    
    pysixt_recorded.record(particle)
pysixt_recorded.convert_to_numpy_arrays()

# track with MADX
rl.replaceline_and_save(fname='track_thin_multi.madx',
    findln='observe, place =', 
    newline='observe, place = %s;\n'%sps_names[-1] )
    
rl.replaceline_and_save(fname='track_thin_multi.madx',
    findln='run, ', 
    newline='run, turns=%d;\n'%n_turns)

os.system(madx_exec_path+' < track_thin_multi.madx')

mad_recorded = optics.open('track.obs0002.p0001')
mad_recorded.tau = mad_recorded.t

import pylab as pl    
pl.close('all')    
for i_fig, coord in enumerate(pysixt_recorded.coord_names):
    pl.figure(i_fig, figsize = (8,9))
    pl.subplot(3,1,1)
    pl.plot(mad_recorded.__dict__[coord], '.-', label='MAD-X')
    pl.plot(pysixt_recorded.__dict__[coord], '.-', label='pysixtrack')
    pl.gca().ticklabel_format(style='sci', scilimits=(0,0),axis='y') 
    pl.ylabel(coord)
    pl.subplot(3,1,2)
    pl.plot(np.abs(mad_recorded.__dict__[coord]-pysixt_recorded.__dict__[coord]), '.-')
    pl.gca().ticklabel_format(style='sci', scilimits=(0,0),axis='y')
    pl.xlabel('# Turn')
    pl.ylabel('# Error on '+coord)
    pl.subplot(3,1,3)
    spectrum_mad =  np.abs(np.fft.rfft(mad_recorded.__dict__[coord]))
    pl.semilogy(np.linspace(0, 0.5, len(spectrum_mad)),
                   spectrum_mad, '.-', label='MAD-X')
    spectrum_pysixtr =  np.abs(np.fft.rfft(pysixt_recorded.__dict__[coord]))
    pl.semilogy(np.linspace(0, 0.5, len(spectrum_pysixtr)),
                    spectrum_pysixtr, '.-', label='pysixtrack')
    pl.xlabel('Tune')
    

pl.show()



    



