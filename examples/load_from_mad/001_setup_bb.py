import numpy as np

_sigma_names = [11,12,13,14,22,23,24,33,34,44] 

class MadPoint(object):
    def __init__(self,name,mad, add_CO=True):
        self.name=name
        twiss=mad.table.twiss
        survey=mad.table.survey
        idx=np.where(survey.name==name)[0][0]
        self.tx=twiss.x[idx]
        self.ty=twiss.y[idx]
        self.sx=survey.x[idx]
        self.sy=survey.y[idx]
        self.sz=survey.z[idx]
        theta=survey.theta[idx]
        phi=survey.phi[idx]
        psi=survey.psi[idx]
        thetam=np.array([[np.cos(theta) ,           0,np.sin(theta)],
             [          0,           1,         0],
             [-np.sin(theta),           0,np.cos(theta)]])
        phim=np.array([[          1,          0,          0],
            [          0,np.cos(phi)   ,   np.sin(phi)],
            [          0,-np.sin(phi)  ,   np.cos(phi)]])
        psim=np.array([[   np.cos(psi),  -np.sin(psi),          0],
            [   np.sin(psi),   np.cos(psi),          0],
            [          0,          0,          1]])
        wm=np.dot(thetam,np.dot(phim,psim))
        self.ex=np.dot(wm,np.array([1,0,0]))
        self.ey=np.dot(wm,np.array([0,1,0]))
        self.ez=np.dot(wm,np.array([0,0,1]))
        self.sp=np.array([self.sx,self.sy,self.sz])
        if add_CO:
            self.p=self.sp+ self.ex * self.tx + self.ey * self.ty
        else:
            self.p=self.sp

    def dist(self,other):
        return np.sqrt(np.sum((self.p-other.p)**2))
    def distxy(self,other):
        dd=self.p-other.p
        return np.dot(dd,self.ex),np.dot(dd,self.ey)

def get_bb_names_xyz_points_sigma_matrices(mad, seq_name):
    mad.use(sequence=seq_name);
    mad.twiss()
    mad.survey()
    
    seq = mad.sequence[seq_name]
   
    bb_names = []
    bb_xyz_points = []
    bb_sigmas = {kk:[] for kk in _sigma_names}
    
    for ee in seq.elements:
        if ee.base_type.name == 'beambeam':
            eename = ee.name
            bb_names.append(eename)
            bb_xyz_points.append(MadPoint(eename+':1', mad))

            i_twiss = np.where(mad.table.twiss.name==(eename+':1')[0][0]
            
            for sn in _sigma_names:
                bb_sigmas[sn].append(getattr(mad.table.twiss, 'sig%d'%sn)[i_twiss]

    return bb_names, bb_xyz_points, bb_sigmas

def norm(v):
    return np.sqrt(np.sum(v**2))


from cpymad.madx import Madx
import pysixtrack

mad=Madx()
mad.options.echo=False;
mad.options.warn=False;
mad.options.info=False;


# Load sequence
mad.call('mad/lhcwbb.seq')

# Parameters to be cross-checked
n_slices = 11


# Disable beam-beam kicks 
mad.globals.on_bb_charge = 0.

ip_names = [1, 2, 5, 8]

# Retrieve geometry information

# IP locations
mad.use('lhcb1'); mad.twiss(); mad.survey()
IP_xyz_b1 = {}
for ip in ip_names:
    IP_xyz_b1[ip] = MadPoint('ip%d'%ip+':1', mad, add_CO=False)

mad.use('lhcb2'); mad.twiss(); mad.survey()
IP_xyz_b2 = {}
for ip in ip_names:
    IP_xyz_b2[ip] = MadPoint('ip%d'%ip+':1', mad, add_CO=False)

# Beam-beam names and locations
bb_names_b1, bb_xyz_b1 = get_bb_names_and_xyz_points(
        mad, seq_name='lhcb1')
bb_names_b2, bb_xyz_b2 = get_bb_names_and_xyz_points(
        mad, seq_name='lhcb2')

# Check naming convention
assert len(bb_names_b1)==len(bb_names_b2)
for nbb1, nbb2 in zip(bb_names_b1, bb_names_b2):
    assert(nbb1==nbb2.replace('b2_','b1_'))

# Check that reference system are parallel at the IPs
for ip in ip_names:
    assert(1. - np.dot(IP_xyz_b1[ip].ex, IP_xyz_b2[ip].ex) <1e-12)
    assert(1. - np.dot(IP_xyz_b1[ip].ey, IP_xyz_b2[ip].ey) <1e-12)
    assert(1. - np.dot(IP_xyz_b1[ip].ez, IP_xyz_b2[ip].ez) <1e-12)

# Shift B2 so that survey is head-on at the closest IP
# and find vector separation
sep_x = []
sep_y = []
for i_bb, name_bb in enumerate(bb_names_b1):
    
    pb1 = bb_xyz_b1[i_bb]
    pb2 = bb_xyz_b2[i_bb]
    
    # Find closest IP
    d_ip = 1e6
    use_ip = 0
    for ip in ip_names:
        dd = norm(pb1.p - IP_xyz_b1[ip].p)
        if dd < d_ip:
            use_ip = ip
            d_ip = dd

    # Shift B2
    shift_12 = IP_xyz_b2[use_ip].p - IP_xyz_b1[use_ip].p
    pb2.p -= shift_12

    # Find v12
    vbb_12 = bb_xyz_b2[i_bb].p - bb_xyz_b1[i_bb].p

    # Check that the two reference system are parallel
    try:
        assert(norm(pb1.ex-pb2.ex)<1e-10) #1e-4 is a reasonable limit
        assert(norm(pb1.ey-pb2.ey)<1e-10) #1e-4 is a reasonable limit
        assert(norm(pb1.ez-pb2.ez)<1e-10) #1e-4 is a reasonable limit
        ex, ey, ex = pb1.ex, pb1.ey, pb1.ez
    except AssertionError:
        print(name_bb, 'Reference systems are not parallel')
        # Check that there is separatio in the survey (we are in the D1)
        assert(norm(pb1.sp - pb2.sp)>1e-3)
        
        # Go to baricentric reference
        ex = (pb1.sp - pb2.sp)
        ex = ex/norm(ex)
        if np.dot(ex, pb1.ex)<0.: ex = -ex
        ey = pb1.ey
        ez = np.cross(ex, ey)

    # Check that there is no longitudinal separation
    try:
        assert(np.abs(np.dot(vbb_12, pb1.ez))<1e-4)
    except AssertionError:
        print(name_bb, 'The beams are longitudinally shifted')

    # Find separations
    sep_x.append(np.dot(vbb_12, ex))
    sep_y.append(np.dot(vbb_12, ey))

    

import matplotlib.pyplot as plt
plt.close('all')
fig1 = plt.figure(1)

plt.plot(           [pb.p[0] for pb in bb_xyz_b1],
                    [pb.p[2] for pb in bb_xyz_b1], 'b.')
plt.quiver(np.array([pb.p[0] for pb in bb_xyz_b1]),
           np.array([pb.p[2] for pb in bb_xyz_b1]), 
          np.array([pb.ex[0] for pb in bb_xyz_b1]),
          np.array([pb.ex[2] for pb in bb_xyz_b1]))
plt.quiver(np.array([pb.p[0] for pb in bb_xyz_b1]),
           np.array([pb.p[2] for pb in bb_xyz_b1]), 
          np.array([pb.ez[0] for pb in bb_xyz_b1]),
          np.array([pb.ez[2] for pb in bb_xyz_b1]))

plt.plot(           [pb.p[0] for pb in bb_xyz_b2],
                    [pb.p[2] for pb in bb_xyz_b2], 'r.')
plt.quiver(np.array([pb.p[0] for pb in bb_xyz_b2]),
           np.array([pb.p[2] for pb in bb_xyz_b2]), 
          np.array([pb.ex[0] for pb in bb_xyz_b2]),
          np.array([pb.ex[2] for pb in bb_xyz_b2]))
plt.quiver(np.array([pb.p[0] for pb in bb_xyz_b2]),
           np.array([pb.p[2] for pb in bb_xyz_b2]), 
          np.array([pb.ez[0] for pb in bb_xyz_b2]),
          np.array([pb.ez[2] for pb in bb_xyz_b2]))
plt.show()


#line, other = pysixtrack.Line.from_madx_sequence(mad.sequence.lhcb1)
