import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

from network_and_neuron import Synapse, Neuron, Network
from network_odes import main, no_observer

# Only simulating two neurons at a time, for when I'm using the Mac and so can view the plots directly.

Tfinal= 8000.
dt=0.01
    
Iconst = lambda t: -3.2
Iconst2 = lambda t: -2.5

# From exp2, commit "Changed DistRej max gs to make disturbance impact clearer."
def Ioffset_dist(t): # So two neurons in HCO don't burst simultaneously
    if t < 400:
        return -7.5
    else:
        return -2.3

def Irb(t): # For rebound burster, the one in 'starting point'
    if t < 400 or t > 600:
        return -2.5 # -3.7
    else:
        return -6

def Irb2(t): # For rebound burster
    if t < 400 or t > 600:
        return -3.2 # -3.7
    else:
        return -7
    

Iapps = [Ioffset_dist, Irb2]

tspan = (0.,Tfinal)

# Starting point (at least on 26/01)
# one   = Neuron(0.1, [120.,0.1,1.6,0,80.,0.2,2.,0.,0.1], [], 0)
# two  = Neuron(0.1, [130.,0.1,3.1,0,80.,1.,2.,0.,0.1], [], 0)

# Remember, order of currents is Na, H, T, A, KD, L, KCA, KIR, leak
one   = Neuron(0.1, [60.,0.1,2.,0,80.,0.4,4,0.,0.12], [], 0) # From exp2, commit "Changed DistRej max gs to make disturbance impact clearer." But increased gKCa to decrease period.
two  = Neuron(0.1, [130.,0.1,3.1,0,80.,1.,2.,0.,0.1], [], 0)

x_0 = [0,0,0,0,0,0,0,0,0,0,0] # 0 syns

network_play = Network([one, two], [])
p_play = (Iapps, network_play)

z_0_play = np.concatenate((x_0, x_0))

print("Starting 'play' simulation. Tfinal = {}".format(Tfinal))
start_time = time.time()
out_play = solve_ivp(lambda t, z: no_observer(t, z, p_play), tspan, z_0_play, rtol=2e-3,atol=2e-3,
                t_eval=np.linspace(0,Tfinal,int(Tfinal/dt)), method='LSODA')#, vectorized=True)
end_time = time.time()
print("'Play' simulation time: {}s".format(end_time-start_time))

t = out_play.t
sol = out_play.y

