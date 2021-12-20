# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 09:12:06 2021

@author: Rafi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

from network_and_neuron import Synapse, Neuron, Network
from network_odes import main, no_observer

Tfinal = 15000. # Textbook notebook has 1800. Simulate to 4k or 6k. Can start slowing at 620.
observe_start_time = 3000.
print("Tfinal = {}".format(Tfinal),file=open("exp1.txt","a"))

# Single neuron reference tracking.
# TODO: can I change the initialisation without 'instability'?
# Same for increasing alpha and decreasing gamma.

# Initial conditions - Single Neuron Reference Tracking
x_0 = [0,0,0,0,0,0,0,0,0,0,0]; # V, m, h, mH, mT, hT, mA, hA, mKD, mL, mCa
x̂_0 = [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
to_observe = np.array([0, 1], dtype=np.int32) # Neurons to observe.
θ̂_0 = np.array([160.,1.,1.,1.,100.,1.,1.,1.,1.,1.]) # np.ones(10); # 1 g_el. From idx 22 to 31,
# or for second neuron, from 142+22.
P_0 = np.eye(10); # From idx 31 I think
Ψ_0 = np.zeros(10);
to_estimate = np.array([0,1,2,3,4,5,6,7,8], dtype=np.int32)
estimate_g_syns_g_els = True

# Remember, order of currents is Na, H, T, A, KD, L, KCA, KIR, leak
neur_one = Neuron(0.1, np.array([0.,0.,0.,0,0.,0.,0.,0.,0.1]), np.array([]), 1)
neur_two = Neuron(0.1, np.array([120.,0.1,2.,0,80.,0.4,2.,0.,0.1]), np.array([]), 1)

res_g = 0.4
is_exp1_res_g = res_g # Hacky...
el_connects = np.array([[res_g, 0, 1]])
network = Network([neur_one, neur_two], el_connects) # for ref tracking

Iconst = lambda t: -2
# Iconstsin = lambda t: -2 + np.sin(2*np.pi/10*t)
Iapps = [Iconst, Iconst]

# Observer parameters
α = 0.0005 # Default is 0.5. Had to decrease as P values were exploding.
γ = 5 # Default is 70, though Thiago's since lowered to 5. But 5 was causing psi to explode.

# gs of reference network. As is_exp1 is set to true, only first neuron will be controlled,
# but the way I've written the code I need to provide a vector for each neuron in the network.
# ref_gs = np.array([[120.,0.1,2.,0,80.,0.4,2.,0.,0.1], [120.,0.1,2.,0,80.,0.4,2.,0.,0.1]]).T
ref_gs = np.array([]).T

is_exp1 = True # Have created controller specifically for eq 1. Easier that way.
control_law = ["RefTrack", ref_gs, is_exp1]
# control_law = [""]

num_neurs = len(network.neurons)
num_estimators = len(θ̂_0)
len_neur_state = network.neurons[0].NUM_GATES + 1
max_num_syns = network.max_num_syns

# Assuming each neuron initialised the same. If not, could use np.ravel()
# and np.reshape()
z_0 = np.zeros(((len_neur_state+max_num_syns)*2+
                num_estimators*2+num_estimators**2,num_neurs))
tmp = np.concatenate((x_0, x̂_0, θ̂_0, P_0.flatten(), Ψ_0))
for j in range(num_neurs): z_0[:,j] = tmp
z_0 = np.ravel(z_0, order='F')
# z_0[0] = -70.

# %%
# Integration initial conditions and parameters
dt = 0.01
# dt = 0.001

tspan = (0.,Tfinal)
p = (Iapps,network,(α,γ),to_estimate,num_estimators,control_law,
     estimate_g_syns_g_els,observe_start_time,to_observe,is_exp1_res_g)

print("Starting simulation",file=open("exp1.txt","a"))
start_time = time.time()
out = solve_ivp(lambda t, z: main(t, z, p), tspan, z_0,rtol=1e-4,atol=1e-4,
                t_eval=np.linspace(0,Tfinal,int(Tfinal/dt)), method='Radau',
                dense_output=False)
end_time = time.time()
print(out.success, file=open("exp1.txt","a"))
print("Simulation time: {}s".format(end_time-start_time),file=open("exp1.txt","a"))

t = out.t
sol = out.y

#%%
t=t.astype('float32')
sol=sol.astype('float32')
np.savez("exp1_coupled.npz", t=t,sol=sol)
