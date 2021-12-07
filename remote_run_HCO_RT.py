# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 22:33:55 2021

@author: Rafi
"""
import numpy as np
from scipy.integrate import solve_ivp
import time

from network_and_neuron import Synapse, Neuron, Network
from network_odes import main, no_observer

# Initial conditions - HCO Reference Tracking
x_0 = [0,0,0,0,0,0,0,0,0,0,0,0]; # V, m, h, mH, mT, hT, mA, hA, mKD, mL, mCa, s
x̂_0 = [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
θ̂_0 = [1, 1, 1, 1]; # Estimating gNa, gKD, gleak and gsyn
P_0 = np.eye(4);
Ψ_0 = [0,0,0,0];
to_estimate = np.array([0, 4, 8])
estimate_g_syns = True
estimate_g_res = False # TODO: Need to write the code for this!!

syn = Synapse(1., 1)
syn2 = Synapse(1., 0)
# Remember, order of currents is Na, H, T, A, KD, L, KCA, leak
neur_one = Neuron(0.1, np.array([120.,0.1,2.,0,80.,0.4,2.,0.,0.1]), np.array([syn]))
neur_two = Neuron(0.1, np.array([120.,0.1,2.,0,80.,0.4,2.,0.,0.1]), np.array([syn2]))
network = Network([neur_one, neur_two], np.zeros((2,2))) # for ref tracking
# ref_gs = np.array([[120,36,0.3,2],[120,72,0.3,2]]).T # gs of reference network.
ref_gs = np.array([[110.,0.09,3.,0,70.,0.5,1.7,0.,0.1],
                    [110.,0.09,3.,0,70.,0.5,1.7,0.,0.1]]).T # gs of reference network.
# orig_gs = np.array([ [130.,43.,0.4,2.], [100.,27.,0.2,2.] ]).T # gs of network, for the csv

# Iapp = lambda t : 2 + np.sin(2*np.pi/10*t)
Iconst = lambda t: -2
Iapps = [Iconst, Iconst]

# Observer parameters
α = 0.05 # Default is 0.5. Had to decrease as P values were exploding.
γ = 70 # Default is 70, though Thiago's since lowered to 5. But 5 was causing psi to explode.

control_law = ["RefTrack", ref_gs]
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
z_0[0] = -70.

# %%
# Integration initial conditions and parameters
dt = 0.01
Tfinal = 60. # Textbook notebook has 1800.

tspan = (0.,Tfinal)
# controller_on = True
control_start_time = 100.
p = (Iapps,network,(α,γ),to_estimate,num_estimators,control_law,
     estimate_g_syns,estimate_g_res,control_start_time)

start_time = time.time()
out = solve_ivp(lambda t, z: main(t, z, p), tspan, z_0,rtol=1e-6,atol=1e-6,
                t_eval=np.linspace(0,Tfinal,int(Tfinal/dt)), method='BDF')
end_time = time.time()
print("Simulation time: {}s".format(end_time-start_time))

t = out.t
sol = out.y

# # Let's calculate Isyn and Isyn_hat
# v = sol[0,:]
# Isyn = syn.g * sol[11,:] * (v - neur_one.Esyn)
# Isyn_hat = sol[27,:] * sol[23,:] * (v - neur_one.Esyn)

# %%
# Reference Tracking
# syn_ref = Synapse(2.5, 1)
# syn2_ref = Synapse(1., 0)

# the original, for comparison: [120.,0.1,2.,0,80.,0.4,2.,0.,0.1]
neur_one_ref = Neuron(0.1, np.array([110.,0.09,3.,0,70.,0.5,1.7,0.,0.1]), np.array([syn]))
neur_two_ref = Neuron(0.1, np.array([110.,0.09,3.,0,70.,0.5,1.7,0.,0.1]), np.array([syn2]))
network_ref = Network([neur_one_ref, neur_two_ref], np.zeros((2,2)))
p_ref = (Iapps, network_ref)
z_0_ref = np.concatenate((x_0, x_0))
# z_0_ref = x_0
z_0_ref[0] = -70.

start_time = time.time()
out_ref = solve_ivp(lambda t, z: no_observer(t, z, p_ref), tspan, z_0_ref,rtol=1e-6,atol=1e-6,
                t_eval=np.linspace(0,Tfinal,int(Tfinal/dt)))
end_time = time.time()
print("'Ref' Simulation time: {}s".format(end_time-start_time))

t_ref = out_ref.t
sol_ref = out_ref.y

# PHASE SHIFTS:
# For single neur RT, phase shift was 11364.

# %%
# Extract variables and label them
V_idxs = np.array(list(range(num_neurs)))*(len(sol)/num_neurs)
V_idxs = V_idxs.astype(int)
Vs = sol[V_idxs,:]

# %%
# To find peaks.
# from scipy.signal import find_peaks
# find_peaks(x) gives the idxs. Then can use np.roll for the phase-shift.

# For HCO_RT it's about 1105, ie np.roll(x, 1105). Remember the spike is every other local max.

# %%
t=t.astype('float32')
sol=sol.astype('float32')
sol_ref=sol_ref.astype('float32')
np.savez("fullmodel_HCO_RT.npz", t=t,sol=sol,sol_ref=sol_ref)
