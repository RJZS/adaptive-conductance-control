# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 19:13:57 2021

@author: Rafi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

from network_and_neuron import Synapse, HHModelNeuron, Network
from network_odes import hhmodel_main, hhmodel_no_observer

# TODO:
# Replace neuron model. Want model from upcoming book. That's 'HCO2' in 'online-learning' repo.
    
# CURRENT STATUS:
# Reference tracking seems to work for one neuron, no synapses. But observer estimates of gs do vary!
# Observer estimates don't settle after 600s. But for the 'spiking HCO', error is within +/- 1 of the 
# true value (for both 600s and if extend to 1500s)!
# RefTrack for HCO: Need to apply a phase shift of course. After that, there is still
# a slight, time-varying phase shift between the system
# and the reference system, for every other neuron or so. So still have an error after 600s, and even
# after 1500s (saved figs for this longer data but not the data as not very different from 600s).


# On the backburner:
# Code for graph plotting. Don't just plot everything (messy!), maybe have a 
# parameter which is a list of which to plot. Ie an object, where the keys are fig discriptions 
# and the values are booleans. Then can have each plot within its own 'if' statement.

# Automate running of 'nosyn' simulation (should be called 'nodist'). Harder than I thought!


# Initial conditions - Disturbance Rejection
# x_0 = [0, 0, 0, 0, 0, 0]; # V, m, h, n, s1, s2
# x̂_0 = [-40, 0.2, 0.3, 0.1, 0.4, 0.3] # Works for single neuron.
# x̂_0 = [30, 0.1, 0.2, 0.4, 0.1, 0.15]
# θ̂_0 = [60, 60, 10, 10, 10]; # [gNa, gK, gL, gsyn1, gsyn2]
# P_0 = np.eye(5);
# Ψ_0 = [0, 0, 0, 0, 0];
# to_estimate = [0, 1, 2]
# estimate_g_syns = True
# estimate_g_res = False # TODO: Need to write the code for this!!

# syn = Synapse(2., 1)
# syn2 = Synapse(2., 0)
# syn_dist = Synapse(2., 2)
# neur_one = Neuron(1., [120.,36.,0.3], [syn, syn_dist])
# neur_two = Neuron(1., [120.,36.,0.3], [syn2])
# neur_dist = Neuron(1., [120.,36.,0.3], [])
# network = Network([neur_one, neur_two, neur_dist], np.zeros((3,3)))

# Iapp = lambda t : 2 + np.sin(2*np.pi/10*t)
# Iapps = [Iapp, lambda t: 6, lambda t: 6]

# Initial conditions - Reference Tracking
x_0 = [0, 0, 0, 0, 0]; # V, m, h, n, s
# x̂_0 = [-40, 0.2, 0.3, 0.1] # Works for single neuron.
x̂_0 = [30, 0.1, 0.2, 0.4, 0.5]
θ̂_0 = [60, 10, 10]; # [gNa, gL, gs]
num_estimators = len(θ̂_0)
P_0 = np.eye(num_estimators);
Ψ_0 = np.zeros(num_estimators);

to_estimate = np.array([0, 2])
estimate_g_syns = True
estimate_g_res = False # TODO: Need to write the code for this!!

syn = Synapse(2., 1)
syn2 = Synapse(2., 0)
syn_dist = Synapse(2., 2)
neur_one = HHModelNeuron(1., np.array([130.,43.,0.4]), np.array([syn]))
neur_two = HHModelNeuron(1., np.array([100.,27.,0.2]), np.array([syn2]))
network = Network([neur_one, neur_two], np.zeros((2,2))) # for ref tracking
# ref_gs = np.array([[120,36,0.3,2],[120,72,0.3,2]]).T # gs of reference network.
ref_gs = np.array([[110,35,0.2,2.5],[145,48,0.6,1.]]).T # gs of reference network.
# orig_gs = np.array([ [130.,43.,0.4,2.], [100.,27.,0.2,2.] ]).T # gs of network, for the csv

# Iapp = lambda t : 2 + np.sin(2*np.pi/10*t)
Iapp = lambda t : 6 + np.sin(2*np.pi/10*t)
Iapps = [Iapp, Iapp] # Neuron 2 converges even with constant current?

## FOR TESTING, REMOVE SYNAPSE:
x_0 = [0, 0, 0, 0]; # V, m, h, n
x̂_0 = [-70, 0.5, 0.5, 0.5]
θ̂_0 = [60, 10]; # [gNa, gL]
P_0 = np.eye(2);
Ψ_0 = [0, 0];
neur_one = HHModelNeuron(1., np.array([120.,36.,0.3]), np.array([]))
neur_two = HHModelNeuron(1., [120.,36.,0.3], [])
network = Network([neur_one], np.zeros((1,1)))
to_estimate = [0, 2]
num_estimators = len(θ̂_0)

Iapp = lambda t : 2 + np.sin(2*np.pi/10*t)
Iapps = [Iapp, lambda t: 6] # Neuron 2 converges even with constant current?

ref_gs = np.array([[110,35,0.2]]).T
###########

## Single Neuron Disturbance Rejection
x_0 = [0, 0, 0, 0, 0]; # V, m, h, n, s1
# x̂_0 = [-40, 0.2, 0.3, 0.1, 0.4, 0.3] # Works for single neuron.
x̂_0 = [30, 0.1, 0.2, 0.4, 0.1]
θ̂_0 = [60, 60, 10, 10]; # [gNa, gK, gL, gsyn1]
num_estimators = len(θ̂_0)
P_0 = np.eye(num_estimators);
Ψ_0 = np.zeros(num_estimators);
to_estimate = [0, 1, 2]
estimate_g_syns = True
estimate_g_res = False # TODO: Need to write the code for this!!

syn_dist = Synapse(2., 1)
neur_one = HHModelNeuron(1., [120.,36.,0.3], [syn_dist])
neur_dist = HHModelNeuron(1., [120.,36.,0.3], [])
network = Network([neur_one, neur_dist], np.zeros((2,2)))

Iapp = lambda t : 2 + np.sin(2*np.pi/10*t)
Iapps = [Iapp, lambda t: 6, lambda t: 6]

###########

# Observer parameters
α = 0.5 # Default is 0.5, I've set to 0.3 and then back to 0.5.
γ = 5 # Default is 70, though Thiago's since lowered to 5. Had it at 90 as well.

# For disturbance rejection, the format is ["DistRej", [(neur, syn), (neur, syn), ...]]
# where (neur, syn) is a synapse to be rejected, identified by the index of the neuron in the network,
# and then the index of the synapse in the neuron.
control_law = ["DistRej", [(0, 0)]]#, (0, 1)]] # Need to change when switch between neuron and HCO!
# control_law = ["RefTrack", ref_gs]
# control_law = [""]

num_neurs = len(network.neurons)
len_neur_state = network.neurons[0].NUM_GATES + 1
max_num_syns = network.max_num_syns

# Assuming each neuron initialised the same. If not, could use np.ravel()
# and np.reshape()
z_0 = np.zeros(((len_neur_state+max_num_syns)*2+
                num_estimators*2+num_estimators**2,num_neurs))
tmp = np.concatenate((x_0, x̂_0, θ̂_0, P_0.flatten(), Ψ_0))
for j in range(num_neurs): z_0[:,j] = tmp
z_0 = np.ravel(z_0, order='F')
#z_0 = np.reshape(z_0, (len(z_0),1))

# %%
# Integration initial conditions and parameters
dt = 0.01
Tfinal = 600
tspan = (0.,Tfinal)
# controller_on = True
p = (Iapps,network,(α,γ),to_estimate,num_estimators,control_law,
     estimate_g_syns,estimate_g_res)

start_time = time.time()
out = solve_ivp(lambda t, z: hhmodel_main(t, z, p), tspan, z_0,rtol=1e-6,atol=1e-6,
                t_eval=np.linspace(0,Tfinal,int(Tfinal/dt)), method='BDF')
end_time = time.time()
print("Simulation time: {}s".format(end_time-start_time))

t = out.t
sol = out.y

# %%
# For comparison, need to calculate undisturbed neuron.
# Want to further automate this, but actually fairly tricky...
# def prepare_nodist_sim(neurons, control_law, z_0):
#     for (neur_i, syn_i) in control_law[1]:
        
# # HCO disturbance rejection
# neur_one_nosyn = HHModelNeuron(1., [120.,36.,0.3], [syn])
# neur_two_nosyn = HHModelNeuron(1., [120.,36.,0.3], [syn2])
# network_nosyn = Network([neur_one_nosyn, neur_two_nosyn], np.zeros((2,2)))
# p_nosyn = (Iapps, network_nosyn)
# z_0_nosyn = np.concatenate((x_0[:5], x_0[:5]))
# out_nosyn = solve_ivp(lambda t, z: hhmodel_no_observer(t, z, p_nosyn), tspan, z_0_nosyn,rtol=1e-6,atol=1e-6,
#                 t_eval=np.linspace(0,Tfinal,int(Tfinal/dt)))

# t_nosyn = out_nosyn.t
# sol_nosyn = out_nosyn.y

# Single neuron disturbance rejection
neur_one_nd = HHModelNeuron(1., [120.,36.,0.3], [])
network_nd = Network([neur_one_nd], np.zeros((1,1)))
p_nd = (Iapps, network_nd)
z_0_nd = x_0[:4]
out_nd = solve_ivp(lambda t, z: hhmodel_no_observer(t, z, p_nd), tspan, z_0_nd,rtol=1e-6,atol=1e-6,
                t_eval=np.linspace(0,Tfinal,int(Tfinal/dt)))

t_nd = out_nd.t
sol_nd = out_nd.y

# # Reference Tracking
# syn_ref = Synapse(2.5, 1)
# syn2_ref = Synapse(1., 0)

# neur_one_ref = HHModelNeuron(1., [110.,35.,0.2], np.array([syn_ref]))
# neur_two_ref = HHModelNeuron(1., [145.,48.,0.6], np.array([syn2_ref]))
# network_ref = Network([neur_one_ref, neur_two_ref], np.zeros((2,2)))

# # Removing synapse for test:
# neur_one_ref = HHModelNeuron(1., [110.,35.,0.2], np.array([]))
# network_ref = Network([neur_one_ref], np.zeros((1,1)))

# p_ref = (Iapps, network_ref)
# z_0_ref = np.concatenate((x_0, x_0))
# z_0_ref = x_0
# out_ref = solve_ivp(lambda t, z: hhmodel_no_observer(t, z, p_ref), tspan, z_0_ref,rtol=1e-6,atol=1e-6,
#                 t_eval=np.linspace(0,Tfinal,int(Tfinal/dt)))

# t_ref = out_ref.t
# sol_ref = out_ref.y

# %%
# Test HCO disturbance rejection. First compare real and estimated Isyns.
# v = sol[0,:]
# Isyn = syn.g * sol[4,:] * (v - neur_one.Esyn)
# Isyn_hat = sol[15,:] * sol[10,:] * (v - neur_one.Esyn)
# Isyn_dist = syn_dist.g * sol[5,:] * (v - neur_one.Esyn)
# Isyn_dist_hat = sol[16,:] * sol[11,:] * (v - neur_one.Esyn)

# v2 = sol[0+47,:]
# Isyn2 = syn2.g * sol[4+47,:] * (v2 - neur_two.Esyn)
# Isyn_hat2 = sol[15+47,:] * sol[10+47,:] * (v2 - neur_two.Esyn)

# # Now compare Vs with V_nosyns (misleading name, as there are synapses,
# # just not the disturbance one).
# v_nosyn = sol_nosyn[0,:]
# v2_nosyn = sol_nosyn[5,:]

# %%
# Extract variables and label them
V_idxs = np.array(list(range(num_neurs)))*(len(sol)/num_neurs)
V_idxs = V_idxs.astype(int)
Vs = sol[V_idxs,:]

# %%
# To find peaks.
from scipy.signal import find_peaks
# find_peaks(x) gives the idxs. Then can use np.roll for the phase-shift.
peaks = find_peaks(sol[0,:])
ref_peaks = find_peaks(sol_ref[0,:])

#shift = 19588 - 17479
# shift = 58606 - 59606
shift = 199606 - 197606

# period = 58606 - 55606
# ref_period = 59606 - 56606
period = 199606 - 196606
ref_period = 197606 - 194606
# For HCO_RT it's about 1105, ie np.roll(x, 1105). Remember the spike is every other local max.

#%%
# Construct data for plotting. For single neur.
# First construct Id and Id_hat (synaptic disturbance current).
v = sol[0,:]
v_nd = sol_nd[0,:]
Id = syn_dist.g * sol[4,:] * (v - neur_one.Esyn)
Id_hat = sol[13,:] * sol[9,:] * (v - neur_one.Esyn)

g_ests = sol[10:14,:]
error = sol[0,:] - sol_nd[0,:]

DR_oneneur_data = np.vstack((t, v, v_nd, g_ests, Id, Id_hat, error)).T
np.savetxt("../reports/ifac-data/DR_oneneur_data.txt",DR_oneneur_data,delimiter=' ')

# # Write data to .txt
# from numpy import savetxt
# np.savetxt("filename.txt",var,delimiter=",") # Note Thiago used delimiter " ", but shouldn't matter?

# # Read data from .txt
# from numpy import genfromtxt
# data=genfromtxt('HH_spike.txt',delimiter=",")