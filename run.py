# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 19:13:57 2021

@author: Rafi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from network_and_neuron import Synapse, Neuron, Network
from network_odes import main, no_observer

# TODO:
# Think about persistent excitation when constant current and NOT estimating c.
# See Thiago's message. To do with constant g_L E_L term. What's the connection with c though?

# Deliverable: HCO successfully rejecting disturbance.
# Code for graph plotting. Don't just plot everything (messy!), maybe have a 
# parameter which is a list of which to plot. Ie an object, where the keys are fig discriptions 
# and the values are booleans. Then can have each plot within its own 'if' statement.

# PROGRESS UPDATE:
# Everything seems to work for one neuron and one disturbance.
# Now on the HCO. Run this code. Adding the neuron for some reason 'breaks' 
# the observation of Isyn (ie the observer estimate for the first synaptic current),
# despite it not having changed from the successful single-synapse case.
# Let's break this down. Hadn't yet tested observing two synapses. Try that (ie without syn2, so no reciprocal inhibition).

# Also in a position to compare simulation output with results from HH_reject.py.
# ie to do the planned 'full test'. <-- THIS WORKS!

# Initial conditions
x_0 = [0, 0, 0, 0, 0, 0]; # V, m, h, n, s1, s2
x̂_0 = [-40, 0.2, 0.3, 0.1, 0.4, 0.3]
θ̂_0 = [60, 60, 10, 10, 10]; # [gNa, gK, gL, gsyn1, gsyn2]
P_0 = np.eye(5);
Ψ_0 = [0, 0, 0, 0, 0];
to_estimate = [0, 1, 2]
estimate_g_syns = True
estimate_g_res = False # TODO: Need to write the code for this!!

syn = Synapse(2., 1)
syn2 = Synapse(2., 0)
syn_dist = Synapse(2., 2)
neur_one = Neuron(1., [120.,36.,0.3], [syn, syn_dist])
neur_two = Neuron(1., [120.,36.,0.3], [syn2])
neur_dist = Neuron(1., [120.,36.,0.3], [])
network = Network([neur_one, neur_two, neur_dist], np.zeros((3,3)))

Iapp = lambda t : 2 + np.sin(2*np.pi/10*t)
Iapps = [Iapp, lambda t: 6, lambda t: 6] # Neuron 2 converges even with constant current?

# ## FOR TESTING, REMOVE SYNAPSE:
# x_0 = [0, 0, 0, 0]; # V, m, h, n
# x̂_0 = [-70, 0.5, 0.5, 0.5]
# θ̂_0 = [50, 10]; # [gNa, gK, gL]
# P_0 = np.eye(2);
# Ψ_0 = [0, 0];
# neur_one = Neuron(1., [120.,36.,0.3], [])
# neur_two = Neuron(1., [120.,36.,0.3], [])
# network = Network([neur_one, neur_two], np.zeros((2,2)))
# to_estimate = [1, 2]

# Iapp = lambda t : 2 + np.sin(2*np.pi/10*t)
# Iapps = [Iapp, lambda t: 6] # Neuron 2 converges even with constant current?

# Observer parameters
α = 0.3 # Default is 0.5, I've set to 0.3.
γ = 90 # Default is 70, though Thiago's since lowered to 5.

# For disturbance rejection, the format is ["DistRej", [(neur, syn), (neur, syn), ...]]
# where (neur, syn) is a synapse to be rejected, identified by the index of the neuron in the network,
# and then the index of the synapse in the neuron.
control_law = ["DistRej", [(0, 0), (0, 1)]]
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

# %%
# Integration initial conditions and parameters
dt = 0.01
Tfinal = 150
tspan = (0.,Tfinal)
# controller_on = True
p = (Iapps,network,(α,γ),to_estimate,num_estimators,control_law,
     estimate_g_syns,estimate_g_res)

out = solve_ivp(lambda t, z: main(t, z, p), tspan, z_0,rtol=1e-6,atol=1e-6,
                t_eval=np.linspace(0,Tfinal,int(Tfinal/dt)))

t = out.t
sol = out.y

# %%
# For comparison, need to calculate undisturbed neuron. Will further automate this, when have HCO dist. rej. working.
#neur_one_nosyn = Neuron(1., [120.,36.,0.3], [])
#network_nosyn = Network([neur_one_nosyn], np.zeros((1,1)))
#p_nosyn = (Iapps, network_nosyn)
#out_nosyn = solve_ivp(lambda t, z: no_observer(t, z, p_nosyn), tspan, x_0[:4],rtol=1e-6,atol=1e-6,
#                t_eval=np.linspace(0,Tfinal,int(Tfinal/dt)))
#
#t_nosyn = out_nosyn.t
#sol_nosyn = out_nosyn.y

# Test against HH_reject.py.
v = sol[0,:]
Isyn = syn.g * sol[4,:] * (v - neur_one.Esyn)
Isyn_hat = sol[13,:] * sol[9,:] * (v - neur_one.Esyn)
Isyn_dist = syn_dist.g * sol[5,:] * (v - neur_one.Esyn)
Isyn_dist_hat = sol[14,:] * sol[10,:] * (v - neur_one.Esyn)

v2 = sol[0+34,:]
Isyn2 = syn.g * sol[4+34,:] * (v - neur_one.Esyn)
Isyn_hat2 = sol[13+34,:] * sol[9+34,:] * (v - neur_one.Esyn)