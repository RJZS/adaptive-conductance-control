# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 12:21:34 2021

@author: Rafi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

from network_and_neuron import Synapse, Neuron, Network
from network_odes import main, no_observer

# CURRENT STATUS:
# Observer works fine for single neuron, no synapses or controller, IF alpha is small enough.
# Added disturbance neuron. Works with no controller. But g_s_hat after 1200s is -4.4.
# But when I add DistRej controller, I get: "ValueError: array must not contain infs or NaNs",
# whether estimating g_s or not. V diverges. I_control makes it unstable.

# Ways to troubleshoot:
# Plot Isyn against Isyn_hat, see if converges.
# See if RefTrack works.

# TODO: Optimise!!    


## OPTIMISING.
# Main time sinks are... (set Tfinal = 10.)

# calc_intrins_dv_terms (especially ones and zeros. len isn't so bad). 2.43s.
#

# also gate_calcs, particularly all the gating functions which are called.
# 9.07s, of which 4.29s for compiling.
# After 'numbafying' the gating fns, it's less.

# NEXT
# gate_calcs itself. fn itself takes 3.15s, plus 1.2s for np.zeros and np.array.



# Initial conditions - HCO Disturbance Rejection
x_0 = [-70.,0,0,0,0,0,0,0,0,0,0,0]; # V, m, h, mH, mT, hT, mA, hA, mKD, mL, mCa, s1
x̂_0 = [0, 0.1, 0.2, 0.4, 0.1, 0.2, 0.4, 0.1, 0.2, 0.4, 0.5, 0.5]
θ̂_0 = [60, 60, 10, 10]; # Estimating gNa, gKD, gleak and 1 gsyn
P_0 = np.eye(4);
Ψ_0 = [0,0,0,0];
to_estimate = np.array([0, 4, 8])
estimate_g_syns = True
estimate_g_res = False # TODO: Need to write the code for this!!

syn = Synapse(2., 1)
syn2 = Synapse(15., 0)
syn_dist = Synapse(2., 2)
neur_one = Neuron(1., [120.,0,0,0,36.,0,0,0,0.3], [syn, syn_dist])
neur_two = Neuron(1., [120.,0,0,0,36.,0,0,0,0.3], [syn2])
neur_dist = Neuron(1., [120.,0,0,0,36.,0,0,0,0.3], [])
network = Network([neur_one, neur_two, neur_dist], np.zeros((3,3)))

neur_one_drion = Neuron(1., np.array([120.,0.02,0.2,0.,30.,0.,0.,0,0.055]), [syn])
neur_dist = Neuron(1., np.array([120.,0.02,0.2,0.,30.,0.,0.,0,0.055]), [])
network = Network([neur_one_drion, neur_dist], np.zeros((2,2)))

# Initial conditions - Single Neuron Disturbance Rejection
x_0 = [0.,0,0,0,0,0,0,0,0,0,0,0]; # V, m, h, mH, mT, hT, mA, hA, mKD, mL, Ca, s
x̂_0 = [30, 0.1, 0.2, 0.4, 0.1, 0.2, 0.4, 0.1, 0.2, 0.4, 0.5, 0.1]
θ̂_0 = [60, 60, 10, 1]; # Estimating gNa, gKD, gleak and 1 gsyn
P_0 = np.eye(4);
Ψ_0 = [0,0,0,0];
to_estimate = np.array([0, 4, 8])
estimate_g_syns = True
estimate_g_res = False # TODO: Need to write the code for this!!

syn = Synapse(2., 1)
neur_one = Neuron(0.1, [120.,0.1,2.,0,80.,0.4,2.,0.,0.1], [syn])  # gNa, gH, gT, gA, gKD, gL, gKCa, gKir, gleak
neur_dist = Neuron(0.1, [120.,0.1,2.,0,80.,0.4,2.,0.,0.1], [])
network = Network([neur_one, neur_dist], np.zeros((2,2)))

# control_law = ["DistRej", [(0, 0)]]#, (0, 1)]]

## Dist Rej Currents
Iapp = lambda t : 2 + np.sin(2*np.pi/10*t)
Iconst = lambda t: -2
Iapps = [Iconst, Iconst, lambda t: 6]

# Initial conditions - Reference Tracking
# x_0 = [0,0,0,0,0,0,0,0,0,0,0,0]; # V, m, h, mH, mT, hT, mA, hA, mKD, mL, mCa, s
# x̂_0 = [30, 0.1, 0.2, 0.4, 0.1, 0.2, 0.4, 0.1, 0.2, 0.4, 0.5, 0.5]
# θ̂_0 = [60, 60, 10, 10]; # Estimating gNa, gKD, gleak and gsyn
# P_0 = np.eye(4);
# Ψ_0 = [0,0,0,0];
# to_estimate = np.array([0, 4, 7])
# estimate_g_syns = True
# estimate_g_res = False # TODO: Need to write the code for this!!

# syn = Synapse(2., 1)
# syn2 = Synapse(2., 0)
# syn_dist = Synapse(2., 2)
# # Remember, order of currents is Na, H, T, A, KD, L, KCA, leak
# neur_one = Neuron(1., np.array([130.,0,0,0,43.,0,0,0.4]), np.array([syn]))
# neur_two = Neuron(1., np.array([100.,0,0,0,27.,0,0,0.2]), np.array([syn2]))
# network = Network([neur_one, neur_two], np.zeros((2,2))) # for ref tracking
# # ref_gs = np.array([[120,36,0.3,2],[120,72,0.3,2]]).T # gs of reference network.
# ref_gs = np.array([[110,0,0,0,35,0,0,0.2,2.5],
#                    [145,0,0,0,48,0,0,0.6,1.]]).T # gs of reference network.
# # orig_gs = np.array([ [130.,43.,0.4,2.], [100.,27.,0.2,2.] ]).T # gs of network, for the csv

# # Iapp = lambda t : 2 + np.sin(2*np.pi/10*t)
# Iapp = lambda t : 6 + np.sin(2*np.pi/10*t)
# Iapps = [Iapp, Iapp] # Neuron 2 converges even with constant current?
# Iapps = [2., 2.]

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
α = 0.05 # Default is 0.5. Had to decrease as P values were exploding.
γ = 70 # Default is 70, though Thiago's since lowered to 5. But 5 was causing psi to explode.

# For disturbance rejection, the format is ["DistRej", [(neur, syn), (neur, syn), ...]]
# where (neur, syn) is a synapse to be rejected, identified by the index of the neuron in the network,
# and then the index of the synapse in the neuron.
control_law = ["DistRej", [(0, 0)]]#, (0, 1)]]
# control_law = ["RefTrack", ref_gs]
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
Tfinal = 10.

tspan = (0.,Tfinal)
# controller_on = True
p = (Iapps,network,(α,γ),to_estimate,num_estimators,control_law,
     estimate_g_syns,estimate_g_res)

start_time = time.time()
out = solve_ivp(lambda t, z: main(t, z, p), tspan, z_0,rtol=1e-6,atol=1e-6,
                t_eval=np.linspace(0,Tfinal,int(Tfinal/dt)))#, method='BDF')
end_time = time.time()
print("Simulation time: {}s".format(end_time-start_time))

t = out.t
sol = out.y

# Let's calculate Isyn and Isyn_hat
v = sol[0,:]
Isyn = syn.g * sol[11,:] * (v - neur_one.Esyn)
Isyn_hat = sol[27,:] * sol[23,:] * (v - neur_one.Esyn)

# plt.plot(t,v)
