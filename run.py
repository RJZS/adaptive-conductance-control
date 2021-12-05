# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 19:13:57 2021

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
# whether estimating g_s or not. V diverges.

# TODO: Optimise!!    

# On the backburner:
# Code for graph plotting. Don't just plot everything (messy!), maybe have a 
# parameter which is a list of which to plot. Ie an object, where the keys are fig discriptions 
# and the values are booleans. Then can have each plot within its own 'if' statement.

# Automate running of 'nosyn' simulation (should be called 'nodist'). Harder than I thought!
# At least can automate addition of synaptic terms to initialisation, as know number of syns!
# (So have to move network definition above initialisation).


# Initial conditions - HCO Disturbance Rejection
x_0 = [-70.,0,0,0,0,0,0,0,0,0,0,0]; # V, m, h, mH, mT, hT, mA, hA, mKD, mL, mCa, s1
x̂_0 = [30, 0.1, 0.2, 0.4, 0.1, 0.2, 0.4, 0.1, 0.2, 0.4, 0.5, 0.5]
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
x̂_0 = [30, 0.1, 0.2, 0.4, 0.1, 0.2, 0.4, 0.1, 0.2, 0.4, 0.5, 0.45]
θ̂_0 = [60, 60, 10, 10]; # Estimating gNa, gKD, gleak and 1 gsyn
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
Tfinal = 0.23 # Should converge by 1800.

tspan = (0.,Tfinal)
# controller_on = True
p = (Iapps,network,(α,γ),to_estimate,num_estimators,control_law,
     estimate_g_syns,estimate_g_res)

start_time = time.time()
out = solve_ivp(lambda t, z: main(t, z, p), tspan, z_0,rtol=1e-6,atol=1e-6,
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
        
# HCO disturbance rejection
neur_one_nodist = Neuron(1., [120.,0,0,0,36.,0,0,0,0.3], [syn])
neur_two_nodist = Neuron(1., [120.,0,0,0,36.,0,0,0,0.3], [syn2])

# # Only one neur
# neur_one_nodist = Neuron(1., [120.,0,0,0,36.,0,0,0.3], [])
network_nodist = Network([neur_one_nodist, neur_two_nodist], np.zeros((2,2)))
p_nodist = (Iapps, network_nodist)
z_0_nodist = np.concatenate((x_0[:12], x_0[:12]))
z_0_nodist[0] = 20
z_0_nodist[12] = -20
# z_0_nodist = x_0[:11] # Only one neur
out_nodist = solve_ivp(lambda t, z: no_observer(t, z, p_nodist), tspan, z_0_nodist,rtol=1e-6,atol=1e-6,
                t_eval=np.linspace(0,Tfinal,int(Tfinal/dt)))

t_nodist = out_nodist.t
sol_nodist = out_nodist.y

# Reference Tracking
# syn_ref = Synapse(2.5, 1)
# syn2_ref = Synapse(1., 0)

# neur_one_ref = Neuron(1., np.array([110.,0,0,0,35.,0,0,0.2]), np.array([syn_ref]))
# neur_two_ref = Neuron(1., np.array([145.,0,0,0,48.,0,0,0.6]), np.array([syn2_ref]))
# network_ref = Network([neur_one, neur_two], np.zeros((2,2)))
# p_ref = (Iapps, network_ref)
# z_0_ref = np.concatenate((x_0, x_0))
# out_ref = solve_ivp(lambda t, z: no_observer(t, z, p_ref), tspan, z_0_ref,rtol=1e-6,atol=1e-6,
#                 t_eval=np.linspace(0,Tfinal,int(Tfinal/dt)))

# t_ref = out_ref.t
# sol_ref = out_ref.y

# %% 
# Playing with model to find bursting behaviour.
dt=0.01
syn = Synapse(0.5, 1) # 0.5 and 2 seem to have the same result.
syn2 = Synapse(2., 0)
Iapp = lambda t : 2 + np.sin(2*np.pi/10*t)
def Irb(t): # For rebound burster
    if t < 10 or t > 40:
        return 0
    else:
        return -6
    
Iconst = lambda t: -2. #-0.1 in HCO2

def Iramp(t):
    if t < 500:
        return -1.
    else:
        return -2 + 0.005*(t-500)

Iapps = [Iconst, Iconst, lambda t: 6]
Tfinal = 350 # In HCO2 it's 15000. In notebook it's 2800.

tspan = (0.,Tfinal)

# neur_one_play = Neuron(1., [120.,0,5.,0,36.,0,0,0.03], []) # For rebound burster.
# neur_two_nodist = Neuron(1., [120.,0,0,0,36.,0,0,0.3], [syn2])

#x_0 = [0,0,0,0,0,0,0,0,0,0,0]; # V, m, h, mH, mT, hT, mA, hA, mKD, mL, Ca

neur_one_play = Neuron(0.1, [120.,0.1,2.,0,80.,0.4,2.,0.,0.1], [syn])  # gNa, gH, gT, gA, gKD, gL, gKCa, gKir, gleak
neur_two_play = Neuron(0.1, [120.,0.1,2.,0,80.,0.4,2.,0.,0.1], [])

# v_0 = np.array([-70.])
# starting_gate_vals = neur_one_play.initialise(v_0)
x_0 = [0,0,0,0,0,0,0,0,0,0,0,0] # 1 syn

# # Rebound burster
# neur_one_nodist = Neuron(1., [120.,0,0,0,36.,0,0,0.3], [])
network_play = Network([neur_one_play, neur_two_play], np.zeros((2,2)))
p_play = (Iapps, network_play)

z_0_play = np.concatenate((x_0, x_0))
# z_0_play = x_0
z_0_play[0] = -70
start_time = time.time()
out_play = solve_ivp(lambda t, z: no_observer(t, z, p_play), tspan, z_0_play, rtol=1e-6,atol=1e-6,
                t_eval=np.linspace(0,Tfinal,int(Tfinal/dt)))#, vectorized=True)
end_time = time.time()
print("'Play' simulation time: {}s".format(end_time-start_time))

t_play = out_play.t
sol_play = out_play.y

plt.plot(t_play,sol_play[0,:])
# %%
# Test HCO disturbance rejection. First compare real and estimated Isyns.
v = sol[0,:]
Isyn = syn.g * sol[11,:] * (v - neur_one.Esyn)
Isyn_hat = sol[29,:] * sol[24,:] * (v - neur_one.Esyn)
Isyn_dist = syn_dist.g * sol[12,:] * (v - neur_one.Esyn)
Isyn_dist_hat = sol[30,:] * sol[25,:] * (v - neur_one.Esyn)

v2 = sol[0+61,:]
Isyn2 = syn2.g * sol[11+61,:] * (v2 - neur_two.Esyn)
Isyn_hat2 = sol[29+61,:] * sol[24+61,:] * (v2 - neur_two.Esyn)

# Now compare Vs with V_nosyns (misleading name, as there are synapses,
# just not the disturbance one).
v_nodist = sol_nodist[0,:]
v2_nodist = sol_nodist[12,:]

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
