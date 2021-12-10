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

Tfinal = 6000. # I think 6000s is enough.
Tfinal = 8000.
control_start_time = 20.
print("Tfinal: {}".format(Tfinal))#, file=open("experiment1out.txt","a"))

# Single neuron reference tracking.
# TODO: can I change the initialisation without 'instability'?
# Same for increasing alpha and decreasing gamma.

# Initial conditions - Single Neuron Reference Tracking
x_0 = [0,0,0,0,0,0,0,0,0,0,0]; # V, m, h, mH, mT, hT, mA, hA, mKD, mL, mCa
x̂_0 = [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
θ̂_0 = np.zeros(9);
P_0 = np.eye(9);
Ψ_0 = np.zeros(9);
to_estimate = np.array([0,1,2,3,4,5,6,7,8], dtype=np.int32)
to_estimate = np.array([0,2,8], dtype=np.int32)
θ̂_0 = np.zeros(3);
P_0 = np.eye(3);
Ψ_0 = np.zeros(3);
estimate_g_syns_g_els = True

# Remember, order of currents is Na, H, T, A, KD, L, KCA, KIR, leak
neur_one = Neuron(0.1, np.array([0.,0.,0.,0,0.,0.,0.,0.,0.1]), np.array([]))
network = Network([neur_one], []) # for ref tracking
# ref_gs = np.array([[120,36,0.3,2],[120,72,0.3,2]]).T # gs of reference network.
ref_gs = np.array([[120.,0.1,2.,0,80.,0.4,2.,0.,0.1]]).T # gs of reference network.
# orig_gs = np.array([ [130.,43.,0.4,2.], [100.,27.,0.2,2.] ]).T # gs of network, for the csv
Iconst = lambda t: -2
Iapps = [Iconst, Iconst]

# Observer parameters
α = 0.5 # Default is 0.5. Had to decrease to 0.05 as P values were exploding.
γ = 5 # Default is 70, though Thiago's since lowered to 5. But 5 was causing psi to explode.

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
# z_0[0] = -70.

# %%
# Integration initial conditions and parameters
dt = 0.01

tspan = (0.,Tfinal)
len_ode_state = len(z_0)
p = (Iapps,network,(α,γ),to_estimate,num_estimators,control_law,
     estimate_g_syns_g_els,control_start_time,len_ode_state)

print("Starting simulation")#, file=open("experiment1out.txt","a"))
start_time = time.time()
out = solve_ivp(lambda t, z: main(t, z, p), tspan, z_0,rtol=1e-5,atol=1e-5,
                t_eval=np.linspace(0,Tfinal,int(Tfinal/dt)), method='LSODA',
                vectorized=True)
end_time = time.time()
print("Simulation time: {}s".format(end_time-start_time))#,file=open("experiment1out.txt","a"))

t = out.t
sol = out.y


# %%
# Comparison simulation

# # Reference Tracking
# syn_ref = Synapse(2.5, 1)
# syn2_ref = Synapse(1., 0)

# the original, for comparison: [120.,0.1,2.,0,80.,0.4,2.,0.,0.1]
neur_one_ref = Neuron(0.1, np.array([120.,0.1,2.,0,80.,0.4,2.,0.,0.1]), np.array([]))
network_ref = Network([neur_one_ref], np.zeros((1,1)))
p_ref = (Iapps, network_ref)
# z_0_ref = np.concatenate((x_0, x_0))
z_0_ref = x_0
# z_0_ref[0] = -70.

start_time = time.time()
out_ref = solve_ivp(lambda t, z: no_observer(t, z, p_ref), tspan, z_0_ref,rtol=1e-6,atol=1e-6,
                t_eval=np.linspace(0,Tfinal,int(Tfinal/dt)))
end_time = time.time()
print("'Ref' Simulation time: {}s".format(end_time-start_time))#,file=open("experiment1out.txt","a"))

t_ref = out_ref.t
sol_ref = out_ref.y

# PHASE SHIFTS:
# For single neur RT, phase shift was 11364.

# %% 
# # Playing with model to find bursting behaviour.
# dt=0.01
# syn = Synapse(0.5, 1) # 0.5 and 2 seem to have the same result.
# syn2 = Synapse(2., 0)
# Iapp = lambda t : 2 + np.sin(2*np.pi/10*t)
# def Irb(t): # For rebound burster
#     if t < 10 or t > 40:
#         return 0
#     else:
#         return -6
    
# Iconst = lambda t: -2. #-0.1 in HCO2

# def Iramp(t):
#     if t < 500:
#         return -1.
#     else:
#         return -2 + 0.005*(t-500)

# Iapps = [Iconst, Iconst, lambda t: 6]
# Tfinal = 400. # In HCO2 it's 15000. In notebook it's 2800.

# tspan = (0.,Tfinal)

# # neur_one_play = Neuron(1., [120.,0,5.,0,36.,0,0,0.03], []) # For rebound burster.
# # neur_two_nodist = Neuron(1., [120.,0,0,0,36.,0,0,0.3], [syn2])

# #x_0 = [0,0,0,0,0,0,0,0,0,0,0]; # V, m, h, mH, mT, hT, mA, hA, mKD, mL, Ca

# neur_one_play = Neuron(0.1, [100.,0.08,3.5,0,70.,0.5,1.6,0.,0.1], [])  # gNa, gH, gT, gA, gKD, gL, gKCa, gKir, gleak
# neur_two_play = Neuron(0.1, [120.,0.1,2.,0,80.,0.4,2.,0.,0.1], [])

# # v_0 = np.array([-70.])
# # starting_gate_vals = neur_one_play.initialise(v_0)
# x_0 = [0,0,0,0,0,0,0,0,0,0,0] # No syns

# # # Rebound burster
# # neur_one_nodist = Neuron(1., [120.,0,0,0,36.,0,0,0.3], [])
# network_play = Network([neur_one_play], np.zeros((1,1)))
# p_play = (Iapps, network_play)

# # z_0_play = np.concatenate((x_0, x_0))
# z_0_play = x_0
# z_0_play[0] = -70
# start_time = time.time()
# out_play = solve_ivp(lambda t, z: no_observer(t, z, p_play), tspan, z_0_play, rtol=1e-6,atol=1e-6,
#                 t_eval=np.linspace(0,Tfinal,int(Tfinal/dt)))#, vectorized=True)
# end_time = time.time()
# print("'Play' simulation time: {}s".format(end_time-start_time))

# t_play2 = out_play.t
# sol_play2 = out_play.y

# # plt.plot(t_play,sol_play[0,:],t_play2,sol_play2[0,:])

# %%
# # Extract variables and label them
# V_idxs = np.array(list(range(num_neurs)))*(len(sol)/num_neurs)
# V_idxs = V_idxs.astype(int)
# Vs = sol[V_idxs,:]

# %%
# To find peaks.
# from scipy.signal import find_peaks
# find_peaks(x) gives the idxs. Then can use np.roll for the phase-shift.

# For HCO_RT it's about 1105, ie np.roll(x, 1105). Remember the spike is every other local max.

# rr = np.roll(sol_ref[0,:],5207)
# j=584600;k=585000;plt.plot(t[j:k],sol[0,j:k]-rr[j:k])

start_time=time.time()
t=t.astype('float32')
sol=sol.astype('float32')
sol_ref = sol_ref.astype('float32')
end_time = time.time()
# print("Conversion time: {}s".format(end_time-start_time),file=open("experiment1out.txt","a"))

start_time = time.time()
np.savez("simulate_experiment_1.npz", t=t,sol=sol,sol_ref=sol_ref)
end_time = time.time()
# print("Saving time: {}s".format(end_time-start_time),file=open("experiment1out.txt","a"))
