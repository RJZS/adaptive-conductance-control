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

Tfinal = 10. # Textbook notebook has 1800. Simulate to 4000. Can start slowing at 620.
control_start_time = 100.
print("Tfinal = {}".format(Tfinal))

# Single neuron reference tracking.
# TODO: can I change the initialisation without 'instability'?
# Same for increasing alpha and decreasing gamma.

# Initial conditions - Single Neuron Reference Tracking
x_0 = [0,0,0,0,0,0,0,0,0,0,0]; # V, m, h, mH, mT, hT, mA, hA, mKD, mL, mCa
x̂_0 = [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
to_observe = np.array([0], dtype=np.int32) # Neurons to observe.
θ̂_0 = np.ones(9);
P_0 = np.eye(9); # From idx 31 I think
Ψ_0 = np.zeros(9);
to_estimate = np.array([0,1,2,3,4,5,6,7,8], dtype=np.int32)
estimate_g_syns_g_els = True

# Remember, order of currents is Na, H, T, A, KD, L, KCA, KIR, leak
neur_one = Neuron(0.1, np.array([0.,0.,0.,0,0.,0.,0.,0.,0.1]), np.array([]), 0)
network = Network([neur_one], []) # for ref tracking
# ref_gs = np.array([[120,36,0.3,2],[120,72,0.3,2]]).T # gs of reference network.
ref_gs = np.array([[120.,0.1,2.,0,80.,0.4,2.,0.,0.1]]).T # gs of reference network.
# orig_gs = np.array([ [130.,43.,0.4,2.], [100.,27.,0.2,2.] ]).T # gs of network, for the csv
Iconst = lambda t: -2
Iconstsin = lambda t: -2 + np.sin(2*np.pi/10*t)
Iapps = [Iconst, Iconst]

# Observer parameters
α = 0.0001 # Default is 0.5. Had to decrease as P values were exploding.
γ = 20 # Default is 70, though Thiago's since lowered to 5. But 5 was causing psi to explode.

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
# dt = 0.001

tspan = (0.,Tfinal)
p = (Iapps,network,(α,γ),to_estimate,num_estimators,control_law,
     estimate_g_syns_g_els,control_start_time,to_observe)

print("Starting simulation")
start_time = time.time()
out = solve_ivp(lambda t, z: main(t, z, p), tspan, z_0,rtol=1e-4,atol=1e-4,
                t_eval=np.linspace(0,Tfinal,int(Tfinal/dt)), method='LSODA',
                dense_output=True)
end_time = time.time()
print("Simulation time: {}s".format(end_time-start_time))

# t = out.t
# sol = out.y

# t = np.linspace(0,Tfinal,int(Tfinal//dt + 1))
# sol = out.sol(t)

# dt = 0.0005
# t = np.linspace(0,Tfinal,int(Tfinal//dt + 1))
# t1 = t[0::2]
# sol1 = out.sol(t1)
# sol1 = sol1.astype('float32')
# t1 = t1.astype('float32')
# t2 = t[1::2]
# sol2 = out.sol(t2)
# sol2 = sol2.astype('float32')
# t2 = t2.astype('float32')

# t = np.concatenate((t1, t2))
# del t1, t2
# sol = np.concatenate((sol1, sol2), axis=1)
# del sol1, sol2

# sort_args = np.argsort(t)
# t = t[sort_args]
# sol = sol[:, sort_args]

# %%
# Comparison simulation

# # Reference Tracking
# syn_ref = Synapse(2.5, 1)
# syn2_ref = Synapse(1., 0)

# the original, for comparison: [120.,0.1,2.,0,80.,0.4,2.,0.,0.1]
neur_one_ref = Neuron(0.1, np.array([120.,0.1,2.,0,80.,0.4,2.,0.,0.1]), np.array([]), 0)
network_ref = Network([neur_one_ref], [])
p_ref = (Iapps, network_ref)
# z_0_ref = np.concatenate((x_0, x_0))
z_0_ref = x_0
# z_0_ref[0] = -70.

start_time = time.time()
out_ref = solve_ivp(lambda t, z: no_observer(t, z, p_ref), tspan, z_0_ref,rtol=1e-4,atol=1e-4,
                t_eval=np.linspace(0,Tfinal,int(Tfinal/dt)), method='LSODA', dense_output=True)
end_time = time.time()
print("'Ref' Simulation time: {}s".format(end_time-start_time))

# t_ref = out_ref.t
# sol_ref = out_ref.y

# t_ref = np.linspace(0,Tfinal,int(Tfinal//dt + 1))
# sol_ref = out_ref.sol(t)

# dt = 0.0005
# t_ref = np.linspace(0,Tfinal,int(Tfinal//dt + 1))
# t1_ref = t_ref[0::2]
# sol1_ref = out_ref.sol(t1_ref)
# sol1_ref = sol1_ref.astype('float32')
# t1_ref = t1_ref.astype('float32')
# t2_ref = t_ref[1::2]
# sol2_ref = out_ref.sol(t2_ref)
# sol2_ref = sol2_ref.astype('float32')
# t2_ref = t2_ref.astype('float32')

# t_ref = np.concatenate((t1_ref, t2_ref))
# del t1_ref, t2_ref
# sol_ref = np.concatenate((sol1_ref, sol2_ref), axis=1)
# del sol1_ref, sol2_ref

# sort_args_ref = np.argsort(t_ref)
# t_ref = t_ref[sort_args_ref]
# sol_ref = sol_ref[:, sort_args_ref]

# PHASE SHIFT: 52044

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
# Evaluate the solutions, accounting for phase shift.
know_ps = False
if know_ps:
    dt = 0.01
    t = np.linspace(0,Tfinal,int(Tfinal//dt + 1))
    sol = out.sol(t)
    
    ps = 52.04336381966011 - 1.1700015 # Phase shift
    ps_idx = int(ps//dt + 1)
    t_ref = t - ps
    t_ref = t_ref[ps_idx:] # Eliminate the negative values at the start.
    sol_ref = out_ref.sol(t_ref)
    
    # plt.plot(t[ps_idx:],sol[0,ps_idx:],t[ps_idx:],sol_ref[0,:])
    # j=745000;k=760000;plt.plot(t[ps_idx+j:ps_idx+k],sol[0,ps_idx+j:ps_idx+k],t[ps_idx+j:ps_idx+k],sol_ref[0,j:k])
    diff = sol[0,ps_idx:] - sol_ref[0,:]
    #check_from_idx=300000
    #diff_trunc = diff[check_from_idx:]
    #to_minimise = np.abs(diff_trunc).max()

calc_ps = True
if calc_ps:
    # The objective function to calculate phase shift, ps. Assumes ps > 0.
    def obj_fn(ps, start, end, dt):
        t = np.linspace(start,end,int((end-start)//dt + 1))
        sol = out.sol(t)
        ps_idx = int(ps//dt + 1)
        t_ref = t - ps
        t_ref = t_ref[ps_idx:] # Eliminate the negative t values at the start.
        sol_ref = out_ref.sol(t_ref)
    
        diff = sol[0,ps_idx:] - sol_ref[0,:]
        to_minimise = np.abs(diff).max()
        return to_minimise
    
    start_time = time.time()
    obj_fn(ps, 7500, 7650, 0.01)
    end_time = time.time()
    print("Time to run obj fn: {}s".format(end_time-start_time))
    
    from scipy.optimize import minimize_scalar
    ps_start = 52.04336
    ps_end = 52.04337
    res = minimize_scalar(obj_fn, bounds=(ps_start, ps_end), method='bounded',
                          options={'maxiter':10,'disp':True}, args=(3200,3400,0.001))
    print(res.x) # Precise phase shift.

# plt.plot(t[ps_idx+check_from_idx:],diff[check_from_idx:])

# rr = np.roll(sol_ref[0,:],52043)
# plt.plot(t,sol[0,:],t,rr)
# j=300000;plt.plot(t[j:],sol[0,j:],t[j:],rr[j:])
# # j=320000;k=340000;plt.plot(t[j:k],sol[0,j:k]-rr[j:k])
# j=3363600;k=3364000;plt.plot(t[j:k],sol[0,j:k],t[j:k],rr[j:k],t[j:k],10*(sol[0,j:k]-rr[j:k]))

# SO the phase shift is 52.04007 + 0.00075 (+ another 0.00258?)

# rr = np.roll(fine_sol_ref[0,:],15)
# j=3500;k=6500;plt.plot(fine_t[j:k],fine_sol[0,j:k],fine_t[j:k],rr[j:k])

# plt.plot(fine_t[j:k],50*(fine_sol[0,j:k]-rr[j:k]))
# pk_idx = find_peaks(fine_sol[0,j:k]-rr[j:k])[0][0]
# print((fine_sol[0,j:k]-rr[j:k])[pk_idx])

# %%
# To find peaks.
# from scipy.signal import find_peaks
# find_peaks(x) gives the idxs. Then can use np.roll for the phase-shift.

# For HCO_RT it's about 1105, ie np.roll(x, 1105). Remember the spike is every other local max.

# t=t.astype('float32')
# t_ref = t_ref.astype('float32')
# sol=sol.astype('float32')
# sol_ref = sol_ref.astype('float32')
# np.savez("simulate_experiment_1.npz", t=t,t_ref=t_ref,sol=sol,sol_ref=sol_ref,ps=ps,ps_idx=ps_idx)
