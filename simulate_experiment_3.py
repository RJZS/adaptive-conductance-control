# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 17:08:56 2021

@author: Rafi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

from network_and_neuron import Synapse, Neuron, Network
from network_odes import main, no_observer

Tfinal = 8000.
control_start_time = 1000.

# TODO: can I change the initialisation without 'instability'?
# Same for increasing alpha and decreasing gamma.

# Initial conditions
x_0 = [0,0,0,0,0,0,0,0,0,0,0,0,0]; # V, m, h, mH, mT, hT, mA, hA, mKD, mL, mCa, s1, s2
x̂_0 = [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
θ̂_0 = np.ones(5); # 1 intrins, 2 syn, 2 el. Starts at idx 26.
P_0 = np.eye(5);
Ψ_0 = np.zeros(5);
to_estimate = np.array([0], dtype=np.int32)
to_observe = np.array([2], dtype=np.int32)
estimate_g_syns_g_els = True # Switch this.

syn1 = Synapse(0.6, 1) # 0.5 and 2 seem to have the same result.
syn2 = Synapse(0.6, 0)
syn3 = Synapse(0.6, 4)
syn4 = Synapse(0.6, 3)

# To hub neuron
synhub1 = Synapse(8, 0)
synhub2 = Synapse(8, 4)


Iapp = lambda t : 2 + np.sin(2*np.pi/10*t)
def Irb(t): # For rebound burster
    if t < 400 or t > 500:
        return -3.7
    else:
        return -7.5

def Ioffset(t): # So two neurons in HCO don't burst simultaneously
    if t < 200:
        return -4.
    else:
        return -2.
    
def Ioffset2(t): # So two neurons in HCO don't burst simultaneously
    if t < 300:
        return -4.
    else:
        return -2.
    
    
Iconst = lambda t: -2.


Iapps = [Iconst, Ioffset, lambda t: 38, Iconst, Ioffset2]

#x_0 = [0,0,0,0,0,0,0,0,0,0,0]; # V, m, h, mH, mT, hT, mA, hA, mKD, mL, Ca

one = Neuron(0.1, [120.,0.1,2.,0,80.,0.4,2.,0.,0.1], [syn1], 0)
two = Neuron(0.1, [120.,0.1,2.,0,80.,0.4,2.,0.,0.1], [syn2], 1)

three = Neuron(0.1, [80.,0.1,2.,0,30.,0.,1.,0.,0.1], [synhub1, synhub2], 2) # Hub neuron

four = Neuron(0.1, [120.,0.1,1.6,0,80.,0.2,2.,0.,0.1], [syn3], 1)
five = Neuron(0.1, [120.,0.1,1.6,0,80.,0.2,2.,0.,0.1], [syn4], 0)
# Remember, order of currents is Na, H, T, A, KD, L, KCA, KIR, leak

res_g = 0.005 # TODO: Need to raise this, otherwise hub isn't affecting HCO.
el_connects = np.array([[res_g, 1, 2],[res_g, 3, 2]])
network = Network([one, two, three, four, five], el_connects)

# Observer parameters
α = 0.05 # Default is 0.5. Had to decrease as P values were exploding.
γ = 5 # Default is 70, though Thiago's since lowered to 5. But 5 was causing psi to explode.

# For disturbance rejection, the format is ["DistRej", [(neur, syn), (neur, syn), ...], reject_els_to...]
# where (neur, syn) is a synapse to be rejected, identified by the index of the neuron in the network,
# and then the index of the synapse in the neuron.
reject_els_to_neur_idxs = [2] # Rejecting connections to the hub neuron
control_law = ["DistRej", [(2, 0), (2, 1)], reject_els_to_neur_idxs]
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
p = (Iapps,network,(α,γ),to_estimate,num_estimators,control_law,
     estimate_g_syns_g_els,control_start_time,to_observe)

print("Tfinal = {}s".format(Tfinal))
print("Starting simulation")
start_time = time.time()
out = solve_ivp(lambda t, z: main(t, z, p), tspan, z_0,rtol=1e-4,atol=1e-4,
                t_eval=np.linspace(0,Tfinal,int(Tfinal/dt)))#, method='LSODA')
end_time = time.time()
print(out.success)
print("Simulation time: {}s".format(end_time-start_time))
# NOTE: LATER, WHEN EVALUATING DENSE OUTPUT, CAN SET DT TO EQUAL PHASE SHIFT!

t = out.t
sol = out.y


# %%
# Comparison simulation. Just the HCO

one_nodist = Neuron(0.1, [120.,0.1,2.,0,80.,0.4,2.,0.,0.1], [syn1], 0)
two_nodist = Neuron(0.1, [120.,0.1,2.,0,80.,0.4,2.,0.,0.1], [syn2], 0)

network_nodist = Network([one_nodist, two_nodist], [])
p_nodist = (Iapps, network_nodist)

z_0_nodist = np.concatenate((x_0[:12], x_0[:12])) # One syn per neuron
# z_0_nodist[0] = 20
# z_0_nodist[12] = -20
# z_0_nodist = x_0[:12] # Only one neur
# z_0_nodist[0] = -70. # Mimicking line above.
start_time = time.time()
out_nodist = solve_ivp(lambda t, z: no_observer(t, z, p_nodist), tspan, z_0_nodist,rtol=1e-8,atol=1e-8,
                t_eval=np.linspace(0,Tfinal,int(Tfinal/dt)), method='LSODA')#, dense_output=True)
end_time = time.time()
print("'Nodist' Simulation time: {}s".format(end_time-start_time))

t_nodist = out_nodist.t
sol_nodist = out_nodist.y

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

# t=t.astype('float32')
# sol=sol.astype('float32')
# sol_ref = sol_ref.astype('float32')
# np.savez("simulate_experiment_1.npz", t=t,sol=sol,sol_ref=sol_ref)

# %% 
# Playing with model to find bursting behaviour.
playing = False
if playing:
    Tfinal = 3500//2.
    dt=0.01
    syn1 = Synapse(0.6, 1) # 0.5 and 2 seem to have the same result.
    syn2 = Synapse(0.6, 0)
    syn3 = Synapse(0.6, 4)
    syn4 = Synapse(0.6, 3)
    
    # To hub neuron
    synhub1 = Synapse(8, 0) # 0.5 does nothing to stop spiking. 6 give very wide bursts.
    synhub2 = Synapse(8, 4)
    
    
    Iapp = lambda t : 2 + np.sin(2*np.pi/10*t)
    def Irb(t): # For rebound burster
        if t < 400 or t > 500:
            return -3.7
        else:
            return -7.5
    
    def Ioffset(t): # So two neurons in HCO don't burst simultaneously
        if t < 200:
            return -4.
        else:
            return -2.
        
    def Ioffset2(t): # So two neurons in HCO don't burst simultaneously
        if t < 300:
            return -4.
        else:
            return -2.
        
    Iconst = lambda t: -2.
    
    
    Iapps = [Iconst, Ioffset, lambda t: 38, Iconst, Ioffset2] 
    # Iapps = [lambda t: 34] # 34 gives bursts, length about 1600. 38 gives spiking.
    
    tspan = (0.,Tfinal)
    
    # neur_one_play = Neuron(1., [120.,0,5.,0,36.,0,0,0.03], []) # For rebound burster.
    # neur_two_nodist = Neuron(1., [120.,0,0,0,36.,0,0,0.3], [syn2])
    
    #x_0 = [0,0,0,0,0,0,0,0,0,0,0]; # V, m, h, mH, mT, hT, mA, hA, mKD, mL, Ca
    
    # neur_one_play = Neuron(0.1, [100.,0.08,3.5,0,70.,0.5,1.6,0.,0.1], [])  # gNa, gH, gT, gA, gKD, gL, gKCa, gKir, gleak
    # neur_two_play = Neuron(0.1, [120.,0.1,2.,0,80.,0.4,2.,0.,0.1], [])
    
    one = Neuron(0.1, [120.,0.1,2.,0,80.,0.4,2.,0.,0.1], [syn1], 0)
    two = Neuron(0.1, [120.,0.1,2.,0,80.,0.4,2.,0.,0.1], [syn2], 1)
    
    three = Neuron(0.1, [80.,0.1,0.2,0,30.,0.,1.,0.,0.1], [synhub1, synhub2], 2) # Hub neuron
    # three = Neuron(0.1, [120.,0.1,2.,0,80.,0.4,2.,0.,0.1], [synhub1], 1) # Hub neuron
    
    four = Neuron(0.1, [120.,0.1,1.,0,80.,0.4,2.,0.,0.1], [syn3], 1)
    five = Neuron(0.1, [120.,0.1,1.,0,80.,0.4,2.,0.,0.1], [syn4], 0)
    # Remember, order of currents is Na, H, T, A, KD, L, KCA, KIR, leak
    
    res_g = 0.001
    el_connects = np.array([[res_g, 1, 2],[res_g, 3, 2]])
    # el_connects = np.array([[res_g, 1, 2]])
    
    # v_0 = np.array([-70.])
    # starting_gate_vals = neur_one_play.initialise(v_0)
    x_0 = [0,0,0,0,0,0,0,0,0,0,0,0,0] # 2 syns
    
    # # Rebound burster
    # neur_one_nodist = Neuron(1., [120.,0,0,0,36.,0,0,0.3], [])
    # network_play = Network([one, two, three, four, five], np.zeros((5,5)))
    network_play = Network([one, two, three, four, five], el_connects)
    # network_play = Network([one, two, three], el_connects)
    # network_play = Network([four, five], np.zeros((2,2)))
    p_play = (Iapps, network_play)
    
    z_0_play = np.concatenate((x_0, x_0, x_0, x_0, x_0))
    # z_0_play = x_0
    # z_0_play[0] = -70
    print("Starting 'play' simulation. Tfinal = {}".format(Tfinal))
    start_time = time.time()
    out_play = solve_ivp(lambda t, z: no_observer(t, z, p_play), tspan, z_0_play, rtol=2e-3,atol=2e-3,
                    t_eval=np.linspace(0,Tfinal,int(Tfinal/dt)), method='LSODA')#, vectorized=True)
    end_time = time.time()
    print("'Play' simulation time: {}s".format(end_time-start_time))
    
    t = out_play.t
    sol = out_play.y
    
    plt.plot(t,sol[0,:],t,sol[13,:],t,sol[26,:])
    # plt.plot(t_play,sol_play[0,:],t_play2,sol_play2[0,:])
    
print("Converting and saving...")
t=t.astype('float32')
sol=sol.astype('float32')
np.savez("obssim_experiment_3.npz", t=t, sol=sol,t_nd=t_nodist,sol_nd=sol_nodist)
