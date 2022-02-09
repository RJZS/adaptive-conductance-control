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

Tfinal0 = 4000.
Tfinal1 = 6000.
# Tfinal2 = 10000.
Tfinal2 = 4500.

#Tfinal0 = 605.
#Tfinal1 = 2000.
## Tfinal2 = 2500.
#Tfinal2 = 20.

print("Tfinal0 = {}".format(Tfinal0),file=open("exp3.txt","a"))
print("Tfinal1 = {}".format(Tfinal1),file=open("exp3.txt","a"))
print("Tfinal2 = {}".format(Tfinal2),file=open("exp3.txt","a"))
# TODO: can I change the initialisation without 'instability'?
# Same for increasing alpha and decreasing gamma.

tol = 1e-3
solvemethod = 'Radau'

# Initial conditions
x_0 = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]; # V, m, h, mH, mT, hT, mA, hA, mKD, mL, mCa, s1, s2
x̂_0 = [10, 0.1, 0.2, 0.01, 0.3, 0.1, 0.2, 0., 0.1, 0.2, 0.1, 0.1, 0.1]
θ̂_0 = np.ones(4); # 2 syn, 2 el. Starts at idx 26.
P_0 = np.eye(4);
Ψ_0 = np.zeros(4);
to_estimate = np.array([], dtype=np.int32)

# Have to change this if choose to reject the other g_el!
# Also have to change the nodist simulation below.
to_observe = np.array([2,3], dtype=np.int32)

estimate_g_syns_g_els = True # Switch this.

syn1 = Synapse(0.8, 1) # 0.5 and 2 seem to have the same result.
syn2 = Synapse(0.8, 0)
syn3 = Synapse(0.6, 4)
syn4 = Synapse(0.6, 3)

# To hub neuron
synhub1 = Synapse(8, 0)
synhub2 = Synapse(8, 4)


Iapp = lambda t : 2 + np.sin(2*np.pi/10*t)
def Irb(t): # For rebound burster
    if t < 400 or t > 600:
        return -3.2 # -3.7
    else:
        return -7
    
def Irb2(t): # For rebound burster
    if t < 400 or t > 600:
        return -2.5 # -3.7
    else:
        return -6

# def Ioffset(t): # So two neurons in HCO don't burst simultaneously
#     if t < 200:
#         return -4.
#     else:
#         return -2.
    
# def Ioffset2(t): # So two neurons in HCO don't burst simultaneously
#     if t < 300:
#         return -4.
#     else:
#         return -2.
    
Iconst = lambda t: -3.2 # -3.7
Iconst2 = lambda t: -2.5


# # Iapps = [Iconst, Ioffset, lambda t: 38, Iconst, Ioffset2] 
# Iapps_nodist = [Irb, Iconst, lambda t:38, Irb2, Iconst2]
# Iapps_before = [Iconst, Iconst, lambda t: 38, Iconst2, Iconst2]
# Iapps = [Iconst, Iconst, lambda t:38, Iconst2, Iconst2]

# Iapps for swapped case.
Iapps_nodist = [Irb2, Iconst2, lambda t:38, Irb, Iconst]
Iapps_before = [Iconst2, Iconst2, lambda t: 38, Iconst, Iconst]
Iapps = [Iconst2, Iconst2, lambda t:38, Iconst, Iconst]

#x_0 = [0,0,0,0,0,0,0,0,0,0,0]; # V, m, h, mH, mT, hT, mA, hA, mKD, mL, Ca

# hco_one_gs = np.array([120.,0.1,2.,0,80.,0.4,2.,0.,0.1])
# hub_gs = np.array([80.,0.1,2.,0,30.,0.,1.,0.,0.1])
# hco_two_gs = np.array([120.,0.1,1.6,0,80.,0.2,2.,0.,0.1])

# # Post-submission parameters
# hco_one_gs = np.array([130.,0.1,3.2,0,80.,1.,2.,0.,0.1])
# hub_gs = np.array([60.,0.1,2.,0,30.,0.,1.,0.,0.1])
# hco_two_gs = np.array([120.,0.1,1.6,0,80.,0.2,2.,0.,0.1])

# Post-submission parameters 2 (as above, except swapped HCOs).
hco_one_gs = np.array([120.,0.1,1.6,0,80.,0.2,2.,0.,0.1])
hub_gs = np.array([60.,0.1,2.,0,30.,0.,1.,0.,0.1])
hco_two_gs = np.array([130.,0.1,3.2,0,80.,1.,2.,0.,0.1])

one = Neuron(0.1, hco_one_gs, [syn1], 0)
two = Neuron(0.1, hco_one_gs, [syn2], 1)

three = Neuron(0.1, hub_gs, [synhub1, synhub2], 2) # Hub neuron

four = Neuron(0.1, hco_two_gs, [syn3], 1)
five = Neuron(0.1, hco_two_gs, [syn4], 0)
# Remember, order of currents is Na, H, T, A, KD, L, KCA, KIR, leak

res_g = 0.01 # TODO: Need to raise this, otherwise hub isn't affecting HCO.
el_connects = np.array([[res_g, 1, 2],[res_g, 3, 2]])
network = Network([one, two, three, four, five], el_connects)

# one_nodist = Neuron(0.1, hco_one_gs, [syn1], 0) # Same as one
# two_nodist = Neuron(0.1, hco_one_gs, [syn2], 1) # Same as two
three_nodist = Neuron(0.1, hub_gs, [synhub1], 2) # Hub neuron
# four_nodist = Neuron(0.1, hco_two_gs, [syn3], 1)
# five_nodist = Neuron(0.1, hco_two_gs, [syn4], 0)

# el_connects_nodist = np.array([[res_g, 1, 2]]) Not rejecting gap jn anymore.
network_nodist = Network([one, two, three_nodist, four,
                          five], el_connects)

# Observer parameters
α = 0.0001 # 0.0001 # 0.0008 # 0.00085 # HCO periods are 1080 and 850. What about combined rhythm?
γ = 5 # Default is 70, though Thiago's since lowered to 5. But 5 was causing psi to explode.

# For disturbance rejection, the format is ["DistRej", [(neur, syn), (neur, syn), ...], reject_els_to...]
# where (neur, syn) is a synapse to be rejected, identified by the index of the neuron in the network,
# and then the index of the synapse in the neuron.
# The next two elements of the control_law array pertain to electrical connections. They're very hacky!!
reject_els_to_neur_idxs = [2] # Rejecting connections to the hub neuron
reject_el_idx_of_reject_els_to_neur = 1 # Reject which of the connections involving 2.
control_law = ["DistRej", [(2, 1)], reject_els_to_neur_idxs,
               reject_el_idx_of_reject_els_to_neur]
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
# Start by simulating the system without the disturbance. So just the hub and first HCO.

p_nodist = (Iapps_nodist, network_nodist)
z_0_nodist = np.concatenate((x_0[:12], x_0[:12], x_0[:12],
                             x_0[:12], x_0[:12])) # No more than one syn per neuron.
# z_0_nodist[0] = -70. # Mimicking line above.

dt = 0.01
tspan = (0.,Tfinal0)
start_time = time.time()
out_nodist = solve_ivp(lambda t, z: no_observer(t, z, p_nodist), tspan, z_0_nodist,rtol=tol,atol=tol,
                t_eval=np.linspace(0,Tfinal0,int(Tfinal0/dt)), method=solvemethod, dense_output=False)
end_time = time.time()
print("'Nodist' Simulation time: {}s".format(end_time-start_time),file=open("exp3.txt","a"))

t_nodist = out_nodist.t
sol_nodist = out_nodist.y

# Now use those parameters to initialise the next sim.
z_0_before = np.concatenate((x_0, x_0, x_0, x_0, x_0))
z_0_before[:12] = sol_nodist[:12,-1]
z_0_before[13:13+12] = sol_nodist[12:24,-1]
z_0_before[26:26+12] = sol_nodist[24:36,-1]
z_0_before[39:39+12] = sol_nodist[36:48,-1]
z_0_before[52:52+12] = sol_nodist[48:,-1]

# %%
# Now run the full 5-neuron system without the observer/controller.
p_before = (Iapps_before, network)

tspan = (0.,Tfinal1)
start_time = time.time()
out_before = solve_ivp(lambda t, z: no_observer(t, z, p_before), tspan, z_0_before,rtol=tol,atol=tol,
                t_eval=np.linspace(0,Tfinal1,int(Tfinal1/dt)), method=solvemethod, dense_output=False)
end_time = time.time()
print("'Before' Simulation time: {}s".format(end_time-start_time),file=open("exp3.txt","a"))

t_before = out_before.t
sol_before = out_before.y

# Now use those parameters to initialise the next sim.
z_0[:13] = out_before.y[:13,-1]
# init = np.load("exp1_resg_0_4_solref.npy")
# z_0[:11] = init[:11]
# z_0[:11] = np.zeros(11)
neur_bef_start_idx = len(z_0) // 5
z_0[neur_bef_start_idx:neur_bef_start_idx+13] = out_before.y[13:2*13,-1] # init[11:]
z_0[2*neur_bef_start_idx:2*neur_bef_start_idx+13] = out_before.y[2*13:3*13,-1] # init[11:]
z_0[3*neur_bef_start_idx:3*neur_bef_start_idx+13] = out_before.y[3*13:4*13,-1] # init[11:]
z_0[4*neur_bef_start_idx:4*neur_bef_start_idx+13] = out_before.y[4*13:,-1] # init[11:]

# %%
# Integration initial conditions and parameters
tspan = (0.,Tfinal2)
# controller_on = True
varying_gT = (False,)
p = (Iapps,network,(α,γ),to_estimate,num_estimators,control_law,
     estimate_g_syns_g_els,0.,to_observe,varying_gT)

print("Starting simulation",file=open("exp3.txt","a"))
start_time = time.time()
out = solve_ivp(lambda t, z: main(t, z, p), tspan, z_0,rtol=tol,atol=tol,
                t_eval=np.linspace(0,Tfinal2,int(Tfinal2/dt)), method=solvemethod,
                dense_output=False)
end_time = time.time()
print(out.success,file=open("exp3.txt","a"))
print(out.y[126:130,-1],file=open("exp3.txt","a"))
print("Simulation time: {}s".format(end_time-start_time),file=open("exp3.txt","a"))
# Hint: could set dt to equal phase shift if decide to do phase shifts again,
# and obtain dense outputs etc.

t = out.t
sol = out.y

# %%
# To find peaks.
# from scipy.signal import find_peaks
# find_peaks(x) gives the idxs. Then can use np.roll for the phase-shift.


# %% 
# Playing with model to find bursting behaviour.
playing = False
if playing:
    Tfinal = 4500.
    Tfinal = 10000.
    Tfinal= 3000.
    dt=0.01
    syn1 = Synapse(0.8, 1) # 0.5 and 2 seem to have the same result.
    syn2 = Synapse(0.8, 0)
    syn3 = Synapse(0.6, 4)
    syn4 = Synapse(0.6, 3)
    
    
    # syn3 = Synapse(0.6, 1)
    # syn4 = Synapse(0.6, 0)
    
    # To hub neuron
    synhub1 = Synapse(8, 0) # 0.5 does nothing to stop spiking. 6 give very wide bursts.
    synhub2 = Synapse(8, 4)
    
    
    Iapp = lambda t : 2 + np.sin(2*np.pi/10*t)
    def Irb(t): # For rebound burster
        if t < 400 or t > 600:
            return -3.2 # -3.7
        else:
            return -7
        
    def Irb2(t): # For rebound burster
        if t < 400 or t > 600:
            return -2.5 # -3.7
        else:
            return -6
    
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
        
    Iconst2 = lambda t: -2.5
    
    Iconst = lambda t: -3.2 # -3.7
    
    
    # Iapps = [Iconst, Ioffset, lambda t: 38, Iconst, Ioffset2] 
    Iapps = [Irb, Iconst, lambda t: 38, Irb2, Iconst2] 
    # Iapps = [Irb2, Iconst2] # For second HCO.
    # Iapps = [lambda t: 34] # 34 gives bursts, length about 1600. 38 gives spiking.
    
    tspan = (0.,Tfinal)
    
    # neur_one_play = Neuron(1., [120.,0,5.,0,36.,0,0,0.03], []) # For rebound burster.
    # neur_two_nodist = Neuron(1., [120.,0,0,0,36.,0,0,0.3], [syn2])
    
    #x_0 = [0,0,0,0,0,0,0,0,0,0,0]; # V, m, h, mH, mT, hT, mA, hA, mKD, mL, Ca
    
    # neur_one_play = Neuron(0.1, [100.,0.08,3.5,0,70.,0.5,1.6,0.,0.1], [])  # gNa, gH, gT, gA, gKD, gL, gKCa, gKir, gleak
    # neur_two_play = Neuron(0.1, [120.,0.1,2.,0,80.,0.4,2.,0.,0.1], [])
    
    one = Neuron(0.1, [120.,0.1,2.,0,80.,0.4,2.,0.,0.1], [syn1], 0)
    two = Neuron(0.1, [120.,0.1,2.,0,80.,0.4,2.,0.,0.1], [syn2], 1)
    
    three = Neuron(0.1, [80.,0.1,2.,0,30.,0.,1.,0.,0.1], [synhub1, synhub2], 2) # Hub neuron
    
    four = Neuron(0.1, [120.,0.1,1.6,0,80.,0.2,2.,0.,0.1], [syn3], 0)# 1)
    five = Neuron(0.1, [120.,0.1,1.6,0,80.,0.2,2.,0.,0.1], [syn4], 0)
    # Remember, order of currents is Na, H, T, A, KD, L, KCA, KIR, leak
    
    res_g = 0.01 # 0.001
    el_connects = np.array([[res_g, 1, 2],[res_g, 3, 2]])
    # el_connects = np.array([[res_g, 1, 2]])
    
    # v_0 = np.array([-70.])
    # starting_gate_vals = neur_one_play.initialise(v_0)
    x_0 = [0,0,0,0,0,0,0,0,0,0,0,0,0] # 2 syns
    
    # # Rebound burster
    # neur_one_nodist = Neuron(1., [120.,0,0,0,36.,0,0,0.3], [])
    # network_play = Network([one, two, three, four, five], el_connects)
    network_play = Network([one, two], [])
    # network_play = Network([one, two, three], el_connects)
    # network_play = Network([four, five], np.zeros((2,2)))
    p_play = (Iapps, network_play)
    
    z_0_play = np.concatenate((x_0[:12], x_0[:12]))# , x_0[:12]))#, x_0, x_0))
    # z_0_play = np.concatenate((x_0, x_0, x_0, x_0, x_0))
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
    
    plt.plot(t,sol[0,:],t,sol[12,:])
    # plt.plot(t,sol[0,:],t,sol[13,:],t,sol[26,:]) # For full system.
    # plt.plot(t_play,sol_play[0,:],t_play2,sol_play2[0,:])
    
print("Converting and saving...",file=open("exp3.txt","a"))
t_before=t_before.astype('float32')
sol_before=sol_before.astype('float32')
t=t.astype('float32')
t_nodist = t_nodist.astype('float32')
sol=sol.astype('float32')
sol_nodist=sol_nodist.astype('float32')
np.savez("exp3.npz", tbef=t_before, t=t, tnd=t_nodist, solbef=sol_before, 
         sol=sol, solnd=sol_nodist)#,ps=ps,ps_idx=ps_idx)
