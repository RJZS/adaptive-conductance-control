# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 15:17:14 2021

@author: Rafi
"""
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

Tfinal = 8000.
# Tfinal = 1000.
control_start_time = 0. # 2000.

# Initial conditions - Single Neuron Disturbance Rejection
x_0 = [0,0,0,0,0,0,0,0,0,0,0,0]; # V, m, h, mH, mT, hT, mA, hA, mKD, mL, Ca, s
x̂_0 = [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
θ̂_0 = np.ones(2); # Estimating all intrinsic and 1 syn.
P_0 = np.eye(2);
Ψ_0 = np.zeros(2);
to_estimate = np.array([0],dtype=np.int32)
to_observe = np.array([0], dtype=np.int32)
estimate_g_syns_g_els = True

syn = Synapse(2., 1)
neur_one = Neuron(0.1, [120.,0.1,2.,0,80.,0.4,2.,0.,0.1], [syn], 0)  # gNa, gH, gT, gA, gKD, gL, gKCa, gKir, gleak
neur_dist = Neuron(0.1, [120.,0.1,2.,0,80.,0.4,2.,0.,0.1], [], 0)
network = Network([neur_one, neur_dist], [])

control_law = ["DistRej", [(0, 0)]]#, (0, 1)]]

## Dist Rej Currents
# Iapp = lambda t : 2 + np.sin(2*np.pi/10*t)
Iconst = lambda t: -2
Iapps = [Iconst, Iconst]

# Observer parameters
α = 0.0005 # Default is 0.5. Had to decrease as P values were exploding.
γ = 5 # Default is 70, though Thiago's since lowered to 5. But 5 was causing psi to explode.

# For disturbance rejection, the format is ["DistRej", [(neur, syn), (neur, syn), ...]]
# where (neur, syn) is a synapse to be rejected, identified by the index of the neuron in the network,
# and then the index of the synapse in the neuron.
# control_law = ["DistRej", [(0, 0)]]#, (0, 1)]]
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
# z_0[0] = -70.

# %%
# Integration initial conditions and parameters
dt = 0.01

tspan = (0.,Tfinal)
# controller_on = True
p = (Iapps,network,(α,γ),to_estimate,num_estimators,control_law,
     estimate_g_syns_g_els,control_start_time,to_observe)

print("Starting simulation",file=open("exp2.txt","a"))
start_time = time.time()
out = solve_ivp(lambda t, z: main(t, z, p), tspan, z_0,rtol=1e-3,atol=1e-3,
                t_eval=np.linspace(0,Tfinal,int(Tfinal/dt)), method='Radau',
                dense_output=True)
end_time = time.time()
print("Simulation time: {}s".format(end_time-start_time),file=open("exp2.txt","a"))

# t = out.t
# sol = out.y

# Let's calculate Isyn and Isyn_hat
# v = sol[0,:]
# Isyn = syn.g * sol[11,:] * (v - neur_one.Esyn)
# Isyn_hat = sol[27,:] * sol[23,:] * (v - neur_one.Esyn)

# plt.plot(t,v)

# %%
# For comparison, need to calculate undisturbed neuron.
# Want to further automate this, but actually fairly tricky...
# def prepare_nodist_sim(neurons, control_law, z_0):
#     for (neur_i, syn_i) in control_law[1]:
        
# Single neuron disturbance rejection
neur_one_nodist = Neuron(0.1, [120.,0.1,2.,0,80.,0.4,2.,0.,0.1], [], 0)

network_nodist = Network([neur_one_nodist], [])
p_nodist = (Iapps, network_nodist)
# z_0_nodist = np.concatenate((x_0[:11], x_0[:11]))
# z_0_nodist[0] = 20
# z_0_nodist[12] = -20
z_0_nodist = x_0[:11] # Only one neur
# z_0_nodist[0] = -70. # Mimicking line above.
start_time = time.time()
out_nodist = solve_ivp(lambda t, z: no_observer(t, z, p_nodist), tspan, z_0_nodist,rtol=1e-3,atol=1e-3,
                t_eval=np.linspace(0,Tfinal,int(Tfinal/dt)), method='Radau', dense_output=True)
end_time = time.time()
print("'Nodist' Simulation time: {}s".format(end_time-start_time),file=open("exp2.txt","a"))

# t_nodist = out_nodist.t
# sol_nodist = out_nodist.y

# %%
# Evaluate the solutions, accounting for phase shift.

know_ps = True
if know_ps:
    ps = 486.86592902676364 # Phase shift
    dt = 0.01
    t = np.linspace(0,Tfinal,int(Tfinal//dt + 1))
    sol = out.sol(t)
    ps_idx = int(ps//dt + 1)
    t_nodist = t - ps
    t_nodist = t_nodist[ps_idx:] # Eliminate the excess values at the start.
    sol_nodist = out_nodist.sol(t_nodist)
    
    # plt.plot(t[-ps_idx:],sol[0,:ps_idx],t_nodist,sol_nodist[0,:])
    diff = sol[0,ps_idx:] - sol_nodist[0,:]
    # plt.plot(t[j-ps_idx:],diff[j:])
    # j=746500;k=749000;plt.plot(t[j-ps_idx:k-ps_idx],sol[j:k],t_nodist[j:k],sol_nodist[j:k])
    # j=400000;plt.plot(t_nodist[j:],sol[j:ps_idx]-np.roll(sol_nodist,-5)[j:])

calc_ps = False
if calc_ps:
    # # To find phase shift.
    def obj_fn(ps, start, end, dt):
        t = np.linspace(start,end,int((end-start)//dt + 1))
        sol = out.sol(t)
        ps_idx = int(ps//dt + 1)
        t_nodist = t - ps
        t_nodist = t_nodist[ps_idx:] # Eliminate the negative values at the start.
        sol_nodist = out_nodist.sol(t_nodist)
        
        diff = sol[0,ps_idx:] - sol_nodist[0,:]
        to_minimise = np.abs(diff).max()
        return to_minimise
    
    # plt.plot(t[ps_idx:],sol[0,ps_idx:],t[ps_idx:],sol_ref[0,:])
    start_time = time.time()
    obj_fn(-81.88, 5600, 5800, 0.001)
    end_time = time.time()
    print("Time to run obj fn: {}s".format(end_time-start_time),file=open("exp2.txt","a"))
    
    from scipy.optimize import minimize_scalar
    ps_start = 350
    ps_end = 500
    res = minimize_scalar(obj_fn, bounds=(ps_start, ps_end), method='bounded',
                          options={'maxiter':50,'disp':True}, args=(6100,7300,0.001))
    res.x # Precise phase shift.
    print(res, file=open("exp2.txt","a"))
    print(res.x, file=open("exp2.txt","a"))

# %%

# print("sol.max: {}".format(sol.max()),file=open("exp2.txt","a"))
# print("sol.min: {}".format(sol.min()),file=open("exp2.txt","a"))
print("Converting and saving...",file=open("exp2.txt","a"))
t=t.astype('float32')
t_nodist = t_nodist.astype('float32')
sol=sol.astype('float32')
sol_nodist=sol_nodist.astype('float32')
np.savez("exp2.npz", t=t, tnd=t_nodist, sol=sol,
         solnd=sol_nodist,ps=ps,ps_idx=ps_idx)

