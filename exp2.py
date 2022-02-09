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

Tfinal0 = 4000.
Tfinal1 = 4000.
Tfinal2 = 6000. # 2000.

Tfinal1 = 6000.
Tfinal2 = 20000.

# Post-submission settings
Tfinal0 = 8000.
Tfinal1 = 8000.
Tfinal2 = 20000. # 2000.


print("Tfinal0 = {}".format(Tfinal0),file=open("exp2.txt","a"))
print("Tfinal1 = {}".format(Tfinal1),file=open("exp2.txt","a"))
print("Tfinal2 = {}".format(Tfinal2),file=open("exp2.txt","a"))

tol = 1e-3
solvemethod = 'Radau'

# Tfinal = 11
# observe_start_time = 10. # 2000.

# Initial conditions - Single Neuron Disturbance Rejection
x_0 = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]; # V, m, h, mH, mT, hT, mA, hA, mKD, mL, Ca, s
x̂_0 = [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
θ̂_0 = np.ones(1); # Estimating 1syn. idx 24
P_0 = np.eye(1);
Ψ_0 = np.zeros(1);
to_estimate = np.array([],dtype=np.int32)
to_observe = np.array([0], dtype=np.int32)
estimate_g_syns_g_els = True

# syn = Synapse(0.8, 1)

# neur_one_gs = np.array([120.,0.1,2.,0,80.,0.4,2.,0.,0.1])
# neur_dist_gs = np.array([120.,0.1,3.,0,80.,1.,2.,0.,0.1])

# Post-submission settings
syn = Synapse(2.5, 1)
neur_one_gs = np.array([60.,0.1,2.,0,80.,0.4,2.,0.,0.12])
neur_dist_gs = np.array([130.,0.1,3.2,0,80.,1.,2.,0.,0.1])

neur_one_nodist = Neuron(0.1, neur_one_gs, [], 0)
network_nodist = Network([neur_one_nodist], [])

neur_one = Neuron(0.1, neur_one_gs, [syn], 0)  # gNa, gH, gT, gA, gKD, gL, gKCa, gKir, gleak
neur_dist = Neuron(0.1, neur_dist_gs, [], 0)
network = Network([neur_one, neur_dist], [])

control_law = ["DistRej", [(0, 0)]]#, (0, 1)]]
# control_law = [""]

## Dist Rej Currents
# Iapp = lambda t : 2 + np.sin(2*np.pi/10*t)
def Ioffset_dist(t): # So two neurons in HCO don't burst simultaneously
    if t < 400:
        return -7.5
    else:
        return -1.
    
Iconst = lambda t: -2.
Iconst_dist = lambda t: -1.

Iapps_nodist = [Iconst]
Iapps_before = [Iconst, Ioffset_dist]
Iapps = [Iconst, Iconst_dist]

# Observer parameters
α = 0.001 # 0.0005 # Default is 0.5. Had to decrease as P values were exploding.
γ = 5  # Default is 70, though Thiago's since lowered to 5. But 5 was causing psi to explode.

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
# Start by simulating the unperturbed system.
        
# Single neuron disturbance rejection

p_nodist = (Iapps_nodist, network_nodist)
z_0_nodist = x_0[:11] # Only one neur

dt = 0.01
tspan = (0.,Tfinal0)
start_time = time.time()
out_nodist = solve_ivp(lambda t, z: no_observer(t, z, p_nodist), tspan, z_0_nodist,rtol=tol,atol=tol,
                t_eval=np.linspace(0,Tfinal0,int(Tfinal0/dt)), method=solvemethod, dense_output=False)
end_time = time.time()
print("'Nodist' Simulation time: {}s".format(end_time-start_time),file=open("exp2.txt","a"))

t_nodist = out_nodist.t
sol_nodist = out_nodist.y

# Now use those parameters to initialise the next sim.
z_0_before = np.concatenate((x_0, x_0))
z_0_before[:11] = sol_nodist[:11,-1]

# %%
# Now run the perturbed system without the observer/controller.
p_before = (Iapps_before, network)

tspan = (0.,Tfinal1)
start_time = time.time()
out_before = solve_ivp(lambda t, z: no_observer(t, z, p_before), tspan, z_0_before,rtol=tol,atol=tol,
                t_eval=np.linspace(0,Tfinal1,int(Tfinal1/dt)), method=solvemethod, dense_output=False)
end_time = time.time()
print("'Before' Simulation time: {}s".format(end_time-start_time),file=open("exp2.txt","a"))

t_before = out_before.t
sol_before = out_before.y

# Now use those parameters to initialise the main sim.
z_0[:12] = sol_before[:12,-1]
# init = np.load("exp1_resg_0_4_solref.npy")
# z_0[:11] = init[:11]
# z_0[:11] = np.zeros(11)
neur_bef_start_idx = len(z_0) // 2 # Test this!
z_0[neur_bef_start_idx:neur_bef_start_idx+12] = sol_before[12:,-1] # init[11:]

# %%
# Integration initial conditions and parameters

tspan = (0.,Tfinal2)
# controller_on = True
varying_gT = (False,)
p = (Iapps,network,(α,γ),to_estimate,num_estimators,control_law,
     estimate_g_syns_g_els,0.,to_observe,varying_gT)

print("Starting simulation",file=open("exp2.txt","a"))
start_time = time.time()
out = solve_ivp(lambda t, z: main(t, z, p), tspan, z_0,rtol=tol,atol=tol,
                t_eval=np.linspace(0,Tfinal2,int(Tfinal2/dt)), method=solvemethod,
                dense_output=False)
end_time = time.time()
print(out.success,file=open("exp2.txt","a"))
print(f"Latest estimates: {out.y[24,-20:]}",file=open("exp2.txt","a"))
print("Simulation time: {}s".format(end_time-start_time),file=open("exp2.txt","a"))

t = out.t
sol = out.y

# Let's calculate Isyn and Isyn_hat
# v = sol[0,:]
# Isyn = syn.g * sol[11,:] * (v - neur_one.Esyn)
# Isyn_hat = sol[27,:] * sol[23,:] * (v - neur_one.Esyn)

# plt.plot(t,v)

# %%

# print("sol.max: {}".format(sol.max()),file=open("exp2.txt","a"))
# print("sol.min: {}".format(sol.min()),file=open("exp2.txt","a"))
print("Converting and saving...",file=open("exp2.txt","a"))
t_before=t_before.astype('float32')
sol_before=sol_before.astype('float32')
t=t.astype('float32')
t_nodist = t_nodist.astype('float32')
sol=sol.astype('float32')
sol_nodist=sol_nodist.astype('float32')
np.savez("exp2.npz", tbef=t_before, t=t, tnd=t_nodist, solbef=sol_before, 
         sol=sol, solnd=sol_nodist)#,ps=ps,ps_idx=ps_idx)

