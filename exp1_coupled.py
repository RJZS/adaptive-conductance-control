# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 09:12:06 2021

@author: Rafi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

from network_and_neuron import Synapse, Neuron, Network
from network_odes import main, no_observer, find_jac_sparsity, init_state_update_dict

Tfinal1 = 3000. # How long to run 'before'
Tfinal2 = 24000. # How long to run observer+controller.
print("Tfinal1 = {}".format(Tfinal1),file=open("exp1.txt","a"))
print("Tfinal2 = {}".format(Tfinal2),file=open("exp1.txt","a"))

# Single neuron reference tracking.
# TODO: can I change the initialisation without 'instability'?
# Same for increasing alpha and decreasing gamma.

# Initial conditions - Single Neuron Reference Tracking
x_0 = [0,0,0,0,0,0,0,0,0,0,0]; # V, m, h, mH, mT, hT, mA, hA, mKD, mL, mCa
x̂_0 = [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
to_observe = np.array([0, 1], dtype=np.int32) # Neurons to observe.
# θ̂_0 = np.array([160.,1.,1.,1.,100.,1.,1.,1.,1.,1.]) # np.ones(10); # 1 g_el. From idx 22 to 31,
# or for second neuron, from 142+22.
θ̂_0 = np.ones(8);
P_0 = np.eye(8); # From idx 31 I think
Ψ_0 = np.zeros(8);
to_estimate = np.array([0,1,2,3,4,5,6,7], dtype=np.int32)
estimate_g_syns_g_els = False

# Remember, order of currents is Na, H, T, A, KD, L, KCA, KIR, 
neur_one_gs = np.array([0.,0.,0.,0,0.,0.,0.,0.,0.1])
neur_ref_gs = np.array([120.,0.1,2.,0,80.,0.4,2.,0.,0.1])

# Initially, the neurons are isolated.
neur_one = Neuron(0.1, neur_one_gs, np.array([]), 0)
neur_ref = Neuron(0.1, neur_ref_gs, np.array([]), 0)

res_g = 0.1
el_connects = np.array([[res_g, 0, 1]])
network = Network([neur_one, neur_ref], []) # for ref tracking. No gap junction yet.

Iconst = lambda t: -2
# Iconstsin = lambda t: -2 + np.sin(2*np.pi/10*t)
Iapps_ref = [Iconst, Iconst]
Iapps = [Iconst, Iconst]

# Observer parameters
α = 0.0005 # Default is 0.5. Had to decrease as P values were exploding.
γ = 2 # Default is 70, though Thiago's since lowered to 5. But 5 was causing psi to explode.

# gs of reference network. As is_exp1 is set to true, only first neuron will be controlled,
# but the way I've written the code I need to provide a vector for each neuron in the network.
# ref_gs = np.array([[120.,0.1,2.,0,80.,0.4,2.,0.,0.1], [120.,0.1,2.,0,80.,0.4,2.,0.,0.1]]).T
ref_gs = np.array([]).T

is_exp1 = True # Have created controller specifically for eq 1. Easier that way.
control_law = [0, ref_gs, is_exp1] # For first element, 0 is RefTrack, 1 is DistRej.
# control_law = [""]

num_neurs = len(network.neurons)
num_estimators = len(θ̂_0)
num_int_gates = network.neurons[0].NUM_GATES
len_neur_state = num_int_gates + 1
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
# First run the pre-observer system.
dt = 0.01
tspan = (0.,Tfinal1)

p_ref = (Iapps_ref, network)
z_0_ref = np.concatenate((x_0, x_0))
# z_0_ref[0] = -70.

start_time = time.time()
out_ref = solve_ivp(lambda t, z: no_observer(t, z, p_ref), tspan, z_0_ref,rtol=1e-3,atol=1e-3,
                t_eval=np.linspace(0,Tfinal1,int(Tfinal1/dt)), method='Radau', dense_output=False)
end_time = time.time()
print("'Ref' Simulation time: {}s".format(end_time-start_time),file=open("exp1.txt","a"))

t_ref = out_ref.t
sol_ref = out_ref.y

# Now use those parameters to initialise the next sim.
z_0[:11] = out_ref.y[:11,-1]
# init = np.load("exp1_resg_0_4_solref.npy")
# z_0[:11] = init[:11]
# z_0[:11] = np.zeros(11)
neur_ref_start_idx = len(z_0) // 2 # Test this!
z_0[neur_ref_start_idx:neur_ref_start_idx+11] = out_ref.y[11:,-1] # init[11:]


# %%
# Now set up the full connected network.
neur_one = Neuron(0.1, neur_one_gs, np.array([]), 1)
neur_ref = Neuron(0.1, neur_ref_gs, np.array([]), 1)
network = Network([neur_one, neur_ref], el_connects)

# Integration initial conditions and parameters
tspan = (0.,Tfinal2)
observe_start_time = 0.
varying_gT = (False,)
state_update_dict = init_state_update_dict(num_neurs, num_int_gates, max_num_syns, num_estimators)
p = (Iapps,network,(α,γ),to_estimate,num_estimators,control_law,
     estimate_g_syns_g_els,observe_start_time,to_observe,varying_gT,state_update_dict)
J_sparse = find_jac_sparsity(num_neurs, num_estimators, len_neur_state, max_num_syns).astype(int) # Define sparsity matrix.

print("Starting simulation",file=open("exp1.txt","a"))
start_time = time.time()
out = solve_ivp(lambda t, z: main(t, z, p), tspan, z_0,rtol=1e-3,atol=1e-3,
                t_eval=np.linspace(0,Tfinal2,int(Tfinal2/dt)), method='Radau',
                dense_output=False,jac_sparsity=J_sparse)
end_time = time.time()
print(out.success, file=open("exp1.txt","a"))
print("Simulation time: {}s".format(end_time-start_time),file=open("exp1.txt","a"))

t = out.t
sol = out.y

#%%
t=t.astype('float32')
sol=sol.astype('float32')
t_ref=t_ref.astype('float32')
sol_ref=sol_ref.astype('float32')
np.savez("exp1.npz", t=t,sol=sol, t_ref=t_ref,sol_ref=sol_ref)
