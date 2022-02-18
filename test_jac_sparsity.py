# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 10:56:01 2022

@author: R
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

from network_and_neuron import Synapse, Neuron, Network
from network_odes import main, no_observer, find_jac_sparsity

Tfinal0 = 1.
Tfinal1 = 1.
Tfinal2 = 400.


tol = 1e-3
solvemethod = 'Radau'

# Tfinal = 11
# observe_start_time = 10. # 2000.

# Initial conditions - Single Neuron Disturbance Rejection
x_0 = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]; # V, m, h, mH, mT, hT, mA, hA, mKD, mL, Ca
x̂_0 = [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
θ̂_0 = np.ones(1); # Estimating 1syn. idx 24
P_0 = np.eye(1);
Ψ_0 = np.zeros(1);
to_estimate = np.array([0],dtype=np.int32)
to_observe = np.array([0], dtype=np.int32)
estimate_g_syns_g_els = False

# syn = Synapse(0.8, 1)

# neur_one_gs = np.array([120.,0.1,2.,0,80.,0.4,2.,0.,0.1])
# neur_dist_gs = np.array([120.,0.1,3.,0,80.,1.,2.,0.,0.1])

# Post-submission settings
syn = Synapse(2.5, 1)
neur_one_gs = np.array([60.,0.1,2.,0,80.,0.4,2.,0.,0.12])
neur_dist_gs = np.array([130.,0.1,3.2,0,80.,1.,2.,0.,0.1])

neur_one_nodist = Neuron(0.1, neur_one_gs, [], 0)
neur_dist = Neuron(0.1, neur_dist_gs, [], 1)
network_nodist = Network([neur_one_nodist, neur_dist], [])

neur_one = Neuron(0.1, neur_one_gs, [], 1)  # gNa, gH, gT, gA, gKD, gL, gKCa, gKir, gleak

res_g = 0.1
el_connects = np.array([[res_g, 0, 1]])
network = Network([neur_one, neur_dist], el_connects)

control_law = [3]

## Dist Rej Currents
# Iapp = lambda t : 2 + np.sin(2*np.pi/10*t)
def Ioffset_dist(t): # So two neurons in HCO don't burst simultaneously
    if t < 400:
        return -7.5
    else:
        return -1.
    
Iconst = lambda t: -2.
Iconst_dist = lambda t: -1.

Iapps = [Iconst,Ioffset_dist]

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

dt=0.01
tspan = (0.,Tfinal2)
# controller_on = True
varying_gT = (False,)
p = (Iapps,network,(α,γ),to_estimate,num_estimators,control_law,
     estimate_g_syns_g_els,0.,to_observe,varying_gT)

J_sparse = find_jac_sparsity(num_neurs, num_estimators, len_neur_state, max_num_syns) # Define sparsity matrix.
# J_sparse = np.ones((len(z_0),len(z_0)))
print("Starting simulation")
start_time = time.time()
out = solve_ivp(lambda t, z: main(t, z, p), tspan, z_0,rtol=tol,atol=tol,
                t_eval=np.linspace(0,Tfinal2,int(Tfinal2/dt)), method=solvemethod,
                dense_output=False,jac_sparsity=J_sparse)
end_time = time.time()
print(out.success)
print(f"Latest estimates: {out.y[24,-20:]}")
print("Simulation time: {}s".format(end_time-start_time))

t = out.t
sol = out.y
plt.plot(t,sol[12,:])
# Let's calculate Isyn and Isyn_hat
# v = sol[0,:]
# Isyn = syn.g * sol[11,:] * (v - neur_one.Esyn)
# Isyn_hat = sol[27,:] * sol[23,:] * (v - neur_one.Esyn)

# plt.plot(t,v)

