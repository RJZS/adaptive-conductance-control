# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 19:13:57 2021

@author: Rafi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from network_and_neuron import Synapse, Neuron, Network
from network_odes import main

# Initial conditions
x_0 = [0, 0, 0, 0, 0]; # V, m, h, n, s
x̂_0 = [-60, 0.5, 0.5, 0.5, 0.5];
θ̂_0 = [60, 60, 10, 10]; # [gNa, gK, gL, gsyn]
P_0 = np.eye(4);
Ψ_0 = [0, 0, 0, 0];
to_estimate = [0, 1, 2, 3]

syn = Synapse(2, 1)
neur_one = Neuron(1., [120.,36.,0.3, 2.], [syn])
neur_two = Neuron(1., [120.,36.,0.3, 2.], [])
network = Network([neur_one, neur_two], np.zeros((2,2)))

Iapp = lambda t : 2 + np.sin(2*np.pi/10*t)
Iapps = [Iapp, lambda t: 6]

# Observer parameters
α = 0.3 # Default is 0.5, I've set to 0.3.
γ = 100 # Default is 70, though Thiago's since lowered to 5.

control_law = None

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
z_0 = np.ravel(z_0)

# %%
# Integration initial conditions and parameters
dt = 0.01
Tfinal = 1.
tspan = (0.,Tfinal)
# controller_on = True
p = (Iapps,network,(α,γ),to_estimate,num_estimators,control_law)

out = solve_ivp(lambda t, z: main(t, z, p), tspan, z_0,rtol=1e-6,atol=1e-6,
                t_eval=np.linspace(0,Tfinal,int(Tfinal/dt)))

t = out.t
sol = out.y