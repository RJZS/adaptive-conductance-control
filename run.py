# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 19:13:57 2021

@author: Rafi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from network_and_neuron import Neuron, Network
from network_odes import main

# Initial conditions
x_0 = [0, 0, 0, 0, 0]; # V, m, h, n, s
x̂_0 = [-60, 0.5, 0.5, 0.5, 0.5];
θ̂_0 = [60, 60, 10, 10, 0, 0, 0, 0, 0.1]; # [gNa, gK, gL, gsyn, gNa*ENa, gK*EK, gL*EL, gsyn*Esyn, 1]/c
P_0 = np.eye(9);
Ψ_0 = [0, 0, 0, 0, 0, 0, 0, 0, 0];

network = Network();

num_neurs = 2
num_estimators = len(θ̂_0)
len_neur_state = network.neurons[0].NUM_GATES + 1 # And synapses?!

z_0 = np.zeros((len_neur_state*2+num_estimators*2+num_estimators**2,num_neurs))
tmp = np.concatenate((x_0, x̂_0, θ̂_0, P_0.flatten(), Ψ_0))
for j in range(num_neurs): z_0[:,j] = tmp

# Integration initial conditions and parameters
dt = 0.01
Tfinal = 1.
tspan = (0.,Tfinal)
# controller_on = True
p = {"sup": 2}

out = solve_ivp(lambda t, z: main(t, z, p), tspan, z_0,rtol=1e-6,atol=1e-6)

t = out.t
sol = out.y