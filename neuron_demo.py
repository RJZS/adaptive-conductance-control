# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 09:12:06 2021

@author: Rafi
"""
# Plot of the different ionic currents that make up the model, for inclusion in
# Example 2 of the paper. Parameters are same as the ref neuron in exp1_coupled.py.
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

from network_and_neuron import Synapse, Neuron, Network, mKir_inf
from network_odes import no_observer

Tfinal1 = 1000.
print("Tfinal1 = {}".format(Tfinal1),file=open("neuron_demo.txt","a"))

# Single neuron reference tracking.
# TODO: can I change the initialisation without 'instability'?
# Same for increasing alpha and decreasing gamma.

# Initial conditions - Single Neuron Reference Tracking
x_0 = [0,0,0,0,0,0,0,0,0,0,0]; # V, m, h, mH, mT, hT, mA, hA, mKD, mL, mCa

# Remember, order of currents is Na, H, T, A, KD, L, KCA, KIR, 
neur_gs = np.array([120.,0.1,2.,0,80.,0.4,2.,0.,0.1])

neur = Neuron(0.1, neur_gs, np.array([]), 0)

network = Network([neur], [])

Iconst = lambda t: -2
# Iconstsin = lambda t: -2 + np.sin(2*np.pi/10*t)
Iapps_ref = [Iconst, Iconst]
Iapps = [Iconst, Iconst]


# z_0[0] = -70.

# %%
# First run the pre-observer system.
dt = 0.01
tspan = (0.,Tfinal1)

p_ref = (Iapps_ref, network)
z_0_ref = x_0
# z_0_ref[0] = -70.

start_time = time.time()
out_ref = solve_ivp(lambda t, z: no_observer(t, z, p_ref), tspan, z_0_ref,rtol=1e-3,atol=1e-3,
                t_eval=np.linspace(0,Tfinal1,int(Tfinal1/dt)), method='Radau', dense_output=False)
end_time = time.time()
print("'Ref' Simulation time: {}s".format(end_time-start_time),file=open("neuron_demo.txt","a"))

t = out_ref.t
sol = out_ref.y

plt.plot(t,sol[0,:])

# %%
# Compute the ionic currents

def calc_terms(v, ints, mKir, ENa, EH, ECa, EK, Eleak, c, I):
    terms = np.divide(np.array([
                -ints[0]**3*ints[1]*(v-ENa), # I_Na
                -ints[2]*(v-EH), # I_H
                -ints[3]**2*ints[4]*(v-ECa), # I_T
                -ints[5]**4*ints[6]*(v-EK), # I_A
                -ints[7]**4*(v-EK), # I_KD
                -ints[8]*(v-ECa), # I_L
                -(ints[9]/(15+ints[9]))**4*(v-EK), # I_KCa
                -mKir*(v-EK), # I_Kir
                -(v-Eleak),
                I
            ]),c)
    return terms
Is = np.zeros(len(t))
for idx in range(len(t)):
    Is[idx] = Iapps[0](t[idx])
terms = calc_terms(sol[0,:], sol[1:,:], mKir_inf(sol[0,:]), neur.ENa, neur.EH, neur.ECa, neur.EK, neur.Eleak, neur.c, Is)

# Now multiply by gs
currs = np.zeros(terms[:-1].shape)
for idx in range(9):
    currs[idx,:] = neur_gs[idx]*terms[idx,:]

# To get one burst, zoom into approx t = 600 to t = 800.

#%%
t=t.astype('float32')
sol=sol.astype('float32')
currs=currs.astype('float32')
np.savez("neuron_demo.npz",t=t,sol=sol,currs=currs)
