# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 11:09:52 2021

@author: Rafi
"""
# Based on Thiago's Julia code.
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Flag for saving data to .txt files 
save_data = 0
from HH_odes import HH_ode, HH_just_synapse_observer

# True Parameters 
c = 1.
g = (120.,36.,0.3, 2) # Na, K, L, syn
E = (55.,-77.,-54.4, -80.)
Iapp = lambda t : 2 + np.sin(2*np.pi/10*t)

# Observer parameters
α = 0.2 # Default is 0.5
γ = 70 # Default is 70

# Initial conditions
x_0 = [0, 0, 0, 0, 0]; # V, m, h, n, s
x̂_0 = [-60, 0.5, 0.5, 0.5, 0.5];
θ̂_0 = [10, 10]; # [gsyn, gsyn]
P_0 = np.eye(2);
Ψ_0 = [0, 0];
x_0_p = [0, 0, 0, 0]; # x_0 for presynaptic neuron

#%%

# Integration initial conditions and parameters
dt = 0.01
Tfinal = 100. # Default is 100.
tspan = (0.,Tfinal)
z_0 = np.concatenate((x_0, x̂_0, θ̂_0, P_0.flatten(), Ψ_0, x_0_p, x_0[:4]))
controller_on = True
p = (Iapp,c,g,E,(α,γ),controller_on)

# Integrate
#prob = ODEProblem(HH_observer,z_0,tspan,p)
#sol = solve(prob,Tsit5(),reltol=1e-8,abstol=1e-8,saveat=0.1,maxiters=1e6)
out = solve_ivp(lambda t, z: HH_just_synapse_observer(t, z, p), tspan, z_0,rtol=1e-6,atol=1e-6)
# out = solve_ivp(lambda t, z: HH_ode(t, z, p), tspan, x_0)
t = out.t
sol = out.y

v = sol[0,:];
w = sol[1:5,:];
v̂ = sol[5,:];
ŵ = sol[6:10,:];
θ̂ = sol[10:12,:];
v_pre = sol[18,:];
v_nosyn = sol[22,:];

if save_data == 1:
    np.savetxt("data/HH_voltages.txt",  np.concatenate((t,v,v̂),axis=1), delimiter=" ")
    np.savetxt("data/HH_parameters.txt",  
               np.concatenate((t,np.transpose(θ̂)),axis=1), delimiter=" ")

# Calculate real and estimated synaptic current. Remember, the algorithm is
# online, so using the parameter estimates for that timestep.
# Estimating E_syn in the correct way??
Isyn = g[3] * w[3,:] * (v - E[3])
Isyn_hat = θ̂ [0,:] * ŵ[3,:] * (v - E[3])

#%% 
## Plots
# Black dashed line is true value. Red is estimate.
# Green dashed is estimate by simpler adaptive observer,
# from Besancon (citation [4] in Thiago's paper).

# Need to sort plots.
# Solution?: https://www.futurelearn.com/info/courses/data-visualisation-with-python-matplotlib-and-visual-analysis/0/steps/192875
plt0 = plt.figure(); plt0ax = plt0.add_axes([0,0,1,1])
plt0ax.plot(t,v)
plt0ax.plot(t,v̂,color="red",linestyle="dashed")
plt0ax.plot(t,v_nosyn)
plt0ax.set_xlabel("t")
plt0ax.legend([r'$V$', r'$\hat{V}$', r'$V_{nosyn}$'])
plt0ax.set_title("Membrane potential")

plt9 = plt.figure(); plt9ax = plt9.add_axes([0,0,1,1])
plt9ax.plot(t,v-v_nosyn)
plt9ax.set_xlabel("t")
plt9ax.set_title(r'$V - V_{nosyn}$')

# Black dashed line is true value. Red is estimate.
# gsyn
plt1 = plt.figure(); plt1ax = plt1.add_axes([0,0,1,1])
plt1ax.plot([0,Tfinal],[g[3],g[3]],color="black",linestyle="dashed",label="gsyn/c")
plt1ax.plot(t,θ̂[0,:],color="red")
plt1ax.set_xlabel("t")
plt1ax.legend(["True", "Estimated"])
plt1ax.set_title(r'$\bar{g}_{syn}$')

# gsyn again
plt4 = plt.figure(); plt4ax = plt4.add_axes([0,0,1,1])
plt4ax.plot([0,Tfinal],[g[3],g[3]],color="black",linestyle="dashed",label="gsyn*Esyn/c")
plt4ax.set_xlabel("t")
plt4ax.plot(t,θ̂[1,:],color="red")
plt4ax.set_title(r'$g_{syn}$')

# Synaptic current (ignoring initial transient)
plt8 = plt.figure(); plt8ax = plt8.add_axes([0,0,1,1])
start_idx = 500
plt8ax.plot(t[start_idx:],Isyn[start_idx:],label="I_syn",color="black",linestyle="dashed")
plt8ax.plot(t[start_idx:],Isyn_hat[start_idx:],color="red")
plt8ax.set_xlabel("t")
plt8ax.legend(["True", "Estimated"])
plt8ax.set_title("$I_{syn}$")

# Synaptic current estimation error
plt13 = plt.figure(); plt13ax = plt13.add_axes([0,0,1,1])
start_idx = 2000
plt13ax.plot(t[start_idx:],Isyn_hat[start_idx:]-Isyn[start_idx:])
plt13ax.set_xlabel("t")
plt13ax.set_title("$\hat{I}_{syn}-I_{syn}$")

# %%
plt10 = plt.figure(); plt10ax = plt10.add_axes([0,0,1,1])
go_from = 3000
phase_shift = 260 # Increasing this shifts V 'to the left' relative to V_nosyn

t_trunc = t[go_from:-phase_shift]
v_trunc = v[go_from+phase_shift:]
v_nosyn_trunc = v_nosyn[go_from:-phase_shift]

plt10ax.plot(t_trunc,v_trunc)
plt10ax.plot(t_trunc,v_nosyn_trunc)
plt10ax.set_xlabel("t")
plt10ax.legend([r'$v$', r'$v_{nosyn}$'])
plt10ax.set_title("Membrane potential (phase-shifted)")

plt11 = plt.figure(); plt11ax = plt11.add_axes([0,0,1,1])

zoom_idx = 4000

plt11ax.plot(t_trunc[:zoom_idx],v_trunc[:zoom_idx])
plt11ax.plot(t_trunc[:zoom_idx],v_nosyn_trunc[:zoom_idx])
plt11ax.set_xlabel("t")
plt11ax.legend([r'$V$', r'$V_{nosyn}$'])
plt11ax.set_title("Membrane potential (phase-shifted, zoomed plot)")

plt12 = plt.figure(); plt12ax = plt12.add_axes([0,0,1,1])
plt12ax.plot(t_trunc,v_trunc-v_nosyn_trunc)
plt12ax.set_xlabel("t")
plt12ax.set_title(r'$V - V_{nosyn}$ (phase-shifted)')

# Code to find phase shift. Can ignore.
from scipy.signal import find_peaks
v_trunc_peaks, _ = find_peaks(v_trunc, height=0)
v_trunc_peaks

v_nosyn_trunc_peaks,_ = find_peaks(v_nosyn_trunc)
diffs = np.diff(v_nosyn_trunc_peaks)
