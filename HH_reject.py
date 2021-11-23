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
from HH_odes import HH_ode, HH_synapse_observer

# True Parameters 
c = 1.
g = (120.,36.,0.3, 2.) # Na, K, L, syn (default is 2)
E = (55.,-77.,-54.4, -80.)
Iapp = lambda t : 2 + np.sin(2*np.pi/10*t)

# Observer parameters
α = 0.3 # Default is 0.5, I've set to 0.3.
γ = 82 # Default is 70. Converges without phase-shift at 85 (on 5th no_syn spike), not 78. (After 200s).

# Initial conditions
x_0 = [0, 0, 0, 0, 0]; # V, m, h, n, s
x̂_0 = [-60, 0.5, 0.5, 0.5, 0.5];
θ̂_0 = [60, 60, 10, 10, 0, 0, 0, 0, 0.1]; # [gNa, gK, gL, gsyn, gNa*ENa, gK*EK, gL*EL, gsyn*Esyn, 1]/c
P_0 = np.eye(9);
Ψ_0 = [0, 0, 0, 0, 0, 0, 0, 0, 0];
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
out = solve_ivp(lambda t, z: HH_synapse_observer(t, z, p), tspan, z_0,rtol=1e-6,atol=1e-6,
                t_eval=np.linspace(0,Tfinal,int(Tfinal/dt)))
# out = solve_ivp(lambda t, z: HH_ode(t, z, p), tspan, x_0)
t = out.t
sol = out.y

v = sol[0,:];
w = sol[1:5,:];
v̂ = sol[5,:];
ŵ = sol[6:10,:];
θ̂ = sol[10:19,:];
v_nosyn = sol[113,:];

if save_data == 1:
    np.savetxt("data/HH_voltages.txt",  np.concatenate((t,v,v̂),axis=1), delimiter=" ")
    np.savetxt("data/HH_parameters.txt",  
               np.concatenate((t,np.transpose(θ̂)),axis=1), delimiter=" ")

# Calculate real and estimated synaptic current. Remember, the algorithm is
# online, so using the parameter estimates for that timestep.
Isyn = g[3] * w[3,:] * (v - E[3])
Isyn_hat = θ̂ [3,:] * ŵ[3,:] * (v - np.divide(θ̂[7,:],θ̂[3,:]))/θ̂[8,:]**2

#%% 
## Plots
# Black dashed line is true value. Red is estimate.
# Green dashed is estimate by simpler adaptive observer,
# from Besancon (citation [4] in Thiago's paper).

# Need to sort plots.
# Solution?: https://www.futurelearn.com/info/courses/data-visualisation-with-python-matplotlib-and-visual-analysis/0/steps/192875
plt0 = plt.figure(); plt0ax = plt0.add_axes([0,0,1,1])
j=5000
plt0ax.plot(t[j:],v[j:])
plt0ax.plot(t[j:],v̂[j:],color="red",linestyle="dashed")
plt0ax.plot(t[j:],v_nosyn[j:])
plt0ax.set_xlabel("t")
plt0ax.legend([r'$V$', r'$\hat{V}$', r'$V_{nosyn}$'])
plt0ax.set_title("Membrane potential")

plt9 = plt.figure(); plt9ax = plt9.add_axes([0,0,1,1])
plt9ax.plot(t[j:],v[j:]-v_nosyn[j:])
plt9ax.set_xlabel("t")
plt9ax.set_title(r'$V - V_{nosyn}$')

# Black dashed line is true value. Red is estimate.
# gNa/c
plt1 = plt.figure(); plt1ax = plt1.add_axes([0,0,1,1])
plt1ax.plot([0,Tfinal],[g[0]/c,g[0]/c],color="black",linestyle="dashed",label="gNa/c")
plt1ax.plot(t,θ̂[0,:],color="red")
plt1ax.set_xlabel("t")
plt1ax.legend(["True", "Estimated"])
plt1ax.set_title(r'$\bar{g}_{Na}/c$')

# gK/c
plt2 = plt.figure(); plt2ax = plt2.add_axes([0,0,1,1])
plt2ax.plot([0,Tfinal],[g[1]/c,g[1]/c],color="black",linestyle="dashed",label="gK/c")
plt2ax.set_xlabel("t")
plt2ax.plot(t,θ̂[1,:],color="red")
plt2ax.set_title(r'$g_K/c$')

# gL/c
plt3 = plt.figure(); plt3ax = plt3.add_axes([0,0,1,1])
plt3ax.plot([0,Tfinal],[g[2]/c,g[2]/c],color="black",linestyle="dashed",label="gL/c")
plt3ax.set_xlabel("t")
plt3ax.plot(t,θ̂[2,:],color="red")
plt3ax.set_title(r'$g_L/c$')

# gNa*ENa/c
plt4 = plt.figure(); plt4ax = plt4.add_axes([0,0,1,1])
plt4ax.plot([0,Tfinal],[g[0]*E[0]/c,g[0]*E[0]/c],color="black",linestyle="dashed",label="gNa*ENa/c")
plt4ax.set_xlabel("t")
plt4ax.plot(t,θ̂[4,:],color="red")
plt4ax.set_title(r'$g_{Na}E_{Na}/c$')

# gK*EK/c
plt5 = plt.figure(); plt5ax = plt5.add_axes([0,0,1,1])
plt5ax.plot([0,Tfinal],[g[1]*E[1]/c,g[1]*E[1]/c],color="black",linestyle="dashed",label="gK*EK/c")
plt5ax.plot(t,θ̂[5,:],color="red")
plt5ax.set_xlabel("t")
plt5ax.set_title(r'$\bar{g}_K E_K/c$')

# gL*EL/c
plt6 = plt.figure(); plt6ax = plt6.add_axes([0,0,1,1])
plt6ax.plot([0,Tfinal],[g[2]*E[2]/c,g[2]*E[2]/c],color="black",linestyle="dashed",label="gL*EL/c")
plt6ax.plot(t,θ̂[6,:],color="red")
plt6ax.set_xlabel("t")
plt6ax.set_title(r'$g_L E_L/c$')

# gs*Es/c
pltsyn = plt.figure(); pltsynax = pltsyn.add_axes([0,0,1,1])
pltsynax.plot([0,Tfinal],[g[3]*E[3]/c,g[3]*E[3]/c],color="black",linestyle="dashed",label="gs*Es/c")
pltsynax.plot(t,θ̂[7,:],color="red")
pltsynax.set_xlabel("t")
pltsynax.set_title(r'$g_s E_s/c$')


# 1/c
plt7 = plt.figure(); plt7ax = plt7.add_axes([0,0,1,1])
plt7ax.plot([0,Tfinal],[1/c,1/c],color="black",linestyle="dashed",label="1/c")
plt7ax.plot(t,θ̂[8,:],color="red")
plt7ax.set_xlabel("t")
plt7ax.set_title(r'$1/c$')

# Synaptic current (ignoring initial transient)
plt8 = plt.figure(); plt8ax = plt8.add_axes([0,0,1,1])
start_idx = 3000
plt8ax.plot(t[start_idx:],Isyn[start_idx:],label="I_syn",color="black",linestyle="dashed")
plt8ax.plot(t[start_idx:],Isyn_hat[start_idx:],color="red")
plt8ax.set_xlabel("t")
plt8ax.legend(["True", "Estimated"])
plt8ax.set_title("$I_{syn}$")
# %%
plt10 = plt.figure(); plt10ax = plt10.add_axes([0,0,1,1])
go_from = 980000
phase_shift = 5000

t_trunc = t[go_from:-phase_shift]
v_trunc = v[go_from+phase_shift:]
v_nosyn_trunc = v_nosyn[go_from:-phase_shift]

plt10ax.plot(t_trunc,v_trunc)
plt10ax.plot(t_trunc,v_nosyn_trunc)
plt10ax.set_xlabel("t")
plt10ax.legend(["v (phase-shifted)", "v if no synapse"])
plt10ax.legend([r'$v$', r'$v_{nosyn}$'])
plt10ax.set_title("Membrane potential (phase-shifted)")

plt11 = plt.figure(); plt11ax = plt11.add_axes([0,0,1,1])

zoom_idx = 5000

plt11ax.plot(t_trunc[zoom_idx:],v_trunc[zoom_idx:])
plt11ax.plot(t_trunc[zoom_idx:],v_nosyn_trunc[zoom_idx:])
plt11ax.set_xlabel("t")
plt11ax.legend([r'$v$', r'$v_{nosyn}$'])
plt11ax.set_title("Membrane potential (phase-shifted) (zoomed plot)")

plt12 = plt.figure(); plt12ax = plt12.add_axes([0,0,1,1])
plt12ax.plot(t_trunc[zoom_idx:],v_trunc[zoom_idx:]-v_nosyn_trunc[zoom_idx:])
plt12ax.set_xlabel("t")
plt12ax.set_title(r'$v - v_{nosyn}$')