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
from HH_odes import HH_observer, HH_ode

# True Parameters 
c = 1.
g = (120.,36.,0.3)
E = (55.,-77.,-54.4)
Iapp = lambda t : 2 + np.sin(2*np.pi/10*t)

# Observer parameters
α = 0.5
γ = 70

# Initial conditions
x_0 = [0, 0, 0, 0]; 
x̂_0 = [-60, 0.5, 0.5, 0.5];
θ̂_0 = [60, 60, 10, 0, 0, 0, 0];
P_0 = np.eye(7);
Ψ_0 = [0, 0, 0, 0, 0, 0, 0];

#%%

# Integration initial conditions and parameters
dt = 0.01
Tfinal = 140. # Default is 100.
tspan = (0.,Tfinal)
z_0 = np.concatenate((x_0, x̂_0, θ̂_0, P_0.flatten(), Ψ_0, x̂_0, θ̂_0))
p = (Iapp,c,g,E,(α,γ))

# Integrate
#prob = ODEProblem(HH_observer,z_0,tspan,p)
#sol = solve(prob,Tsit5(),reltol=1e-8,abstol=1e-8,saveat=0.1,maxiters=1e6)
out = solve_ivp(lambda t, z: HH_observer(t, z, p), tspan, z_0,rtol=1e-6,atol=1e-6)
# out = solve_ivp(lambda t, z: HH_ode(t, z, p), tspan, x_0)
t = out.t
sol = out.y

v = sol[0,:];
w = sol[1:4,:];
v̂ = sol[4,:];
ŵ = sol[5:8,:];
θ̂ = sol[8:15,:];
N = 15+49+7;
ṽ = sol[N,:];
w̃ = sol[N+1:N+4,:];
θ̃ = sol[N+4:N+11,:];

if save_data == 1:
    np.savetxt("data/HH_voltages.txt",  np.concatenate((t,v,v̂,ṽ),axis=1), delimiter=" ")
    np.savetxt("data/HH_parameters.txt",  
               np.concatenate((t,np.transpose(θ̂),np.transpose(θ̃)),axis=1), delimiter=" ")

## Plots
# Black dashed line is true value. Red is estimate.
# Green dashed is estimate by simpler adaptive observer,
# from Besancon (citation [4] in Thiago's paper).

# Need to sort plots.
# Solution?: https://www.futurelearn.com/info/courses/data-visualisation-with-python-matplotlib-and-visual-analysis/0/steps/192875
plt0 = plt.figure(); plt0ax = plt0.add_axes([0,0,1,1])
plt0ax.plot(t,v)
plt0ax.plot(t,v̂,color="red",linestyle="dashed")
plt0ax.plot(t,ṽ,color="green",linestyle="dashdot")

# gNa/c
plt1 = plt.figure(); plt1ax = plt1.add_axes([0,0,1,1])
plt1ax.plot([0,Tfinal],[g[0]/c,g[0]/c],color="black",linestyle="dashed",label="gNa/c")
plt1ax.plot(t,θ̂[0,:],color="red")
plt1ax.plot(t,θ̃[0,:],color="green",linestyle="dashdot")

# gK/c
plt2 = plt.figure(); plt2ax = plt2.add_axes([0,0,1,1])
plt2ax.plot([0,Tfinal],[g[1]/c,g[1]/c],color="black",linestyle="dashed",label="gK/c")
plt2ax.plot(t,θ̂[1,:],color="red")
plt2ax.plot(t,θ̃[1,:],color="green",linestyle= "dashdot")

# gL/c
plt3 = plt.figure(); plt3ax = plt3.add_axes([0,0,1,1])
plt3ax.plot([0,Tfinal],[g[2]/c,g[2]/c],color="black",linestyle="dashed",label="gL/c")
plt3ax.plot(t,θ̂[2,:],color="red")
plt3ax.plot(t,θ̃[2,:],color="green",linestyle= "dashdot")

# gNa*ENa/c
plt4 = plt.figure(); plt4ax = plt4.add_axes([0,0,1,1])
plt4ax.plot([0,Tfinal],[g[0]*E[0]/c,g[0]*E[0]/c],color="black",linestyle="dashed",label="gNa*ENa/c")
plt4ax.plot(t,θ̂[3,:],color="red")
plt4ax.plot(t,θ̃[3,:],color="green",linestyle= "dashdot")

# gK*EK/c
plt5 = plt.figure(); plt5ax = plt5.add_axes([0,0,1,1])
plt5ax.plot([0,Tfinal],[g[1]*E[1]/c,g[1]*E[1]/c],color="black",linestyle="dashed",label="gK*EK/c")
plt5ax.plot(t,θ̂[4,:],color="red")
plt5ax.plot(t,θ̃[4,:],color="green",linestyle= "dashdot")

# gL*EL/c
plt6 = plt.figure(); plt6ax = plt6.add_axes([0,0,1,1])
plt6ax.plot([0,Tfinal],[g[2]*E[2]/c,g[2]*E[2]/c],color="black",linestyle="dashed",label="gL*EL/c")
plt6ax.plot(t,θ̂[5,:],color="red")
plt6ax.plot(t,θ̃[5,:],color="green",linestyle= "dashdot")

# 1/c
plt7 = plt.figure(); plt7ax = plt7.add_axes([0,0,1,1])
plt7ax.plot([0,Tfinal],[1/c,1/c],color="black",linestyle="dashed",label="1/c")
plt7ax.plot(t,θ̂[6,:],color="red")
plt7ax.plot(t,θ̃[6,:],color="green",linestyle= "dashdot")