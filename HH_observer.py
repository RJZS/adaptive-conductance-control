# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 11:09:52 2021

@author: Rafi
"""
# Based on Thiago's Julia code.
import numpy as np
import matplotlib.pyplot as plt

# Flag for saving data to .txt files 
save_data = 0

from HH_odes import HH_observer

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
P_0 = Matrix(I, 7, 7);
Ψ_0 = [0, 0, 0, 0, 0, 0, 0];

# Integration initial conditions and parameters
dt = 0.01
Tfinal = 100.
tspan = (0.,Tfinal)
z_0 = [x_0, x̂_0, θ̂_0, reshape(P_0,1,49), Ψ_0, x̂_0, θ̂_0]
p = (Iapp,c,g,E,(α,γ))

# Integrate
prob = ODEProblem(HH_observer,z_0,tspan,p)
sol = solve(prob,Tsit5(),reltol=1e-8,abstol=1e-8,saveat=0.1,maxiters=1e6)
t = sol.t
v = sol[1,1,:];
w = sol[1,2:4,:];
v̂ = sol[1,5,:];
ŵ = sol[1,6:8,:];
θ̂ = sol[1,9:15,:];
N = 15+49+7;
ṽ = sol[1,N+1,:];
w̃ = sol[1,N+2:N+4,:];
θ̃ = sol[1,N+5:N+11,:];

if save_data == 1:
    np.savetxt("data/HH_voltages.txt",  hcat(t,v,v̂,ṽ), delimiter=" ")
    np.savetxt("data/HH_parameters.txt",  hcat(t,np.transpose(θ̂),np.transpose(θ̃)), delimiter=" ")

## Plots
# Black dashed line is true value. Red is estimate.
# Green dashed is estimate by simpler adaptive observer,
# from Besancon (citation [4] in Thiago's paper).

# Need to sort plots.
# Solution?: https://www.futurelearn.com/info/courses/data-visualisation-with-python-matplotlib-and-visual-analysis/0/steps/192875
# TODO: REMEMBER TO CHANGE INDICES, INCLUDING ON OTHER FILE!!!!
plt0 = plt.plot(t,v,
                t,v̂,color="red",linestyle="dashed",
                t,ṽ,color="green",linestyle="dashdotted")
# plot(plt1,plt2,plt3,layout=(3,1),legend=false)

# gNa/c
plt1 = plt.plot([0,Tfinal],[g[1]/c,g[1]/c],linecolor="black",linestyle="dashed",labels="gNa/c",
            t,θ̂[1,:],linecolor="red",
            t,θ̃[1,:],linecolor="green",linestyle="dashdotted")

# gK/c
plt2 = plt.plot([0,Tfinal],[g[2]/c,g[2]/c],linecolor="black",linestyle="dashed",labels="gK/c",
            t,θ̂[2,:],linecolor="red",
            t,θ̃[2,:],linecolor="green",linestyle= "dashdotted")

# gL/c
plt3 = plt.plot([0,Tfinal],[g[3]/c,g[3]/c],linecolor="black",linestyle="dashed",labels="gL/c",
            t,θ̂[3,:],linecolor="red",
            t,θ̃[3,:],linecolor="green",linestyle= "dashdotted")

# gNa*ENa/c
plt4 = plt.plot([0,Tfinal],[g[1]*E[1]/c,g[1]*E[1]/c],linecolor="black",linestyle="dashed",labels="gNa*ENa/c",
            t,θ̂[4,:],linecolor="red",
            t,θ̃[4,:],linecolor="green",linestyle= "dashdotted")

# gK*EK/c
plt5 = plt.plot([0,Tfinal],[g[2]*E[2]/c,g[2]*E[2]/c],linecolor="black",linestyle="dashed",labels="gK*EK/c",
            t,θ̂[5,:],linecolor="red",
            t,θ̃[5,:],linecolor="green",linestyle= "dashdotted")

# gL*EL/c
plt6 = plt.plot([0,Tfinal],[g[3]*E[3]/c,g[3]*E[3]/c],linecolor="black",linestyle="dashed",labels="gL*EL/c",
            t,θ̂[6,:],linecolor="red",
            t,θ̃[6,:],linecolor="green",linestyle= "dashdotted")

# 1/c
plt7 = plt.plot([0,Tfinal],[1/c,1/c],linecolor="black",linestyle="dashed",labels="1/c",
            t,θ̂[7,:],linecolor="red",
            t,θ̃[7,:],linecolor="green",linestyle= "dashdotted")