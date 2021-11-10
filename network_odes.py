# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 19:16:03 2021

@author: Rafi
"""
import numpy as np

from network_and_neuron import Neuron, Network

def main(t,z,p):
    Iapps = p[0]
    network = p[1]
    (α,γ) = p[2]
    to_estimate = p[3] # Which maximal conductances to estimate
    num_estimators = p[4] # Combine this with prev?
    controller_law = p[5] # Control law to use for the neurons
    estimate_g_syns = p[6]
    estimate_g_res = p[7]
    
    # Assuming all the neurons are of the same model:
    num_neur_gates = network.neurons[0].NUM_GATES
    len_neur_state = num_neur_gates + 1 # Effectively hardcoded below anyway.
    max_num_syns = network.max_num_syns
    num_neurs = len(network.neurons)
    
    # Now break out components of z.
    z_mat = np.reshape(z, (len(z)//num_neurs, num_neurs))
    # True system.
    Vs = z_mat[0,:]
    ms = z_mat[1,:]
    hs = z_mat[2,:]
    ns = z_mat[3,:]
    syns = z_mat[4:4+max_num_syns,:]
    # Terms for adaptive observer
    v̂s = z_mat[4+max_num_syns,:]
    m̂s = z_mat[4+max_num_syns+1,:]
    ĥs = z_mat[4+max_num_syns+2,:]
    n̂s = z_mat[4+max_num_syns+3,:]
    syns_hat = z_mat[4+max_num_syns+4:4+max_num_syns*2+4,:]
    idx_so_far = 4+max_num_syns*2+4 # Just to make code less ugly
    θ̂s = z_mat[idx_so_far:idx_so_far+num_estimators,:]
    Ps = np.reshape(z_mat[idx_so_far+num_estimators:idx_so_far+
                          num_estimators+num_estimators**2,:],
                    (num_estimators,num_estimators,num_neurs));    
    for j in range(num_neurs):
        P = Ps[:,:,j]
        P = (P+np.transpose(P))/2
        Ps[:,:,j] = P
    Ψs = z_mat[idx_so_far+num_estimators+num_estimators**2:
               idx_so_far+num_estimators*2+num_estimators**2,:]
    
    # Gating variable dynamics
    taus = [num_neur_gates, num_neurs]
    sigmas = [num_neur_gates, num_neurs]
    for (i, neur) in network.neurons:
        (taus[0,i],sigmas[0,i]) = neur.gating_m(Vs[i])
        (taus[1,i],sigmas[1,i]) = neur.gating_h(Vs[i])
        (taus[2,i],sigmas[2,i]) = neur.gating_n(Vs[i])
        (taus[3,i],sigmas[3,i]) = neur.gating_s(Vs[i])
        
    injected_currents = np.zeros(3)
    for i in range(num_neurs): injected_currents[i] = Iapps[i](t)
    
    # Up to line 274 of HH_odes. Ie θ = gsyn; ϕ = np.divide(-s*(v-Esyn),c);
        
    dz = -0*z
    return dz

