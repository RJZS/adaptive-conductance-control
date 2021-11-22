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
    num_estimators = p[4] # Combine this with prev? But includes syns and res...
    controller_law = p[5] # Control law to use for the neurons
    estimate_g_syns = p[6]
    estimate_g_res = p[7]
    
    # Assuming all the neurons are of the same model:
    num_neur_gates = network.neurons[0].NUM_GATES + network.max_num_syns
    len_neur_state = num_neur_gates + 1 # Effectively hardcoded below anyway.
    max_num_syns = network.max_num_syns
    num_neurs = len(network.neurons)
    
    # Now break out components of z.
    z_mat = np.reshape(z, (len(z)//num_neurs, num_neurs), order='F')
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
    # print(z_mat[10:14,:])
    Ps = np.reshape(z_mat[idx_so_far+num_estimators:idx_so_far+
                          num_estimators+num_estimators**2,:],
                    (num_estimators,num_estimators,num_neurs), order='F');
    for j in range(num_neurs):
        P = Ps[:,:,j]
        P = (P+np.transpose(P))/2
        Ps[:,:,j] = P
    Ψs = z_mat[idx_so_far+num_estimators+num_estimators**2:
               idx_so_far+num_estimators*2+num_estimators**2,:]
    
    injected_currents = np.zeros(num_neurs)
    for i in range(num_neurs): injected_currents[i] = Iapps[i](t)
    
    # Now make one time step. First, initialise the required vectors.
    bs = np.zeros(num_neurs)
    dvs = np.zeros(num_neurs); dv̂s = np.zeros(num_neurs)
    dms = np.zeros(num_neurs); dns = np.zeros(num_neurs); dhs = np.zeros(num_neurs);
    dm̂s = np.zeros(num_neurs); dn̂s = np.zeros(num_neurs); dĥs = np.zeros(num_neurs);
    dsyns_mat = np.zeros((max_num_syns, num_neurs))
    dsyns_hat_mat = np.zeros((max_num_syns, num_neurs))
    
    dθ̂s = np.zeros((num_estimators, num_neurs))
    dΨs = np.zeros((num_estimators, num_neurs))
    dPs = np.zeros((num_estimators, num_estimators, num_neurs))
    for (i, neur) in enumerate(network.neurons):
        # Need to 'reduce' terms. This is as vectors/matrices are sized for
        # the neuron/s with the largest number of synapses.
        num_neur_ests = len(to_estimate)
        if estimate_g_syns: num_neur_ests = num_neur_ests + neur.num_syns
        θ̂ = θ̂s[:num_neur_ests,i]
        P = Ps[:num_neur_ests,:num_neur_ests,i];
        Ψ = Ψs[:num_neur_ests,i]
        
        # Now, run the true system.
        (θ, ϕ, b) = neur.define_dv_terms(to_estimate, estimate_g_syns, 
                                         Vs[i], ms[i], hs[i], ns[i], syns[:,i], injected_currents[i])
        dvs[i] = np.dot(ϕ,θ) + b
        bs[i] = b # Will reuse this in the adaptive observer.
        # b here includes the input current, which is different from the paper I think
        
        v_pres = Vs[neur.pre_neurs]
        (dms[i], dhs[i], dns[i], dsyns_mat[:neur.num_syns,i]) = neur.gate_calcs(
            Vs[i], ms[i], hs[i], ns[i], syns[:,i], v_pres)
        
        # Finally, run the adaptive observer
        (_, ϕ̂, _) = neur.define_dv_terms(to_estimate, estimate_g_syns, 
                                         Vs[i], m̂s[i], ĥs[i], n̂s[i], syns_hat[:,i], injected_currents[i])
        
        dv̂s[i] = np.dot(ϕ̂,θ̂) + b + γ*(1+Ψ@P@Ψ.T)*(Vs[i]-v̂s[i])
        (dm̂s[i], dĥs[i], dn̂s[i], dsyns_hat_mat[:neur.num_syns,i]) = neur.gate_calcs(
            Vs[i], m̂s[i], ĥs[i], n̂s[i], syns_hat[:,i], v_pres)
        
        dθ̂s[:num_neur_ests,i] = γ*np.matmul(P,Ψ.T)*(Vs[i]-v̂s[i]);
        dΨs[:num_neur_ests,i] = np.array([-γ*Ψ + ϕ̂]);
        aux = np.outer(Ψ,Ψ)
        dP = α*P - P@aux@P;
        dP = (dP+np.transpose(dP))/2;
        dPs[:num_neur_ests,:num_neur_ests,i] = dP
        
    # Finally, need to stack and flatten everything.
    # To stack, need to 'reduce' dP to 2 axes instead of 3
    dPs = np.reshape(dPs, (num_estimators**2, num_neurs), order='F')
    dz_mat = np.vstack((dvs, dms, dhs, dns, dsyns_mat,
                         dv̂s, dm̂s, dĥs, dn̂s, dsyns_hat_mat, dθ̂s, dPs, dΨs))
    dz = np.reshape(dz_mat, (len(z),), order='F')
    return dz

