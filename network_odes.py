# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 19:16:03 2021

@author: Rafi
"""
# from numba import jit, njit
import numpy as np

from network_and_neuron import Neuron, Network

def disturbance_rejection(to_reject, g_syns, syns_hat, Vs, Esyn, num_neurs):
    Isyn_estimates = np.zeros(num_neurs)
    for (neur_i, syn_i) in to_reject:
        Isyn_estimates[neur_i] = Isyn_estimates[neur_i] - g_syns[syn_i,neur_i] * syns_hat[syn_i,neur_i] * (Vs[neur_i] - Esyn)
    return -Isyn_estimates

def reference_tracking(Vs, ints_hat, syns_hat, gs, ref_gs, network, num_neurs, num_neur_gs):
    Es = network.neurons[0].Es # Same for every neuron, so can pick any.
    max_num_syns = network.max_num_syns
    cs = np.zeros(num_neurs)
    for (idx, neur) in enumerate(network.neurons):
        cs[idx] = neur.c
    adjusting_currents = reference_tracking_njit(Vs, ints_hat, syns_hat, gs, ref_gs, Es, num_neurs, num_neur_gs, max_num_syns, cs)
    return adjusting_currents

def reference_tracking_njit(Vs, ints_hat, syns_hat, gs, ref_gs, Es, num_neurs, num_neur_gs, max_num_syns, cs):
    adjusting_currents = np.zeros(num_neurs)
    g_diffs = ref_gs-gs
    terms = np.zeros((num_neur_gs+max_num_syns, num_neurs))
    for i in range(num_neurs):
        terms[:num_neur_gs,i] = np.divide(np.array([
                                    -ints_hat[0,i]**3*ints_hat[1,i]*(Vs[i]-Es[0]),
                                    -ints_hat[2,i]*(Vs[i]-Es[1]), # I_H
                                    -ints_hat[3,i]**2*ints_hat[4,i]*(Vs[i]-Es[2]), # I_T
                                    -ints_hat[5,i]**4*ints_hat[6,i]*(Vs[i]-Es[3]), # I_A
                                    -ints_hat[7,i]**4*(Vs[i]-Es[3]), # I_KD
                                    -ints_hat[8,i]*(Vs[i]-Es[2]), # I_L
                                    -ints_hat[9,i]**4*(Vs[i]-Es[3]), # I_KCa
                                    -(Vs[i]-Es[4])
                                ]),cs[i])
        terms[num_neur_gs:,i] = -syns_hat[:,i]*(Vs[i] - Es[5])
        adjusting_currents[i] = np.dot(g_diffs[:,i],terms[:,i]) # diag(A^T B)?
    return adjusting_currents

def hhmodel_reference_tracking(Vs, m̂s, ĥs, n̂s, syns_hat, gs, ref_gs, network, num_neurs, num_neur_gs):
    Es = network.neurons[0].Es # Same for every neuron, so can pick any.
    max_num_syns = network.max_num_syns
    cs = np.zeros(num_neurs)
    for (idx, neur) in enumerate(network.neurons):
        cs[idx] = neur.c
    adjusting_currents = hhmodel_reference_tracking_njit(Vs, m̂s, ĥs, n̂s, syns_hat, gs, ref_gs, Es, num_neurs, num_neur_gs, max_num_syns, cs)
    return adjusting_currents

def hhmodel_reference_tracking_njit(Vs, m̂s, ĥs, n̂s, syns_hat, gs, ref_gs, Es, num_neurs, num_neur_gs, max_num_syns, cs):
    adjusting_currents = np.zeros(num_neurs)
    g_diffs = ref_gs-gs
    terms = np.zeros((num_neur_gs+max_num_syns, num_neurs))
    for i in range(num_neurs):
        terms[:num_neur_gs,i] = np.divide(np.array([-m̂s[i]**3*ĥs[i]*(Vs[i]-Es[0]),-n̂s[i]**4*(Vs[i]-Es[1]),
                                    -(Vs[i]-Es[2])]),cs[i])
        terms[num_neur_gs:,i] = -syns_hat[:,i]*(Vs[i] - Es[3])
        adjusting_currents[i] = np.dot(g_diffs[:,i],terms[:,i]) # diag(A^T B)?
    return adjusting_currents
        
def main(t,z,p):
    Iapps = p[0]
    network = p[1]
    (α,γ) = p[2]
    to_estimate = p[3] # Which maximal conductances to estimate
    num_estimators = p[4] # Combine this with prev? But includes syns and res...
    controller_settings = p[5] # Control law to use for the neurons
    estimate_g_syns = p[6]
    estimate_g_res = p[7]
    
    # Assuming all the neurons are of the same model:
    num_neur_gates = network.neurons[0].NUM_GATES + network.max_num_syns
    len_neur_state = num_neur_gates + 1 # Effectively hardcoded below anyway.
    num_neur_gs = 8 # Na, H, T, A, KD, L, KCa, leak
    max_num_syns = network.max_num_syns
    num_neurs = len(network.neurons)
    
    # Now break out components of z.
    z_mat = np.reshape(z, (len(z)//num_neurs, num_neurs), order='F')
    # True system.
    Vs = z_mat[0,:]
    ms = z_mat[1,:]
    hs = z_mat[2,:]
    mHs = z_mat[3,:]
    mTs = z_mat[4,:]
    hTs = z_mat[5,:]
    mAs = z_mat[6,:]
    hAs = z_mat[7,:]
    mKDs = z_mat[8,:]
    mLs = z_mat[9,:]
    mCas = z_mat[10,:]
    ints = z_mat[1:11,:] # All the intrinsic gates.
    syns = z_mat[11:11+max_num_syns,:]
    # Terms for adaptive observer
    v̂s = z_mat[11+max_num_syns,:]
    m̂s = z_mat[11+max_num_syns+1,:] # Can remove
    ĥs = z_mat[11+max_num_syns+2,:] # ""
    n̂s = z_mat[11+max_num_syns+3,:] # ""
    ints_hat = z_mat[11+max_num_syns+1:11+max_num_syns+11,:] # All the intrinsic gate estimates.
    syns_hat = z_mat[11+max_num_syns+11:11+max_num_syns*2+11,:]
    idx_so_far = 11+max_num_syns*2+11 # Just to make code less ugly
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
    # Run controller
    if controller_settings[0] == "DistRej":
        if estimate_g_syns:
            g_syns = θ̂s[len(to_estimate):,:] # Start after intrinsic gs.
        else:
            g_syns = np.zeros((max_num_syns, num_neurs))
            for (idx, neur) in enumerate(network.neurons):
                g_syns[:neur.num_syns, idx] = neur.g_syns
        control_currs = disturbance_rejection(controller_settings[1], g_syns, syns_hat, Vs, network.neurons[0].Esyn, num_neurs)
        injected_currents = injected_currents + control_currs
    elif controller_settings[0] == "RefTrack":
        # If not estimating all the intrinsic gs, will feed controller a mix of true
        # and estimated gs. Need to generate this list of gs to feed in.
        neur_gs = np.zeros((num_neur_gs, num_neurs))
        neur_gs[to_estimate,:] = θ̂s[:len(to_estimate),:]
        
        # Use idxs of true gs where not estimating. So need to 'invert' to_estimate.
        # Ie need an array listing the elements that are NOT in to_estimate.
        tmp_list = np.array(range(num_neur_gs))
        tmp_list_mask = np.ones(num_neur_gs, dtype=bool)
        tmp_list_mask[to_estimate] = False
        known_g_idxs = tmp_list[tmp_list_mask]
        
        if known_g_idxs.any(): # Otherwise, neur_gs already fully populated with estimates.
            for (neur_idx, neur) in enumerate(network.neurons):
                neur_gs[known_g_idxs,neur_idx] = neur.gs[known_g_idxs]
        # if-else block below is same as for "DistRej" case above.
        if estimate_g_syns:
            g_syns = θ̂s[len(to_estimate):,:] # Start after intrinsic gs.
        else:
            g_syns = np.zeros((max_num_syns, num_neurs))
        for (idx, neur) in enumerate(network.neurons):
            g_syns[:neur.num_syns, idx] = neur.g_syns
        observer_gs = np.vstack((neur_gs, g_syns))
        control_currs = reference_tracking(Vs, ints_hat, syns_hat, observer_gs, 
                                           controller_settings[1], network, num_neurs, num_neur_gs)
        injected_currents = injected_currents + control_currs
    
    #### UP TO HERE !!!!
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

def hhmodel_main(t,z,p):
    Iapps = p[0]
    network = p[1]
    (α,γ) = p[2]
    to_estimate = p[3] # Which maximal conductances to estimate
    num_estimators = p[4] # Combine this with prev? But includes syns and res...
    controller_settings = p[5] # Control law to use for the neurons
    estimate_g_syns = p[6]
    estimate_g_res = p[7]
    
    # Assuming all the neurons are of the same model:
    num_neur_gates = network.neurons[0].NUM_GATES + network.max_num_syns
    len_neur_state = num_neur_gates + 1 # Effectively hardcoded below anyway.
    num_neur_gs = 3 # sodium, potassium, leak
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
    # Run controller
    if controller_settings[0] == "DistRej":
        if estimate_g_syns:
            g_syns = θ̂s[len(to_estimate):,:] # Start after intrinsic gs.
        else:
            g_syns = np.zeros((max_num_syns, num_neurs))
            for (idx, neur) in enumerate(network.neurons):
                g_syns[:neur.num_syns, idx] = neur.g_syns
        control_currs = disturbance_rejection(controller_settings[1], g_syns, syns_hat, Vs, network.neurons[0].Esyn, num_neurs)
        injected_currents = injected_currents + control_currs
    elif controller_settings[0] == "RefTrack":
        # If not estimating all the intrinsic gs, will feed controller a mix of true
        # and estimated gs. Need to generate this list of gs to feed in.
        neur_gs = np.zeros((num_neur_gs, num_neurs))
        neur_gs[to_estimate,:] = θ̂s[:len(to_estimate),:]
        
        # Use idxs of true gs where not estimating. So need to 'invert' to_estimate.
        # Ie need an array listing the elements that are NOT in to_estimate.
        tmp_list = np.array(range(num_neur_gs))
        tmp_list_mask = np.ones(num_neur_gs, dtype=bool)
        tmp_list_mask[to_estimate] = False
        known_g_idxs = tmp_list[tmp_list_mask]
        
        if known_g_idxs.any(): # Otherwise, neur_gs already fully populated with estimates.
            for (neur_idx, neur) in enumerate(network.neurons):
                neur_gs[known_g_idxs,neur_idx] = neur.gs[known_g_idxs]
        # if-else block below is same as for "DistRej" case above.
        if estimate_g_syns:
            g_syns = θ̂s[len(to_estimate):,:] # Start after intrinsic gs.
        else:
            g_syns = np.zeros((max_num_syns, num_neurs))
        for (idx, neur) in enumerate(network.neurons):
            g_syns[:neur.num_syns, idx] = neur.g_syns
        observer_gs = np.vstack((neur_gs, g_syns))
        control_currs = hhmodel_reference_tracking(Vs, m̂s, ĥs, n̂s, syns_hat, observer_gs, 
                                           controller_settings[1], network, num_neurs, num_neur_gs)
        injected_currents = injected_currents + control_currs
    
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

def no_observer(t,z,p):
    Iapps = p[0]
    network = p[1]
    
    # Assuming all the neurons are of the same model:
    num_neur_gates = network.neurons[0].NUM_GATES + network.max_num_syns
    len_neur_state = num_neur_gates + 1 # Effectively hardcoded below anyway.
    max_num_syns = network.max_num_syns
    num_neurs = len(network.neurons)
    
    # Now break out components of z.
    z_mat = np.reshape(z, (len(z)//num_neurs, num_neurs), order='F')
    Vs = z_mat[0,:]
    ms = z_mat[1,:]
    hs = z_mat[2,:]
    ns = z_mat[3,:]
    syns = z_mat[4:4+max_num_syns,:]
    
    injected_currents = np.zeros(num_neurs)
    for i in range(num_neurs): injected_currents[i] = Iapps[i](t)
    
    dvs = np.zeros(num_neurs)
    dms = np.zeros(num_neurs); dns = np.zeros(num_neurs); dhs = np.zeros(num_neurs);
    dsyns_mat = np.zeros((max_num_syns, num_neurs))
    for (i, neur) in enumerate(network.neurons):
        dvs[i] = neur.calc_dv_no_observer(Vs[i], ms[i], hs[i], ns[i], syns[:,i], injected_currents[i])
        
        v_pres = Vs[neur.pre_neurs]
        (dms[i], dhs[i], dns[i], dsyns_mat[:neur.num_syns,i]) = neur.gate_calcs(
            Vs[i], ms[i], hs[i], ns[i], syns[:,i], v_pres)
        
    dz_mat = np.vstack((dvs, dms, dhs, dns, dsyns_mat))
    dz = np.reshape(dz_mat, (len(z),), order='F')
    return dz