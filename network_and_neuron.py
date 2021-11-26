# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 19:28:00 2021

@author: Rafi
"""
import numpy as np
from numba import njit, f8, i4

class Synapse:
    def __init__(self, g, pre_neur):
        self.g = g
        self.pre_neur = pre_neur # Index of presynaptic neuron
        

# Note that 'gs' is a list which can include both floats and functions!
class Neuron: # Let's start with neuron in HH_odes not Thiago's HCO2_kinetics
    NUM_GATES = 3
    
    def __init__(self, c, gs, synapses):
        self.c = c
        self.gNa = gs[0]
        self.gK = gs[1]
        self.gL = gs[2]
        self.gs = gs # Useful to keep as a list
        
        self.ENa = 55
        self.EK = -77
        self.EL = -54.4
        self.Esyn = -80
        self.Es = np.array([self.ENa, self.EK, self.EL, self.Esyn]) # Useful.
        
        self.syns = synapses
        self.num_syns = len(synapses)
        
        self.g_syns = np.zeros(self.num_syns)
        for (idx, syn) in enumerate(self.syns):
            self.g_syns[idx] = syn.g
            
        self.pre_neurs = np.zeros(self.num_syns, dtype=np.int8)
        for (idx, syn) in enumerate(self.syns):
            self.pre_neurs[idx] = syn.pre_neur
        
    # Sodium activation
    def gating_m(self, v):
        Vhalf = -40.;
        k = 9.;              #15 in izhikevich
        Vmax = -38.;
        std = 30.;
        Camp = 0.46;
        Cbase = 0.04;
        (τ, σ) = calc_tau_and_sigma(v, Cbase, Camp, Vmax, std, Vhalf, k)
        return τ, σ 
    
    # Sodium inactivation
    def gating_h(self, v):
        Vhalf = -62.;
        k = -7.;
        Vmax = -67.;
        std = 20.;
        Camp = 7.4;
        Cbase = 1.2;
        (τ, σ) = calc_tau_and_sigma(v, Cbase, Camp, Vmax, std, Vhalf, k)
        return τ, σ
    
    # Potassium activation
    def gating_n(self, v):
        Vhalf = -53.;
        k = 15.;
        Vmax = -79.;
        std = 50.;
        Camp = 4.7;
        Cbase = 1.1;
        (τ, σ) = calc_tau_and_sigma(v, Cbase, Camp, Vmax, std, Vhalf, k)
        return τ, σ
    
    # Synaptic gate
    def gating_s(self, v): # Terms are same as m unless stated.
        Vhalf = -45.; # From Dethier et al - 2015
        k = 2.; # From Dethier et al - 2015
        Vmax = -38.;
        std = 30.;
        Camp = 0.46;
        Cbase = 0.04;
        (τ, σ) = calc_tau_and_sigma(v, Cbase, Camp, Vmax, std, Vhalf, k)
        return τ, σ
    
    def neuron_calcs(self, v, m, h, n, I): # What's this function for?
        (τm,σm) = self.gating_m(v);
        (τh,σh) = self.gating_h(v);
        (τn,σn) = self.gating_n(v);
    
        g = [self.gNa, self.gK, self.gL]
        phi = [-m**3*h*(v-self.ENa),-n**4*(v-self.EK),-(v-self.EL)];
    
        dv = 1/self.c * (np.dot(phi,g) + I);
        dm = 1/τm*(-m + σm);
        dh = 1/τh*(-h + σh);
        dn = 1/τn*(-n + σn);
    
        return [dv,dm,dh,dn]
    
    def gate_calcs(self, v, m, h, n, syn_gates, v_pres):
        (τm,σm) = self.gating_m(v);
        (τh,σh) = self.gating_h(v);
        (τn,σn) = self.gating_n(v);
        dm = calc_dgate(τm, m, σm)
        dh = calc_dgate(τh, h, σh)
        dn = calc_dgate(τn, n, σn)
        
        dsyns = np.zeros(self.num_syns)
        for (idx, syn) in enumerate(self.syns):
            (τs,σs) = self.gating_s(v_pres[idx])
            dsyns[idx] = calc_dgate(τs, syn_gates[idx], σs);
            
        return (dm, dh, dn, dsyns)
    
    # TODO: Include resistive connections. 
    # Note this function spits out the length of vectors tailored to the neuron,
    # not the standardised 'max length' required by the ODE solver.
    def define_dv_terms(self, to_estimate, est_gsyns, v, m, h, n, syn_gates, I):
        # First deal with intrinsic conductances.
        gs = np.array([self.gNa, self.gK, self.gL, 1.])
        terms = calc_terms(v, m, h, n, self.ENa, self.EK, self.EL, self.c, I)
        
        gs, terms, θ_intrins, ϕ_intrins = calc_intrins_dv_terms(gs, terms, to_estimate)
        
        # Now look at synaptic terms.
        syn_terms = np.zeros(self.num_syns)
        for (idx, syn) in enumerate(self.syns):
            syn_terms[idx] = - syn_gates[idx] * (v - self.Esyn)
        
        if est_gsyns:
            θ, ϕ, b = calc_dv_terms_final_step_if_est_gsyns(θ_intrins, 
                                self.g_syns, ϕ_intrins, syn_terms, gs, terms)
            return (θ, ϕ, b)
        else:
            b = calc_dv_terms_final_step_if_not_est_gsyns(gs, self.g_syns, terms, syn_terms)
            return (θ_intrins, ϕ_intrins, b)
    
    def calc_dv_no_observer(self, v, m, h, n, syn_gates, I):
        gs = np.array([self.gNa, self.gK, self.gL, 1])
        terms = np.divide(np.array([-m**3*h*(v-self.ENa),-n**4*(v-self.EK),
                                    -(v-self.EL),I]),self.c)
        dv = np.dot(gs, terms)
        
        if syn_gates:
            # In numpy, asterisk operator performs elementwise multiplication.
            dv = dv - self.g_syns * syn_gates * (v - self.Esyn) # !! NEED TO DIVIDE BY C??
        return dv
    
class Network:
    def __init__(self, neurons, res_connect):
        self.neurons = neurons
        self.res_connect = res_connect
        
        max_num_syns = 0
        for neur in neurons:
            if neur.num_syns > max_num_syns:
                max_num_syns = neur.num_syns
        self.max_num_syns = max_num_syns
        
# Needed to take some functions out of the class, for numba.
@njit(cache=True)
def calc_tau_and_sigma(v, Cbase, Camp, Vmax, std, Vhalf, k):
    τ = Cbase + Camp*np.exp(-np.power((v-Vmax),2)/std**2);
    σ = np.divide(1, (1+np.exp(-(v-Vhalf)/k)));
    return τ, σ

@njit(cache=True)
def calc_dgate(τ, x, σ):
    dx = 1/τ*(-x + σ)
    return dx

@njit(cache=True)
def calc_terms(v, m, h, n, ENa, EK, EL, c, I):
    terms = np.divide(np.array([-m**3*h*(v-ENa),-n**4*(v-EK),
                                    -(v-EL),I]),c)
    return terms

# Providing types as 'typeof' was taking a long time in the profiler.
@njit((f8[:],f8[:],i4[:]), cache=True)
def calc_intrins_dv_terms(gs, terms, to_estimate):
    # First deal with intrinsic conductances.
    θ_intrins = np.zeros(len(to_estimate))
    ϕ_intrins = np.zeros(len(to_estimate))
    
    gs_del_idxs = np.zeros(len(to_estimate), dtype=np.int8)
    terms_del_idxs = np.zeros(len(to_estimate), dtype=np.int8)
    for (idx,val) in enumerate(to_estimate):
        θ_intrins[idx] = gs[val]
        ϕ_intrins[idx] = terms[val]
        gs_del_idxs[idx] = val; terms_del_idxs[idx] = val
    gs_mask = np.ones(len(gs), dtype=np.bool_); terms_mask = np.ones(len(terms), dtype=np.bool_)
    gs_mask[gs_del_idxs] = False; terms_mask[terms_del_idxs] = False;
    gs = gs[gs_mask]
    terms = terms[terms_mask]
    return (gs, terms, θ_intrins, ϕ_intrins)

@njit(cache=True)
def calc_dv_terms_final_step_if_est_gsyns(θ_intrins, g_syns, ϕ_intrins, syn_terms, gs, terms):
    θ = np.concatenate((θ_intrins, g_syns))
    ϕ = np.concatenate((ϕ_intrins, syn_terms))
    b = np.dot(gs, terms)
    return (θ, ϕ, b)

@njit(cache=True)
def calc_dv_terms_final_step_if_not_est_gsyns(gs, g_syns, terms, syn_terms):
    b = np.dot(
                    np.concatenate((gs, g_syns)),
                    np.concatenate((terms, syn_terms))
                )
    return b
