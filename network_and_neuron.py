# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 19:28:00 2021

@author: Rafi
"""
from typing import List
from numba import int32, float64
from numba.experimental import jitclass
from numba.typed import List as NumbaList
from numba import typed, typeof
import numpy as np

synapse_spec = [
    ('g', float64),
    ('pre_neur', int32),
    ]
@jitclass(synapse_spec)
class Synapse:
    def __init__(self, g, pre_neur):
        self.g = g
        self.pre_neur = pre_neur # Index of presynaptic neuron
        
syn_list_example = typed.List(); syn_list_example.append(Synapse(2.,0))
neuron_spec = [
    ('c', float64), ('gs',float64[:]), ('gNa',float64), ('gK',float64),
    ('gL',float64), ('ENa',float64), ('EK',float64), ('EL',float64),
    ('Esyn',float64), ('Es',float64[:]), ('num_syns',int32List),
    ('num_gates',float64), ('g_syns',float64[:]), ('pre_neurs',int32[:]),
    ('syns',typeof(syn_list_example))
    ]
# Note that 'gs' is a list which can include both floats and functions!
@jitclass(neuron_spec)
class Neuron: # Let's start with neuron in HH_odes not Thiago's HCO2_kinetics
    synapses: List[Synapse]
    
    def __init__(self, c, gs, synapses: List[Synapse]):
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
        # self.num_syns = len(synapses)
        
        self.num_gates = 3
        
        self.g_syns = np.zeros(self.num_syns)
        for (idx, syn) in enumerate(self.syns):
            self.g_syns[idx] = syn.g
            
        self.pre_neurs = np.zeros(self.num_syns)
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
        τ = Cbase + Camp*np.exp(-np.power((v-Vmax),2)/std**2);
        σ = np.divide(1, (1+np.exp(-(v-Vhalf)/k)));
        return τ, σ 
    
    # Sodium inactivation
    def gating_h(self, v):
        Vhalf = -62.;
        k = -7.;
        Vmax = -67.;
        std = 20.;
        Camp = 7.4;
        Cbase = 1.2;
        τ = Cbase + Camp*np.exp(-np.power((v-Vmax),2)/std**2);
        σ = np.divide(1, (1+np.exp(-(v-Vhalf)/k)));
        return τ, σ
    
    # Potassium activation
    def gating_n(self, v):
        Vhalf = -53.;
        k = 15.;
        Vmax = -79.;
        std = 50.;
        Camp = 4.7;
        Cbase = 1.1;
        τ = Cbase + Camp*np.exp(-np.power((v-Vmax),2)/std**2);
        σ = np.divide(1, (1+np.exp(-(v-Vhalf)/k)));
        return τ, σ
    
    # Synaptic gate
    def gating_s(self, v): # Terms are same as m unless stated.
        Vhalf = -45.; # From Dethier et al - 2015
        k = 2.; # From Dethier et al - 2015
        Vmax = -38.;
        std = 30.;
        Camp = 0.46;
        Cbase = 0.04;
        τ = Cbase + Camp*np.exp(-np.power((v-Vmax),2)/std**2);
        σ = np.divide(1, (1+np.exp(-(v-Vhalf)/k)));
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
        dm = 1/τm*(-m + σm);
        dh = 1/τh*(-h + σh);
        dn = 1/τn*(-n + σn);
        
        dsyns = np.zeros(self.num_syns)
        for (idx, syn) in enumerate(self.syns):
            (τs,σs) = self.gating_s(v_pres[idx])
            dsyns[idx] = 1/τs*(-syn_gates[idx] + σs);
            
        return (dm, dh, dn, dsyns)
    
    # TODO: Include resistive connections. 
    # Note this function spits out the length of vectors tailored to the neuron,
    # not the standardised 'max length' required by the ODE solver.
    def define_dv_terms(self, to_estimate, est_gsyns, v, m, h, n, syn_gates, I):
        # First deal with intrinsic conductances.
        gs = np.array([self.gNa, self.gK, self.gL, 1])
        terms = np.divide(np.array([-m**3*h*(v-self.ENa),-n**4*(v-self.EK),
                                    -(v-self.EL),I]),self.c)
        
        θ_intrins = np.zeros(len(to_estimate))
        ϕ_intrins = np.zeros(len(to_estimate))
        
        gs_del_idxs = np.zeros(len(to_estimate), dtype=np.int8)
        terms_del_idxs = np.zeros(len(to_estimate), dtype=np.int8)
        for (idx,val) in enumerate(to_estimate):
            θ_intrins[idx] = gs[val]
            ϕ_intrins[idx] = terms[val]
            gs_del_idxs[idx] = val; terms_del_idxs[idx] = val
        gs = np.delete(gs, gs_del_idxs)
        terms = np.delete(terms, terms_del_idxs)
            
        # Now look at synaptic terms.
        syn_terms = np.zeros(self.num_syns)
        for (idx, syn) in enumerate(self.syns):
            syn_terms[idx] = - syn_gates[idx] * (v - self.Esyn)
        
        if est_gsyns:
            θ = np.concatenate((θ_intrins, self.g_syns))
            ϕ = np.concatenate((ϕ_intrins, syn_terms))
            b = np.dot(gs, terms)
            return (θ, ϕ, b)
        else:
            b = np.dot(
                    np.concatenate((gs, self.g_syns)),
                    np.concatenate((terms, syn_terms))
                )
            return (θ_intrins, ϕ_intrins, b)
    
    def calc_dv_no_observer(self, v, m, h, n, syn_gates, I):
        gs = np.array([self.gNa, self.gK, self.gL, 1])
        terms = np.divide(np.array([-m**3*h*(v-self.ENa),-n**4*(v-self.EK),
                                    -(v-self.EL),I]),self.c)
        dv = np.dot(gs, terms)
        
        # In numpy, asterisk operator performs elementwise multiplication.
        dv = dv - self.g_syns * syn_gates * (v - self.Esyn) # NEED TO DIVIDE BY C??
        return dv
    
network_spec = [
    ('res_connect', float64[:]),
    ]
@jitclass
class Network:
    neurons: List[Neuron]
    
    def __init__(self, neurons, res_connect):
        self.neurons = neurons
        self.res_connect = res_connect
        
        max_num_syns = 0
        for neur in neurons:
            if neur.num_syns > max_num_syns:
                max_num_syns = neur.num_syns
        self.max_num_syns = max_num_syns

        
    
