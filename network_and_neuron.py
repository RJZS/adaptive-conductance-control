# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 19:28:00 2021

@author: Rafi
"""
import numpy as np

class Synapse:
    def __init__(self, g, pre_neur):
        self.g = g
        self.pre_neur = pre_neur # Index of presynaptic neuron
        

# Note that 'gs' is a list which can include both floats and functions!
class Neuron: # Let's start with neuron in HH_odes not Thiago's HCO2_kinetics
    NUM_GATES = 4 # Including synapse
    
    def __init__(self, c, gs, synapses):
        self.c = c
        self.gNa = gs[0]
        self.gK = gs[1]
        self.gL = gs[2]
        
        self.ENa = 120
        self.EK = 36
        self.EL = 0.3
        self.Esyn = -80
        
        self.syns = synapses
        self.num_syns = len(synapses)
        
        self.syn_gs = np.zeros(self.num_syns)
        for (idx, syn) in enumerate(self.syns):
            self.syn_gs[idx] = syn.g
        
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
    
    # TODO: Include resistive connections.
    def split_out_terms(self, to_estimate, est_gsyns, v, m, h, n, syn_gates, I):
        # First deal with intrinsic conductances.
        gs = [self.gNa, self.gK, self.gL]
        terms = np.divide(np.array([-m**3*h*(v-self.ENa),-n**4*(v-self.EK),
                                    -(v-self.EL),I]),self.c)
        
        θ_intrins = np.zeros(len(to_estimate))
        ϕ_intrins = np.zeros(len(to_estimate))
        
        for (idx,val) in enumerate(to_estimate):
            θ_intrins[idx] = gs[val]
            ϕ_intrins[idx] = terms[val]
            gs.pop[val]
            terms.pop[val]
            
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
    
class Network:
    def __init__(self, neurons, res_connect):
        self.neurons = neurons
        self.res_connect = res_connect
        
        max_num_syns = 0
        for neur in neurons:
            if neur.num_syns > max_num_syns:
                max_num_syns = neur.num_syns
        self.max_num_syns = max_num_syns

        
    
