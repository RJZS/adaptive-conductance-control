# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 19:28:00 2021

@author: Rafi
"""
import numpy as np

# Note that 'gs' is a list which can include both floats and functions!
class Neuron: # Let's start with neuron in HH_odes not Thiago's HCO2_kinetics
    def __init__(self, c, gs, Es):
        self.c = c
        self.gNa = gs[0]
        self.gK = gs[1]
        self.gL = gs[2]
        
        self.ENa = Es[0]
        self.EK = Es[1]
        self.EL = Es[2]
        
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
    
class Network:
    def __init__(self, neurons, syn_connect, res_connect):
        self.neurons = neurons
        self.syn_connect = syn_connect
        self.res_connect = res_connect
        
    
