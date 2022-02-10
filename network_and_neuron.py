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
class Neuron:
    NUM_GATES = 10
    
    def __init__(self, c, gs, synapses, num_els):
        self.c = c
        self.gNa = gs[0]
        self.gH = gs[1]
        self.gT = gs[2]
        self.gA = gs[3]
        self.gKD = gs[4]
        self.gL = gs[5]
        self.gKCA = gs[6]
        self.gKir = gs[7]
        self.gleak = gs[8]
        self.gs = gs # Useful to keep as a list.
        
        # Es from upcoming book (notebook 2-3)
        self.ENa = 45.
        self.EH = -43.
        self.ECa = 120.
        self.EK = -90.
        self.Eleak = -55.
        
        # self.Esyn = -90. # Thiago's HCO2 sets to the same as EK.
        
        # self.EK = -77 # For rebound burster
        # self.Esyn = -120 # Needs to be below EK I think for rebound bursting...
        
        # self.Esyn = -70 # GABA A from Drion 2018 (see Fig1C, CaT_ncells.jl)
        self.Esyn = -90
        # self.Esyn = -85 # GABA B
        
        # # Drion plos 18
        # self.ENa = 50.
        # self.EK = -85.
        # self.ECa = 120.
        # self.EH = -20.

        self.Es = np.array([self.ENa, self.EH, self.ECa, self.EK, self.Eleak, self.Esyn]) # Useful.
        
        self.gL_for_Ca=0.4
        
        # Number electrical connections. Could automate finding this but not worth the time.
        self.num_els = num_els
        
        self.syns = synapses
        self.num_syns = len(synapses)
        
        self.g_syns = np.zeros(self.num_syns)
        for (idx, syn) in enumerate(self.syns):
            self.g_syns[idx] = syn.g
            
        self.pre_neurs = np.zeros(self.num_syns, dtype=np.int8)
        for (idx, syn) in enumerate(self.syns):
            self.pre_neurs[idx] = syn.pre_neur

        self.dints = np.zeros(self.NUM_GATES)
        self.dsyns = np.zeros(self.num_syns)
        self.taus = np.zeros(9)
        self.sigmas = np.zeros(9)
    
    def gate_calcs(self, v, int_gates, syn_gates, v_pres):
        self.taus[0], self.sigmas[0] = gating_mNa(v)
        self.taus[1], self.sigmas[1] = gating_hNa(v)
        self.taus[2], self.sigmas[2] = gating_mH(v)
        self.taus[3], self.sigmas[3] = gating_mT(v)
        self.taus[4], self.sigmas[4] = gating_hT(v)
        self.taus[5], self.sigmas[5] = gating_mA(v)
        self.taus[6], self.sigmas[6] = gating_hA(v)
        self.taus[7], self.sigmas[7] = gating_mKD(v)
        self.taus[8], self.sigmas[8] = gating_mL(v)

        self.dints[:9] = calc_dgate(self.taus, int_gates[:9], self.sigmas)
        
        dCa = (-0.1*self.gL_for_Ca*int_gates[8]*(v-self.ECa)-0.01*int_gates[9])/4
        self.dints[9] = dCa
        
        for (idx, syn) in enumerate(self.syns):
            self.dsyns[idx] = dGABA_A(v_pres[idx], syn_gates[idx])
            
        return (self.dints, self.dsyns)
    
    # Note this function spits out the length of vectors tailored to the neuron,
    # not the standardised 'max length' required by the ODE solver.
    def define_dv_terms(self, to_estimate, est_gsyns_gels, v, ints, syn_gates, I,
                        no_res_connections, el_connects, neur_idx, network_Vs):
        # First deal with intrinsic conductances.
        gs = np.concatenate((self.gs, [1.]))
        
        mKir = mKir_inf(v) # Gate modelled as instantaneous.
        terms = calc_terms(v, ints, mKir, self.ENa, self.EH, self.ECa, self.EK, self.Eleak, self.c, I)
        
        gs, terms, θ_intrins, ϕ_intrins = calc_intrins_dv_terms(gs, terms, to_estimate)
        
        # Now look at synaptic terms.
        syn_terms = np.zeros(self.num_syns)
        for (idx, syn) in enumerate(self.syns):
            syn_terms[idx] = - np.divide(syn_gates[idx] * (v - self.Esyn), self.c)
            
        # Now resistive terms.
        if not no_res_connections:
            (res_gs, res_terms, _) = calc_res_gs_and_terms(el_connects, neur_idx, network_Vs, self.c)
        else:
            res_gs = np.array([]); res_terms = np.array([])
        
        if est_gsyns_gels:
            θ, ϕ, b = calc_dv_terms_final_step_if_est_gsyns_gels(θ_intrins, 
                                self.g_syns, ϕ_intrins, syn_terms, gs, terms, res_gs, res_terms)
            return (θ, ϕ, b)
        else:
            b = calc_dv_terms_final_step_if_not_est_gsyns_gels(gs, self.g_syns, terms, syn_terms,
                                                               res_gs, res_terms)
            return (θ_intrins, ϕ_intrins, b)
    
    def calc_dv_no_observer(self, v, ints, syn_gates, I, no_res_connections, el_connects, neur_idx, network_Vs):
        gs = np.concatenate((self.gs, [1.]))
        mKir = mKir_inf(v) # Gate modelled as instantaneous.
        terms = calc_terms(v, ints, mKir, self.ENa, self.EH, self.ECa, self.EK, self.Eleak, self.c, I)
        dv = np.dot(gs, terms)
        
        if syn_gates.any():
            # In numpy, asterisk operator performs elementwise multiplication.
            syn_vector = np.divide(self.g_syns * syn_gates * (v - self.Esyn),self.c)
            dv = dv - np.sum(syn_vector)
        
        # Now resistive terms.
        if not no_res_connections:
            (res_gs, res_terms, _) = calc_res_gs_and_terms(el_connects, neur_idx, network_Vs, self.c)
            dv = dv + np.dot(res_gs, res_terms) # res_terms contains the negative sign.
        
        return dv        
    
    # Initialise the neuron by setting the gating variables to their 'x_inf' values.
    def initialise(self, v):
        (τm,σm) = self.gating_m(v);
        (τh,σh) = self.gating_h(v);
        (τmH,σmH) = self.gating_mH(v);
        (τmT,σmT) = self.gating_mT(v);
        (τhT,σhT) = self.gating_hT(v);
        (τmA,σmA) = self.gating_mA(v);
        (τhA,σhA) = self.gating_hA(v);
        (τmKD,σmKD) = self.gating_mKD(v);
        (τmL,σmL) = self.gating_mL(v);
        Ca = -10*self.gL_for_Ca*σmL*(v-self.ECa)
        
        init_gates = np.array([σm, σh, σmH, σmT, σhT, σmA, σhA, σmKD, σmL, Ca])
        init_gates = np.reshape(init_gates, (self.NUM_GATES,))
        return init_gates

# Note that 'gs' is a list which can include both floats and functions!
class HHModelNeuron: # Let's start with neuron in HH_odes not Thiago's HCO2_kinetics
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
        terms = hhmodel_calc_terms(v, m, h, n, self.ENa, self.EK, self.EL, self.c, I)
        
        gs, terms, θ_intrins, ϕ_intrins = calc_intrins_dv_terms(gs, terms, to_estimate)
        
        # Now look at synaptic terms.
        syn_terms = np.zeros(self.num_syns)
        for (idx, syn) in enumerate(self.syns):
            syn_terms[idx] = - syn_gates[idx] * (v - self.Esyn)
        
        if est_gsyns:
            θ, ϕ, b = calc_dv_terms_final_step_if_est_gsyns_gels(θ_intrins, 
                                self.g_syns, ϕ_intrins, syn_terms, gs, terms) # Need to add g_res, res_terms.
            return (θ, ϕ, b)
        else:
            b = calc_dv_terms_final_step_if_not_est_gsyns_gels(gs, self.g_syns, terms, syn_terms) # Here too.
            return (θ_intrins, ϕ_intrins, b)
    
    def calc_dv_no_observer(self, v, m, h, n, syn_gates, I):
        gs = np.array([self.gNa, self.gK, self.gL, 1])
        terms = hhmodel_calc_terms(v, m, h, n, self.ENa, self.EK, self.EL, self.c, I)
        dv = np.dot(gs, terms)
        
        if syn_gates:
            # In numpy, asterisk operator performs elementwise multiplication.
            dv = dv - self.g_syns * syn_gates * (v - self.Esyn) # !! NEED TO DIVIDE BY C??
        return dv
    
class Network:
    def __init__(self, neurons, el_connects):
        self.neurons = neurons
        
        # Electrical connections, in form [[g, neur_idx_1, neur_idx_2],[g, ...],...]
        self.el_connects = el_connects
        
        max_num_syns = 0
        for neur in neurons:
            if neur.num_syns > max_num_syns:
                max_num_syns = neur.num_syns
        self.max_num_syns = max_num_syns
        
        max_num_els = 0
        for neur in neurons:
            if neur.num_els > max_num_els:
                max_num_els = neur.num_els
        self.max_num_els = max_num_els
        
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

# @njit(cache=True)
# def calc_resistive_contribution_to_dv(res_mat, neur_idx, Vs, c):
#     I_res = 0
#     # First look at connections where this neuron is the 'pre' neuron.
#     for (i, g) in enumerate(res_mat[:,neur_idx]):
#         I_res = I_res - g*(Vs[i] - Vs[neur_idx])
#     # And then where it's the 'post' neuron.
#     for (i, g) in enumerate(res_mat[neur_idx,:]):
#         I_res = I_res - g*(Vs[neur_idx] - Vs[i])
#     dv_contribution = I_res / c
#     return dv_contribution

@njit(cache=True)
def res_condition(x, neur_idx): 
    f = np.zeros(len(x), dtype=np.bool_)
    ones = x[:,0]; twos = x[:,1] # Doing it this way due to numba error: 'iterating over 2D array'
    for i in range(len(x)):
        if ones[i] == neur_idx or twos[i] == neur_idx:
            f[i] = True
    return f

@njit(cache=True)
def calc_res_gs_and_terms(el_connects, neur_idx, Vs, c):
    # Filter to find connections involving our neuron.
    res_idxs = el_connects[:,1:]
    
    neur_res_bool = res_condition(res_idxs, neur_idx)
    neur_res_idxs = res_idxs[neur_res_bool]
    neur_res_idxs = neur_res_idxs.astype(np.int8)
    neur_res_gs = el_connects[neur_res_bool,0]
        
    # Now calculate terms.
    terms = np.zeros(len(neur_res_idxs))
    for i in range(len(neur_res_idxs)):
        other_neur_idx = np.extract(neur_res_idxs[i] != neur_idx, neur_res_idxs[i])[0]
        terms[i] = - (Vs[neur_idx] - Vs[other_neur_idx])
    terms = np.divide(terms, c)
    
    return (neur_res_gs, terms, neur_res_bool)

@njit(cache=True)
def calc_terms(v, ints, mKir, ENa, EH, ECa, EK, Eleak, c, I):
    terms = np.divide(np.array([
                -ints[0]**3*ints[1]*(v-ENa), # I_Na
                -ints[2]*(v-EH), # I_H
                -ints[3]**2*ints[4]*(v-ECa), # I_T
                -ints[5]**4*ints[6]*(v-EK), # I_A
                -ints[7]**4*(v-EK), # I_KD
                -ints[8]*(v-ECa), # I_L
                -(ints[9]/(15+ints[9]))**4*(v-EK), # I_KCa
                -mKir*(v-EK), # I_Kir
                -(v-Eleak),
                I
            ]),c)
    return terms

@njit(cache=True)
def hhmodel_calc_terms(v, m, h, n, ENa, EK, EL, c, I):
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
def calc_dv_terms_final_step_if_est_gsyns_gels(θ_intrins, g_syns, ϕ_intrins, syn_terms, gs, terms, g_res, res_terms):
    θ = np.concatenate((θ_intrins, g_syns, g_res))
    ϕ = np.concatenate((ϕ_intrins, syn_terms, res_terms))
    b = np.dot(gs, terms)
    return (θ, ϕ, b)

@njit(cache=True)
def calc_dv_terms_final_step_if_not_est_gsyns_gels(gs, g_syns, terms, syn_terms, g_res, res_terms):
    b = np.dot(
                    np.concatenate((gs, g_syns, g_res)),
                    np.concatenate((terms, syn_terms, res_terms))
                )
    return b

## Gating Functions for full model
# Na-current (m=activation variable, h=inactivation variable)
@njit(cache=True)
def gating_mNa(V):
    alpha_m = -0.025*(V+40.)/( np.exp(-(V+40)/10) - 1.0 )
    beta_m = np.exp(-(V+65)/18)
    τ = 1. / (alpha_m + beta_m)/5 # Activation time-constant
    σ = alpha_m / (alpha_m + beta_m) # Activation function
    return τ, σ

@njit(cache=True)
def gating_hNa(V):
    alpha_h = 0.0175*np.exp(-(V+65)/20)
    beta_h = 0.25/(1.0 + np.exp(-(V+35)/10) )
    τ = 1 / (alpha_h + beta_h)/5 # Inactivation time-constant
    σ = alpha_h / (alpha_h + beta_h) # Inactivation function
    return τ, σ

# KD-current (mKD=activation variable)
@njit(cache=True)
def gating_mKD(V):
    Kdshift = 10.0
    alpha_mKd = 0.0025*(V+55.-Kdshift)/(1. - np.exp(-(V+55.-Kdshift)/10.) )
    beta_mKd = 0.03125*np.exp(-(V+65-Kdshift)/80.)
    τ = 1. / (alpha_mKd + beta_mKd)/5 # Activation time-constant
    σ = alpha_mKd / (alpha_mKd + beta_mKd) # Activation function
    return τ, σ

# H-current (mH=activation variable)
@njit(cache=True)
def gating_mH(V):
    alpha_mH = np.exp(-14.59-(0.086*V))
    beta_mH = np.exp(-1.87+(0.0701*V))
    τ = 1 /(alpha_mH + beta_mH)
    σ = alpha_mH /(alpha_mH + beta_mH)
    return τ, σ

# A-current (mA=activation variable, hA=inactivation variable)
@njit(cache=True)
def gating_mA(V):
    τ = 0.37 + 1/(np.exp((V+35.82)/19.697)+np.exp((V+79.69)/-12.7))/5
    σ = 1/(1+np.exp(-(V+90)/8.5))
    return τ, σ

@njit(cache=True)
def gating_hA(V):
    tau_hA = 19
    if V < -63:
        tau_hA = 1/(np.exp((V+46.05)/5)+np.exp((V+238.4)/-37.45))
    τ = tau_hA / 5
    σ = 1/(1+np.exp((V+78)/6))
    return τ, σ

# T-type Ca-current (mt=activation variable, ht=inactivation variable)
@njit(cache=True)
def gating_mT(V):
    τ = 0.612 + 1/(np.exp(-(V+131.6)/16.7)+np.exp((V+16.8)/18.2))
    σ = 1/(1+np.exp(-(V+57)/6.2))
    return τ, σ

@njit(cache=True)
def gating_hT(V):
    if V < -80:
        tau_ht = np.exp((V+467)/66.6)
    else:
        tau_ht = (np.exp(-(V+21.88)/10.2)+28)
    τ = tau_ht
    σ = 1/(1+np.exp((V+81)/4.03))
    return τ, σ

# L-type Ca-current (mL=activation variable) (from Drion2011)
@njit(cache=True)
def gating_mL(V):
    τ = (72*np.exp(-(V+45.)**2/400)+6.)
    σ = 1/(1+np.exp(-(V+55.)/3))
    return τ, σ

# Kir-current (mKIR=activation variable). Modelled as instantaneous.
@njit(cache=True)
def mKir_inf(V):
    σ = 1/(1+np.exp((V+97.9+10)/9.7)) # Activation function
    return σ

# Synapse
@njit(cache=True)
def dGABA_A(V, s):
    ds = 0.53*(1/(1+np.exp(-(V-2)/5)))*(1-s)-0.18*s
    return ds
