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
    
    def __init__(self, c, gs, synapses):
        self.c = c
        self.gNa = gs[0]
        self.gH = gs[1]
        self.gT = gs[2]
        self.gA = gs[3]
        self.gKD = gs[4]
        self.gL = gs[5]
        self.gKCA = gs[6]
        self.gleak = gs[7]
        self.gs = gs # Useful to keep as a list.
        
        self.ENa = 45
        self.EH = -43
        self.ECa = 120
        self.EK = -90
        self.Eleak = -55
        self.Esyn = -90 # Thiago's HCO2 sets to the same as EK.
        self.Es = np.array([self.ENa, self.EH, self.ECa, self.EK, self.Eleak, self.Esyn]) # Useful.
        
        self.syns = synapses
        self.num_syns = len(synapses)
        
        self.g_syns = np.zeros(self.num_syns)
        for (idx, syn) in enumerate(self.syns):
            self.g_syns[idx] = syn.g
            
        self.pre_neurs = np.zeros(self.num_syns, dtype=np.int8)
        for (idx, syn) in enumerate(self.syns):
            self.pre_neurs[idx] = syn.pre_neur
        
    # NEED TO CHANGE THE GATES TO MATCH HCO2!!
    # Sodium activation
    # def gating_m(self, v):
    #     Vhalf = -40.;
    #     k = 9.;              #15 in izhikevich
    #     Vmax = -38.;
    #     std = 30.;
    #     Camp = 0.46;
    #     Cbase = 0.04;
    #     (τ, σ) = calc_tau_and_sigma(v, Cbase, Camp, Vmax, std, Vhalf, k)
    #     return τ, σ 
    
    ## Model gating functions. CURRENTLY TRANSLATING THESE FROM JULIA!
    # Need to check divisions are doing the right thing, especially if I
    # decide to vectorise (ie calculate for the whole network at once).

    # Synaptic current
    V12syn=-20.0;
    ksyn=4.0
    def msyn_inf(V): 1.0 / ( 1.0 + np.exp(-(V-V12syn)/ksyn) )
    # WHAT ABOUT TAU_SYN?
    
    # Na-current (m=activation variable, h=inactivation variable)
    vh_α_m = 40
    vh_β_m = 65
    k_α_m = 10
    k_β_m = 18
    def alpha_m(V): -0.025*(V+vh_α_m)/(np.exp(-(V+vh_α_m)/k_α_m) - 1.0 )
    def beta_m(V): np.exp(-(V+vh_β_m)/k_β_m)
    def m_inf(V): alpha_m(V) / (alpha_m(V) + beta_m(V))
    def tau_m(V): 1.0 / (alpha_m(V) + beta_m(V))
    
    ## up to here!
    vh_α_h = 65
    vh_β_h = 35
    k_α_h = 20
    k_β_h = 10
    alpha_h(V::Float64,i) = 0.0175*exp(-(V+vh_α_h[i])/k_α_h[i])
    beta_h(V::Float64,i) = 0.25/(1.0 + exp(-(V+vh_β_h[i])/k_β_h[i]) )
    h_inf(V::Float64,i) = alpha_h(V,i) / (alpha_h(V,i) + beta_h(V,i))
    tau_h(V::Float64,i) = 1 / (alpha_h(V,i) + beta_h(V,i))
    
    # KD-current (mKD=activation variable)
    vh_α_mKD = 55
    vh_β_mKD = 65
    k_α_mKD = 10
    k_β_mKD = 80
    KDshift=10.0
    alpha_mKD(V::Float64,i) = 0.0025*(V+vh_α_mKD[i])/(1. - exp(-(V+vh_α_mKD[i])/k_α_mKD[i]) )
    beta_mKD(V::Float64,i) = 0.03125*exp(-(V+vh_β_mKD[i])/k_β_mKD[i])
    mKD_inf(V::Float64,i) = alpha_mKD(V-KDshift,i) / (alpha_mKD(V-KDshift,i) + beta_mKD(V-KDshift,i))
    tau_mKD(V::Float64,i) = 1 / (alpha_mKD(V-KDshift,i) + beta_mKD(V-KDshift,i))
    
    # H-current (mH=activation variable)
    w_α_mH = 14.59
    w_β_mH = 1.87
    b_α_mH = 0.086
    b_β_mH = 0.0701
    alpha_mH(V::Float64,i)= exp(-w_α_mH[i]-(b_α_mH[i]*V))
    beta_mH(V::Float64,i)= exp(-w_β_mH[i]+(b_β_mH[i]*V))
    mH_inf(V::Float64,i)= alpha_mH(V,i) /(alpha_mH(V,i) + beta_mH(V,i))
    tau_mH(V_taumH,i) = 1/(alpha_mH(V_taumH,i) + beta_mH(V_taumH,i))
    # dmH_inf(V::Float64)=((((0 - (0.086 * 1)) * exp(-14.59 - 0.086*V)) * (exp(-14.59 - 0.086*V) + exp(-1.87 + 0.0701*V)) - exp(-14.59 - 0.086*V) * ((0 - (0.086 * 1)) * exp(-14.59 - 0.086*V) + (0.0701 * 1) * exp(-1.87 + 0.0701*V))) / (exp(-14.59 - 0.086*V) + exp(-1.87 + 0.0701*V)) ^ 2)
    
    # A-current (mA=activation variable, hA=inactivation variable)
    vh_∞_mA = 60
    vh_τ_mA1 = 35.82
    vh_τ_mA2 = 79.69
    k_∞_mA = 8.5
    k_τ_mA1 = 19.697
    k_τ_mA2 = -12.7
    mA_inf(V::Float64,i) = 1/(1+exp(-(V+vh_∞_mA[i])/k_∞_mA[i]))
    tau_mA_temp(V::Float64,i) = 0.37 + 1/(exp((V+vh_τ_mA1[i])/k_τ_mA1[i])+exp((V+vh_τ_mA2[i])/k_τ_mA2[i]))
    tau_mA(V::Float64,i) = tau_mA_temp(V,i)
    
    vh_∞_hA = 78
    vh_τ_hA1 = 46.05
    vh_τ_hA2 = 238.4
    k_∞_hA = 6
    k_τ_hA1 = 5
    k_τ_hA2 = -37.45
    hA_inf_temp(V::Float64,i) = 1/(1+exp((V+vh_∞_hA[i])/k_∞_hA[i]))
    hA_inf(V,i) = hA_inf_temp(V,i)
    function tau_hA(V::Float64,i)
        if V < -63
            tau_hA = 1/(exp((V+vh_τ_hA1[i])/k_τ_hA1[i])+exp((V+vh_τ_hA2[i])/k_τ_hA2[i]))
        else
            tau_hA = 19
        end
        return tau_hA
    end
    #tau_hA(V::Float64)=50.
    
    # T-type Ca-current (mt=activation variable, ht=inactivation variable)
    vh_∞_mt = 57
    vh_τ_mt1 = 131.6
    vh_τ_mt2 = 16.8
    k_∞_mt = 6.2
    k_τ_mt1 = 16.7
    k_τ_mt2 = 18.2
    mt_inf(V::Float64,i) = 1/(1+exp(-(V+vh_∞_mt[i])/k_∞_mt[i]))
    tau_mt(V::Float64,i) = 0.612 + 1/(exp(-(V+vh_τ_mt1[i])/k_τ_mt1[i])+exp((V+vh_τ_mt2[i])/k_τ_mt2[i]))*2
    
    vh_∞_ht = 81
    vh_τ_ht1 = 467
    vh_τ_ht2 = 21.88
    k_∞_ht = 4.03
    k_τ_ht1 = 66.6
    k_τ_ht2 = 10.2
    ht_inf(V::Float64,i) = 1/(1+exp((V+vh_∞_ht[i])/k_∞_ht[i]))
    function tau_ht(V::Float64,i)
        if V < -80
            tau_ht = exp((V+vh_τ_ht1[i])/k_τ_ht1[i])*2
        else
            tau_ht = (exp(-(V+vh_τ_ht2[i])/k_τ_ht2[i])+28)*2
        end
        return tau_ht
    end
    
    # L-type Ca-current (mL=activation variable) (from Drion2011)
    vh_∞_mL = 55
    vh_τ_mL = 45
    k_∞_mL = 3
    k_τ_mL = 400
    mL_inf(V::Float64,i) = 1/(1+exp(-(V+vh_∞_mL[i])/k_∞_mL[i]))
    tau_mL(V::Float64,i) = (72*exp(-(V+vh_τ_mL[i])^2/k_τ_mL[i])+6.)*2
    
    # Intracellular calcium
    ICa_pump(Ca::Float64)=0.1*Ca/(Ca+0.0001)
    
    # HAVENT DONE THIS ONE!
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
    # NB: Have converted this fn to the new neuron model!
    def define_dv_terms(self, to_estimate, est_gsyns, v, ints, syn_gates, I):
        # First deal with intrinsic conductances.
        gs = np.concatenate((self.gs, [1.]))
        terms = calc_terms(v, ints, self.ENa, self.EH, self.ECa, self.EK, self.Eleak, self.c, I)
        
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
    
    # HAVEN'T YET UPDATED FOR NEW MODEL!!
    def calc_dv_no_observer(self, v, m, h, n, syn_gates, I):
        gs = np.array([self.gNa, self.gK, self.gL, 1])
        terms = hhmodel_calc_terms(v, m, h, n, self.ENa, self.EK, self.EL, self.c, I)
        dv = np.dot(gs, terms)
        
        if syn_gates:
            # In numpy, asterisk operator performs elementwise multiplication.
            dv = dv - self.g_syns * syn_gates * (v - self.Esyn) # !! NEED TO DIVIDE BY C??
        return dv        

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
            θ, ϕ, b = calc_dv_terms_final_step_if_est_gsyns(θ_intrins, 
                                self.g_syns, ϕ_intrins, syn_terms, gs, terms)
            return (θ, ϕ, b)
        else:
            b = calc_dv_terms_final_step_if_not_est_gsyns(gs, self.g_syns, terms, syn_terms)
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
def calc_terms(v, ints, ENa, EH, ECa, EK, Eleak, c, I):
    terms = np.divide(np.array([
                -ints[0]**3*ints[1]*(v-ENa), # I_Na
                -ints[2]*(v-EH), # I_H
                -ints[3]**2*ints[4]*(v-ECa), # I_T
                -ints[5]**4*ints[6]*(v-EK), # I_A
                -ints[7]**4*(v-EK), # I_KD
                -ints[8]*(v-ECa), # I_L
                -ints[9]**4*(v-EK), # I_KCa
                -(v-Eleak)
            ]),c)

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
