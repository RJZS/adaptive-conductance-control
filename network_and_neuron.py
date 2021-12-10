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
        self.Esyn = -70 # GABA A from Drion 2018 (see Fig1C, CaT_ncells.jl)
        # self.Esyn = -85 # GABA B
        
        # # Drion plos 18
        # self.ENa = 50.
        # self.EK = -85.
        # self.ECa = 120.
        # self.EH = -20.

        self.Es = np.array([self.ENa, self.EH, self.ECa, self.EK, self.Eleak, self.Esyn]) # Useful.
        
        self.gL_for_Ca=0.4
        
        self.syns = synapses
        self.num_syns = len(synapses)
        
        self.g_syns = np.zeros(self.num_syns)
        for (idx, syn) in enumerate(self.syns):
            self.g_syns[idx] = syn.g
            
        self.pre_neurs = np.zeros(self.num_syns, dtype=np.int8)
        for (idx, syn) in enumerate(self.syns):
            self.pre_neurs[idx] = syn.pre_neur
    
    ## Gating functions. All the functions being used have been moved out of
    ## the class definition!
    
    # # Intracellular calcium. USING THIS?
    # def ICa_pump(self, Ca): 0.1*Ca/(Ca+0.0001)
    
    # # Gate for KCa. Instead of modelling [Ca]. Term is g_{KCa} mCa^4 (V - E_{KCa})
    # # This is from Thiago's HCO2.
    # def gating_mCa(self, v):
    #     vh_inf_mL = 55
    #     vh_τ_mL = 45
    #     k_inf_mL = 3
    #     k_τ_mL = 400
    #     def mCa_inf(V): return 1/(1+np.exp(-(V+vh_inf_mL)/k_inf_mL))
    #     def tau_mCa(V): return (72*np.exp(-(V+vh_τ_mL)**2/k_τ_mL)+6.)*2
    #     τ = tau_mCa(v)
    #     σ = mCa_inf(v)
    #     return τ, σ
    
    
    
    # # Synaptic current
    # def gating_s(self, v):
    #     V12syn=-20.0
    #     ksyn=4.0
    #     def msyn_inf(V): return 1.0 / ( 1.0 + np.exp(-(V-V12syn)/ksyn) )
    #     tausyn = 20.0
    #     τ = tausyn
    #     σ = msyn_inf(v)
    #     return τ, σ
    
    # What's this function for? Not used?
    # def neuron_calcs(self, v, m, h, n, I):
    #     (τm,σm) = self.gating_m(v);
    #     (τh,σh) = self.gating_h(v);
    #     (τn,σn) = self.gating_n(v);
    
    #     g = [self.gNa, self.gK, self.gL]
    #     phi = [-m**3*h*(v-self.ENa),-n**4*(v-self.EK),-(v-self.EL)];
    
    #     dv = 1/self.c * (np.dot(phi,g) + I);
    #     dm = 1/τm*(-m + σm);
    #     dh = 1/τh*(-h + σh);
    #     dn = 1/τn*(-n + σn);
    
    #     return [dv,dm,dh,dn]
    
    # ------ FOR REBOUND BURSTER ------
    # def gating_m(self, v):
    #     Vhalf = -40.;
    #     k = 9.;              #15 in izhikevich
    #     Vmax = -38.;
    #     std = 30.;
    #     Camp = 0.46;
    #     Cbase = 0.04;
    #     (τ, σ) = calc_tau_and_sigma(v, Cbase, Camp, Vmax, std, Vhalf, k)
    #     return τ, σ 
    
    # # Sodium inactivation
    # def gating_h(self, v):
    #     Vhalf = -62.;
    #     k = -7.;
    #     Vmax = -67.;
    #     std = 20.;
    #     Camp = 7.4;
    #     Cbase = 1.2;
    #     (τ, σ) = calc_tau_and_sigma(v, Cbase, Camp, Vmax, std, Vhalf, k)
    #     return τ, σ
    
    # # Potassium activation
    # def gating_mKD(self, v):
    #     Vhalf = -53.;
    #     k = 15.;
    #     Vmax = -79.;
    #     std = 50.;
    #     Camp = 4.7;
    #     Cbase = 1.1;
    #     (τ, σ) = calc_tau_and_sigma(v, Cbase, Camp, Vmax, std, Vhalf, k)
    #     return τ, σ
    
    # # Synaptic gate
    # def gating_s(self, v): # Terms are same as m unless stated.
    #     Vhalf = -45.; # From Dethier et al - 2015
    #     k = 2.; # From Dethier et al - 2015
    #     Vmax = -38.;
    #     std = 30.;
    #     Camp = 0.46;
    #     Cbase = 0.04;
    #     (τ, σ) = calc_tau_and_sigma(v, Cbase, Camp, Vmax, std, Vhalf, k)
    #     return τ, σ   
    
    # # Replace gating variables for mT. This is Ca current from Thiago's 'HCO'.
    # # mCa activation
    # def thiago_gam(v,c,vhalf,sig): return c/(1+np.exp((v+vhalf)/sig))
    # def thiago_tau(v,c1,c2,vhalf,sig): return c1 + c2/(1+np.exp((v+vhalf)/sig))
    
    # # this is HCO_knietics.jl... so what's the one right above?
    # def t_σ(self, v,r,k): 
    #     return 1/(1+np.exp(-(v-r)/k))
        
    # def t_τ(self, v,c1,c2,r,k):
    #     return c1 + c2*self.t_σ(v,r,k)
    
    # def gating_mT(self, v):
    #     rCa_m = -67.1;          
    #     kCa_m = 7.2;
    #     σCa_m = self.t_σ(v,rCa_m,kCa_m)
        
    #     # mCa time constant
    #     c1Ca_m = 43.4;
    #     c2Ca_m = -42.6;
    #     rτCa_m = -68.1;
    #     kτCa_m = 20.5;
    #     τCa_m = self.t_τ(v,c1Ca_m,c2Ca_m,rτCa_m,kτCa_m)
    #     return τCa_m, σCa_m
        
    # def gating_hT(self, v):
    #     rCa_h = -100 # -82.1;
    #     kCa_h = -2.2 # -5.5;
    #     σCa_h = self.t_σ(v,rCa_h,kCa_h)
        
    #     # hCa time constant
    #     c1Ca_h = 40 #140;
    #     c2Ca_h = -35 # -100;
    #     rτCa_h = -55;
    #     kτCa_h = 16.9;
    #     τCa_h = self.t_τ(v,c1Ca_h,c2Ca_h,rτCa_h,kτCa_h)
    #     return τCa_h, σCa_h
    
    # ------ END OF CODE FOR REBOUND BURSTER ------
    
    # ------ DRION PLOS 18 ------
    
    # # gating functions
    # def boltz(self,V,A,B): return 1./(1. + np.exp((V+A)/B))
    # def tauX(self,V,A,B,D,E): return A - B/(1+np.exp((V+D)/E))
    # def mNainf(self,V): return self.boltz(V,35.5,-5.29)
    # def taumNa(self,V): return self.tauX(V,1.32,1.26,120.,-25.)
    # def hNainf(self,V): return self.boltz(V,48.9,5.18)
    # def tauhNa(self,V): return (0.67/(1+np.exp((V+62.9)/-10.0)))*(1.5 + 1/(1+np.exp((V+34.9)/3.6)))
    # def mKdinf(self,V): return self.boltz(V,12.3,-11.8)
    # def taumKd(self,V): return self.tauX(V,7.2,6.4,28.3,-19.2)
    # def mCaTinf(self,V): return self.boltz(V,67.1,-7.2)
    # def taumCaT(self,V): return self.tauX(V,21.7,21.3,68.1,-20.5)
    # def hCaTinf(self,V): return self.boltz(V,80.1,5.5)
    # def tauhCaT(self,V): return 2*self.tauX(V,205.,89.8,55.,-16.9)
    # def mHinf(self,V): return self.boltz(V,80.,6.)
    # def taumH(self,V): return self.tauX(V,272.,-1149.,42.2,-8.73)
    # def mKCainf(self,Ca): return (Ca/(Ca+170))**2
    
    # def gating_m(self,v):
    #     return self.taumNa(v), self.mNainf(v)
    # def gating_h(self,v):
    #     return self.tauhNa(v), self.hNainf(v)
    # def gating_mH(self,v):
    #     return self.taumH(v), self.mHinf(v)
    # def gating_mT(self,v):
    #     return self.taumCaT(v), self.mCaTinf(v)
    # def gating_hT(self,v):
    #     return self.tauhCaT(v), self.hCaTinf(v)
    # def gating_mKD(self,v):
    #     return self.taumKd(v), self.mKdinf(v)
    
    
    
    # dGABAB(V::Float64,GABAB::Float64) = (dt)*(0.016*Tm(V)*(1-GABAB)-0.0047*GABAB)
    
    # ------ END OF CODE FOR DRION PLOS 18 ------
    
    def gate_calcs(self, v, int_gates, syn_gates, v_pres):
        num_vectorized_points = v.shape[-1]
        print("num_vectorized_points: {}".format(num_vectorized_points))
        print("in gate_calcs, vs.shape: {}".format(v.shape))
        dints = np.zeros((self.NUM_GATES, num_vectorized_points))
        (τm,σm) = gating_mNa(v);
        (τh,σh) = gating_hNa(v);
        (τmH,σmH) = gating_mH(v);
        (τmT,σmT) = gating_mT(v);
        (τhT,σhT) = gating_hT(v);
        (τmA,σmA) = gating_mA(v);
        (τhA,σhA) = gating_hA(v);
        (τmKD,σmKD) = gating_mKD(v);
        (τmL,σmL) = gating_mL(v);
        # (τCa,σCa) = self.gating_mCa(v);
        
        taus = np.array([τm, τh, τmH, τmT, τhT, τmA, τhA, τmKD, τmL])
        sigmas = np.array([σm, σh, σmH, σmT, σhT, σmA, σhA, σmKD, σmL])
        dints[:9,:] = calc_dgate(taus, int_gates[:9,:], sigmas)
        
        dCa = (-0.1*self.gL_for_Ca*int_gates[8]*(v-self.ECa)-0.01*int_gates[9])/4
        dints[9] = dCa
        
        dsyns = np.zeros((self.num_syns, num_vectorized_points))
        for (idx, syn) in enumerate(self.syns):
            # (τs,σs) = self.gating_s(v_pres[idx])
            # dsyns[idx] = calc_dgate(τs, syn_gates[idx], σs);
            dsyns[idx] = dGABA_A(v_pres[idx], syn_gates[idx])
            
        return (dints, dsyns)
    
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
            (res_gs, res_terms) = calc_res_gs_and_terms(el_connects, neur_idx, network_Vs, self.c)
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
    
    def calc_dv_no_observer(self, v, ints, syn_gates, I):
        gs = np.concatenate((self.gs, [1.]))
        mKir = mKir_inf(v) # Gate modelled as instantaneous.
        terms = calc_terms(v, ints, mKir, self.ENa, self.EH, self.ECa, self.EK, self.Eleak, self.c, I)
        dv = np.dot(gs, terms)
        
        if syn_gates:
            # In numpy, asterisk operator performs elementwise multiplication.
            dv = dv - np.divide(self.g_syns * syn_gates * (v - self.Esyn),self.c)
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
        
# Needed to take some functions out of the class, for numba.
# @njit(cache=True)
def calc_tau_and_sigma(v, Cbase, Camp, Vmax, std, Vhalf, k):
    τ = Cbase + Camp*np.exp(-np.power((v-Vmax),2)/std**2);
    σ = np.divide(1, (1+np.exp(-(v-Vhalf)/k)));
    return τ, σ

# @njit(cache=True)
def calc_dgate(τ, x, σ):
    print("sigma and tau")
    print(σ.shape)
    print(τ.shape)
    τ = np.reshape(τ, (len(τ), 1)) # Bug was here! But why affecting tau and not sigma??
    dx = np.divide(1,np.multiply(τ,(-x + σ)))
    return dx

# # @njit(cache=True)
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

# # @njit(cache=True)
def calc_res_gs_and_terms(el_connects, neur_idx, Vs, c):
    # Filter to find connections involving our neuron.
    def res_condition(x): 
        f = np.zeros(len(x), dtype=np.bool_)
        for (i, (one,two)) in enumerate(x):
            if one == neur_idx or two == neur_idx:
                f[i] = True
        return f
    res_idxs = el_connects[:,1:]
    
    neur_res_bool = res_condition(res_idxs)
    neur_res_idxs = res_idxs[neur_res_bool]
    neur_res_idxs = neur_res_idxs.astype(np.int8)
    neur_res_gs = el_connects[neur_res_bool,0]
        
    # Now calculate terms.
    terms = np.zeros(len(neur_res_idxs))
    for (i, res_pair) in enumerate(neur_res_idxs):
        other_neur_idx = np.extract(res_pair != neur_idx, res_pair)[0]
        terms[i] = - neur_res_gs[i] * (Vs[neur_idx] - Vs[other_neur_idx])
    terms = np.divide(terms, c)
    
    return (neur_res_gs, terms)

# # @njit(cache=True)
def calc_terms(v, ints, mKir, ENa, EH, ECa, EK, Eleak, c, I):
    print("calc_terms")
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
    print(terms.shape)
    return terms

# @njit(cache=True)
def hhmodel_calc_terms(v, m, h, n, ENa, EK, EL, c, I):
    terms = np.divide(np.array([-m**3*h*(v-ENa),-n**4*(v-EK),
                                    -(v-EL),I]),c)
    return terms

# Providing types as 'typeof' was taking a long time in the profiler.
# # @njit((f8[:],f8[:],i4[:]), cache=True)
def calc_intrins_dv_terms(gs, terms, to_estimate):
    # First deal with intrinsic conductances.
    num_pts = terms.shape[-1]
    θ_intrins = np.zeros((len(to_estimate), num_pts))
    ϕ_intrins = np.zeros((len(to_estimate), num_pts))
    
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

# # @njit(cache=True)
def calc_dv_terms_final_step_if_est_gsyns_gels(θ_intrins, g_syns, ϕ_intrins, syn_terms, gs, terms, g_res, res_terms):
    print("calc_dv_terms_final_step_if_est_ shapes: {}, {}, {}".format(θ_intrins.shape, g_syns.shape, g_res.shape))
    num_pts = θ_intrins.shape[-1]
    g_syns = np.reshape(g_syns, (len(g_res),num_pts))
    g_res = np.reshape(g_res, (len(g_res),num_pts))
    syn_terms = np.reshape(syn_terms, (len(g_res),num_pts))
    res_terms = np.reshape(res_terms, (len(g_res),num_pts))
    θ = np.concatenate((θ_intrins, g_syns, g_res))
    ϕ = np.concatenate((ϕ_intrins, syn_terms, res_terms))
    b = np.dot(gs, terms)
    return (θ, ϕ, b)

# @njit(cache=True)
def calc_dv_terms_final_step_if_not_est_gsyns_gels(gs, g_syns, terms, syn_terms, g_res, res_terms):
    b = np.dot(
                    np.concatenate((gs, g_syns, g_res)),
                    np.concatenate((terms, syn_terms, res_terms))
                )
    return b

## Gating Functions for full model
# Na-current (m=activation variable, h=inactivation variable)
# @njit(cache=True)
def gating_mNa(v):
    def alpha_m(V): return -0.025*(V+40.)/( np.exp(-(V+40)/10) - 1.0 )
    def beta_m(V): return np.exp(-(V+65)/18)
    def mNa_inf(V): return alpha_m(V) / (alpha_m(V) + beta_m(V)) # Activation function
    def tau_mNa(V): return 1. / (alpha_m(V) + beta_m(V))/5 # Activation time-constant 
    τ = tau_mNa(v)
    σ = mNa_inf(v)
    return τ, σ
    
# @njit(cache=True)
def gating_hNa(v):
    def alpha_h(V): return 0.0175*np.exp(-(V+65)/20)
    def beta_h(V): return 0.25/(1.0 + np.exp(-(V+35)/10) )
    def hNa_inf(V): return alpha_h(V) / (alpha_h(V) + beta_h(V)) # Inactivation function
    def tau_hNa(V): return 1 / (alpha_h(V) + beta_h(V))/5 # Inactivation time-constant
    τ = tau_hNa(v)
    σ = hNa_inf(v)
    return τ, σ

# KD-current (mKD=activation variable)
# @njit(cache=True)
def gating_mKD(v):
    Kdshift=10.0
    def alpha_mKd(V): return 0.0025*(V+55.)/(1. - np.exp(-(V+55.)/10.) )
    def beta_mKd(V): return 0.03125*np.exp(-(V+65.)/80.)
    def mKd_inf(V): return alpha_mKd(V-Kdshift) / (alpha_mKd(V-Kdshift) + beta_mKd(V-Kdshift)) # Activation function
    def tau_mKd(V): return 1. / (alpha_mKd(V-Kdshift) + beta_mKd(V-Kdshift))/5 # Activation time-constant
    τ = tau_mKd(v)
    σ = mKd_inf(v)
    return τ, σ

# H-current (mH=activation variable)
# @njit(cache=True)
def gating_mH(v):
    def alpha_mH(V): return np.exp(-14.59-(0.086*V))
    def beta_mH(V): return np.exp(-1.87+(0.0701*V))
    def mH_inf(V): return alpha_mH(V) /(alpha_mH(V) + beta_mH(V)) # Activation function
    def tau_mH(V): return 1./(alpha_mH(V) + beta_mH(V)) # Activation time-constant
    τ = tau_mH(v)
    σ = mH_inf(v)
    return τ, σ

# A-current (mA=activation variable, hA=inactivation variable)
# @njit(cache=True)
def gating_mA(v):
    def mA_inf(V): return 1/(1+np.exp(-(V+90)/8.5)) # Activation function
    def tau_mA(V): return 0.37 + 1/(np.exp((V+35.82)/19.697)+np.exp((V+79.69)/-12.7))/5 # Activation time-constant
    τ = tau_mA(v)
    σ = mA_inf(v)
    return τ, σ

# @njit(cache=True)
def gating_hA(v):
    def hA_inf(V): return 1/(1+np.exp((V+78)/6)) # Inactivation function
    def tau_hA(V): # Inactivation time-constant
        if V < -63:
            tau_hA = 1/(np.exp((V+46.05)/5)+np.exp((V+238.4)/-37.45))
        else:
            tau_hA = 19
        return tau_hA/5
    τ = tau_hA(v)
    σ = hA_inf(v)
    return τ, σ

# T-type Ca-current (mt=activation variable, ht=inactivation variable)
# @njit(cache=True)
def gating_mT(v):
    def mT_inf(V): return 1/(1+np.exp(-(V+57)/6.2)) # Activation function
    def tau_mT(V): return 0.612 + 1/(np.exp(-(V+131.6)/16.7)+np.exp((V+16.8)/18.2)) # Activation time-constant
    τ = tau_mT(v)
    σ = mT_inf(v)
    return τ, σ
    
# @njit(cache=True)
def gating_hT(v):
    def hT_inf(V): return 1/(1+np.exp((V+81)/4.03)) # Inactivation function
    def tau_hT(V):
        if V < -80:
            tau_ht = np.exp((V+467)/66.6)
        else:
            tau_ht = (np.exp(-(V+21.88)/10.2)+28)
        return tau_ht
    τ = tau_hT(v)
    σ = hT_inf(v)
    return τ, σ

# L-type Ca-current (mL=activation variable) (from Drion2011)
# @njit(cache=True)
def gating_mL(v):
    def mL_inf(V): return 1/(1+np.exp(-(V+55.)/3)) # Activation function
    def tau_mL(V): return (72*np.exp(-(V+45.)**2/400)+6.) # Activation time-constant
    τ = tau_mL(v)
    σ = mL_inf(v)
    return τ, σ

# Kir-current (mKIR=activation variable). Modelled as instantaneous.
# @njit(cache=True)
def mKir_inf(V):
    σ = 1/(1+np.exp((V+97.9+10)/9.7)) # Activation function
    return σ

# Synapse
# @njit(cache=True)
def dGABA_A(v, s):
    def Tm(V): return 1/(1+np.exp(-(V-2)/5))
    ds = 0.53*Tm(v)*(1-s)-0.18*s # In paper it's -0.19*s
    return ds
