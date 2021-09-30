# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 09:28:07 2021

@author: Rafi
"""
# Based on Thiago's Julia code.
import numpy as np

# Sodium activation
def gating_m(v):
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
def gating_h(v):
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
def gating_n(v):
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
def gating_s(v): # NEED TO SET PARAMS!!!
    Vhalf = 0.;
    k = 9.;
    Vmax = -38.;
    std = 30.;
    Camp = 0.46;
    Cbase = 0.04;
    τ = Cbase + Camp*np.exp(-np.power((v-Vmax),2)/std**2);
    σ = np.divide(1, (1+np.exp(-(v-Vhalf)/k)));
    return τ, σ

def HH_ode(t,z,p):
    Iapp =          p[0]
    c =             p[1]
    (gNa,gK,gL) =   p[2]
    (ENa,EK,EL) =   p[3]

    v = z[0]
    m = z[1]
    h = z[2]
    n = z[3]

    (τm,σm) = gating_m(v);
    (τh,σh) = gating_h(v);
    (τn,σn) = gating_n(v);

    g = [gNa, gK, gL];
    phi = [-m**3*h*(v-ENa),-n**4*(v-EK),-(v-EL)];

    dv = 1/c * (np.dot(phi,g) + Iapp(t));
    dm = 1/τm*(-m + σm);
    dh = 1/τh*(-h + σh);
    dn = 1/τn*(-n + σn);

    return [dv,dm,dh,dn]

def HH_synapse_observer(t,z,p):
    Iapp =               p[0]
    c =                  p[1]
    (gNa,gK,gL,gsyn) =   p[2]
    (ENa,EK,EL,Esyn) =   p[3]
    (α,γ) = p[4]

    # True system
    v = z[0]
    m = z[1]
    h = z[2]
    n = z[3]
    s = z[4]

    (τm,σm) = gating_m(v);
    (τh,σh) = gating_h(v);
    (τn,σn) = gating_n(v);
    (τs,σs) = gating_s(v);

    # θ = np.divide(1,c*np.array([gNa, gK, gL, gNa*ENa, gK*EK, gL*EL, 1]))
    θ = np.divide(np.array([gNa, gK, gL, gsyn,
                            gNa*ENa, gK*EK, gL*EL, gsyn*Esyn, 1]),c)
    ϕ = np.array([-m**3*h*v,
         -n**4*v, 
         -v,
         -s*v,
         m**3*h,
         n**4,
         1,
         s,
         Iapp(t)]);

    dv = np.dot(ϕ,θ)
    dm = 1/τm*(-m + σm);
    dh = 1/τh*(-h + σh);
    dn = 1/τn*(-n + σn);
    ds = 1/τs*(-s + σs);

    # Adaptive observer
    v̂ = z[4]
    m̂ = z[5]
    ĥ = z[6]
    n̂ = z[7]
    s_hat = z[8]
    θ̂ = z[9:18]
    P = np.reshape(z[18:18+81],(9,9));    
    P = (P+np.transpose(P))/2
    Ψ = z[18+81:18+81+9]

    (τm̂,σm̂) = gating_m(v);
    (τĥ,σĥ) = gating_h(v);
    (τn̂,σn̂) = gating_n(v);
    (τs_hat,σs_hat) = gating_s(v);

    ϕ̂ = np.array([-m̂**3*ĥ*v,
         -n̂**4*v, 
         -v,
         -s_hat*v,
         m̂**3*ĥ,
         n̂**4,
         1,
         s,
         Iapp(t)]);

    dv̂ = np.dot(ϕ̂,θ̂) + γ*Ψ@P@Ψ.T*(v-v̂)
    dm̂ = 1/τm̂*(-m̂ + σm̂);
    dĥ = 1/τĥ*(-ĥ + σĥ);
    dn̂ = 1/τn̂*(-n̂ + σn̂);
    ds_hat = 1/τs_hat*(-s_hat + σs_hat);

    dθ̂ = γ*P@Ψ.T*(v-v̂);
    dΨ = -γ*Ψ + ϕ̂; 
    aux = np.outer(Ψ,Ψ)
    dP = α*P - P@aux@P;
    dP = (dP+np.transpose(dP))/2;
    
    # dz[:] = [dv;dm;dh;dn;dv̂;dm̂;dĥ;dn̂;dθ̂;dP[:];dΨ;dṽ;dm̃;dh̃;dñ;dθ̃]';
    ## BUT z_0 only has length 82...
    dz = np.concatenate(( [dv,dm,dh,dn,ds],[dv̂,dm̂,dĥ,dn̂,ds_hat],dθ̂.flatten(),
                         dP.flatten(),dΨ))
    return dz    

def HH_one_observer(t,z,p):
    Iapp =          p[0]
    c =             p[1]
    (gNa,gK,gL) =   p[2]
    (ENa,EK,EL) =   p[3]
    (α,γ) = p[4]

    # True system
    v = z[0]
    m = z[1]
    h = z[2]
    n = z[3]

    (τm,σm) = gating_m(v);
    (τh,σh) = gating_h(v);
    (τn,σn) = gating_n(v);

    # θ = np.divide(1,c*np.array([gNa, gK, gL, gNa*ENa, gK*EK, gL*EL, 1]))
    θ = np.divide(np.array([gNa, gK, gL, gNa*ENa, gK*EK, gL*EL, 1]),c)
    ϕ = np.array([-m**3*h*v,
         -n**4*v, 
         -v,
         m**3*h,
         n**4,
         1,
         Iapp(t)]);

    dv = np.dot(ϕ,θ)
    dm = 1/τm*(-m + σm);
    dh = 1/τh*(-h + σh);
    dn = 1/τn*(-n + σn);

    # Adaptive observer
    v̂ = z[4]
    m̂ = z[5]
    ĥ = z[6]
    n̂ = z[7]
    θ̂ = z[8:15]
    P = np.reshape(z[15:15+49],(7,7));    
    P = (P+np.transpose(P))/2
    Ψ = z[15+49:15+49+7]

    (τm̂,σm̂) = gating_m(v);
    (τĥ,σĥ) = gating_h(v);
    (τn̂,σn̂) = gating_n(v);

    ϕ̂ = np.array([-m̂**3*ĥ*v,
         -n̂**4*v, 
         -v,
         m̂**3*ĥ,
         n̂**4,
         1,
         Iapp(t)]);

    dv̂ = np.dot(ϕ̂,θ̂) + γ*Ψ@P@Ψ.T*(v-v̂)
    dm̂ = 1/τm̂*(-m̂ + σm̂);
    dĥ = 1/τĥ*(-ĥ + σĥ);
    dn̂ = 1/τn̂*(-n̂ + σn̂);

    dθ̂ = γ*P@Ψ.T*(v-v̂);
    dΨ = -γ*Ψ + ϕ̂; 
    aux = np.outer(Ψ,Ψ)
    dP = α*P - P@aux@P;
    dP = (dP+np.transpose(dP))/2;
    
    # dz[:] = [dv;dm;dh;dn;dv̂;dm̂;dĥ;dn̂;dθ̂;dP[:];dΨ;dṽ;dm̃;dh̃;dñ;dθ̃]';
    ## BUT z_0 only has length 82...
    dz = np.concatenate(( [dv,dm,dh,dn],[dv̂,dm̂,dĥ,dn̂],dθ̂.flatten(),
                         dP.flatten(),dΨ))
    return dz

def HH_observer(t,z,p):
    Iapp =          p[0]
    c =             p[1]
    (gNa,gK,gL) =   p[2]
    (ENa,EK,EL) =   p[3]
    (α,γ) = p[4]

    # True system
    v = z[0]
    m = z[1]
    h = z[2]
    n = z[3]

    (τm,σm) = gating_m(v);
    (τh,σh) = gating_h(v);
    (τn,σn) = gating_n(v);

    # θ = np.divide(1,c*np.array([gNa, gK, gL, gNa*ENa, gK*EK, gL*EL, 1]))
    θ = np.divide(np.array([gNa, gK, gL, gNa*ENa, gK*EK, gL*EL, 1]),c)
    ϕ = np.array([-m**3*h*v,
         -n**4*v, 
         -v,
         m**3*h,
         n**4,
         1,
         Iapp(t)]);

    dv = np.dot(ϕ,θ)
    dm = 1/τm*(-m + σm);
    dh = 1/τh*(-h + σh);
    dn = 1/τn*(-n + σn);

    # Adaptive observer
    v̂ = z[4]
    m̂ = z[5]
    ĥ = z[6]
    n̂ = z[7]
    θ̂ = z[8:15]
    P = np.reshape(z[15:15+49],(7,7));    
    P = (P+np.transpose(P))/2
    Ψ = z[15+49:15+49+7]

    (τm̂,σm̂) = gating_m(v);
    (τĥ,σĥ) = gating_h(v);
    (τn̂,σn̂) = gating_n(v);

    ϕ̂ = np.array([-m̂**3*ĥ*v,
         -n̂**4*v, 
         -v,
         m̂**3*ĥ,
         n̂**4,
         1,
         Iapp(t)]);

    dv̂ = np.dot(ϕ̂,θ̂) + γ*Ψ@P@Ψ.T*(v-v̂)
    dm̂ = 1/τm̂*(-m̂ + σm̂);
    dĥ = 1/τĥ*(-ĥ + σĥ);
    dn̂ = 1/τn̂*(-n̂ + σn̂);

    dθ̂ = γ*P@Ψ.T*(v-v̂);
    dΨ = -γ*Ψ + ϕ̂; 
    aux = np.outer(Ψ,Ψ)
    dP = α*P - P@aux@P;
    dP = (dP+np.transpose(dP))/2;

    # Besancon 2000 observer
    N = 15+49+7;

    ṽ = z[N]
    m̃ = z[N+1]
    h̃ = z[N+2]
    ñ = z[N+3]
    θ̃ = z[N+4:N+11]

    ϕ̃ = np.array([-m̃**3*h̃*v,
         -ñ**4*v, 
         -v,
         m̃**3*h̃,
         ñ**4,
         1,
         Iapp(t)]);
            
    (τm̃,σm̃) = gating_m(v);
    (τh̃,σh̃) = gating_h(v);
    (τñ,σñ) = gating_n(v);
    
    dṽ = np.dot(ϕ̃,θ̃) + γ*(v-ṽ);
    dm̃ = 1/τm̃*(-m̃ + σm̃);
    dh̃ = 1/τh̃*(-h̃ + σh̃);
    dñ = 1/τñ*(-ñ + σñ);
    dθ̃ = γ*ϕ̃*(v-ṽ);
    
    # dz[:] = [dv;dm;dh;dn;dv̂;dm̂;dĥ;dn̂;dθ̂;dP[:];dΨ;dṽ;dm̃;dh̃;dñ;dθ̃]';
    ## BUT z_0 only has length 82...
    dz = np.concatenate(( [dv,dm,dh,dn],[dv̂,dm̂,dĥ,dn̂],dθ̂.flatten(),
                         dP.flatten(),dΨ,[dṽ,dm̃,dh̃,dñ],dθ̃))
    return dz

### REMOVE!
# function HH_loss_ode!(dz,z,p,t)
#     # The only parameter here is ĝNa
#     # This makes it simple to compute the gradient of the loss function
#     # Fixed parameters
#     c =             1.
#     (gNa,gK,gL) =   (120.,36.,0.3)
#     (ENa,EK,EL) =   (55.,-77.,-54.4)

#     # Variable parameter
#     ĝNa = p

#     v = z[1]
#     m = z[2]
#     h = z[3]
#     n = z[4]
#     v̂ = z[5]
#     m̂ = z[6]
#     ĥ = z[7]
#     n̂ = z[8]

#     # True model
#     (τm,σm) = gating_m(v);
#     (τh,σh) = gating_h(v);
#     (τn,σn) = gating_n(v);

#     μ = [gNa; gK; gL];
#     phi = [-m^3*h*(v-ENa);-n^4*(v-EK);-(v-EL)];

#     dv = 1/c * (np.dot(phi,g) + Iapp(t));
#     dm = 1/τm*(-m + σm);
#     dh = 1/τh*(-h + σh);
#     dn = 1/τn*(-n + σn);

#     # Estimator model
#     (τm̂,σm̂) = gating_m(v̂);
#     (τĥ,σĥ) = gating_h(v̂);
#     (τn̂,σn̂) = gating_n(v̂);

#     ĝ = [ĝNa; gK; gL];
#     pĥi = [-m̂^3*ĥ*(v̂-ENa);-n̂^4*(v̂-EK);-(v̂-EL)];
#     dv̂ = 1/c * (np.dot(pĥi,ĝ) + Iapp(t));
#     dm̂ = 1/τm̂*(-m̂ + σm̂);
#     dĥ = 1/τĥ*(-ĥ + σĥ);
#     dn̂ = 1/τn̂*(-n̂ + σn̂);

#     dz[:] = [dv,dm,dh,dn,dv̂,dm̂,dĥ,dn̂]
# end

