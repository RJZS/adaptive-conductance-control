# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 16:21:18 2021

@author: Rafi
"""
import numpy as np

# Note: in the DistRej experiments I'm applying the phase shift to the 
# 'controller plots', but not to the 'observer plots'.

prep_exp1 = False
if prep_exp1:
    data = np.load("downloaded_data/exp1fff.npz")
    t=data['t']; t_ref=data['t_ref']
    sol=data['sol']; sol_ref=data['sol_ref']
    
    tc = np.concatenate((t_ref[:-1], t+t_ref[-1]))
    vc = np.concatenate((sol_ref[0,:-1], sol[0,:]))
    v2c = np.concatenate((sol_ref[11,:-1], sol[102,:]))
    # plt.plot(t[ps_idx:],sol[0,ps_idx:],t[ps_idx:],sol_ref[0,:])
    
    t = t+t_ref[-1] # Need one consistent t in the plots.
    
    v = sol[0,:]
    v_hat = sol[11,:]
    
    v_ref = sol[102,:]
    v_ref_hat = sol[102+11,:]
    # v = sol[0,:]
    # v_ref = sol[142,:]
    # g_ests = sol[22:31,:]
    # error = v - v_ref
    
    # Can truncate last 2000 seconds.
    
    skp = 10
    tc = tc[0::skp]; vc = vc[0::skp]; v2c = v2c[0::skp]
    t = t[0::skp]
    v = v[0::skp]; v_hat = v_hat[0::skp]; v_ref = v_ref[0::skp]; v_ref_hat = v_ref_hat[0::skp]
    error = v - v_hat
    error_ref = v_ref - v_ref_hat
    
    exp1_control_data = np.vstack((tc, vc, v2c)).T
    exp1_observe_data = np.vstack((t, v, v_hat, v_ref, v_ref_hat, error, error_ref)).T
    np.savetxt("../reports/ifac-data/exp1_controller_performance.txt",exp1_control_data,delimiter=' ')
    np.savetxt("../reports/ifac-data/exp1_observer_performance.txt",exp1_observe_data,delimiter=' ')
    
prep_exp2 = False
if prep_exp2:
    data = np.load("downloaded_data/exp2f.npz")
    tbef = data['tbef']; t=data['t']; tnd=data['tnd']
    solbef = data['solbef']; sol=data['sol']; solnd=data['solnd']
    t = t[:1500000] # Truncate
    sol = sol[:,:1500000]
    # diff = sol[0,:ps_idx] - sol_nodist[0,:]
    # plt.plot(t[-ps_idx:],sol[0,:ps_idx],t_nodist,sol_nodist[0,:],
    #           t_nodist, diff)
    # t_psd = t[ps_idx:] # Phase shifted
    # sol_psd = sol[:,ps_idx:] # Phase shifted
    
    t_control = t + tnd[-1] + tbef[-1]
    tc = np.concatenate((tnd[:-1], tbef[:-1] + tnd[-1], t_control))
    vc = np.concatenate((solnd[0,:-1], solbef[0,:-1], sol[0,:]))
    
    tdc = np.concatenate((tbef[:-1] + tnd[-1], t_control))
    vdc = np.concatenate((solbef[12,:-1], sol[27,:]))
    
    gsyn_hat = sol[24,:]

    syn_g = 0.8; Esyn = -90; # Saving these parameters could be automated.
    Id = syn_g * sol[11,:] * (sol[0,:] - Esyn)
    Id_hat = gsyn_hat * sol[23,:] * (sol[12,:] - Esyn)
    error = Id - Id_hat
    
    # For the 'zoomed in' error plot.
    j=50000;k=1000000 # NB: this cuts out a large initial transient! Down to -1323...
    t_trunc = t_control[j:k]; error_trunc = error[j:k]
    # trnc = 75000 # As observer converges so quickly, can truncate time-axis.
    # t_trunc = t[:trnc]; g_ests = g_ests[:,:trnc]
    # Id = Id[:trnc]; Id_hat = Id_hat[:trnc]
    
    skp = 10 # Skip every 'skp'th index. As lualatex struggles with the memory requirements.
    t_control = t_control[0::skp]; tc = tc[0::skp]; vc = vc[0::skp]
    vdc = vdc[0::skp]; Id = Id[0::skp]; Id_hat = Id_hat[0::skp]
    gsyn_hat = gsyn_hat[0::skp]
    error = error[0::skp]; t_trunc = t_trunc[0::skp]; error_trunc = error_trunc[0::skp]
    tnd = tnd[0::skp]
    
    exp2_control_data = np.vstack((tc, vc, vdc)).T
    exp2_observe_data = np.vstack((t_control, gsyn_hat, Id, Id_hat, error)).T
    exp2_observe_trunc_data = np.vstack((t_trunc, error_trunc)).T
    
    np.savetxt("../reports/ifac-data/exp2_controller_performance.txt",exp2_control_data,delimiter=' ')
    np.savetxt("../reports/ifac-data/exp2_observer_performance.txt",exp2_observe_data,delimiter=' ')
    np.savetxt("../reports/ifac-data/exp2_observer_trunc_performance.txt",exp2_observe_trunc_data,delimiter=' ')
    
prep_exp3 = False
if prep_exp3:
    # Note how controller plots use t_psd, observer plots use t.
    data = np.load("downloaded_data/exp3f.npz")
    tbef = data['tbef']; t=data['t']; tnd=data['tnd']
    solbef = data['solbef']; sol=data['sol']; solnd=data['solnd']
    t = t[:450000] # Truncate
    sol = sol[:,:450000]
    
    t_control = t + tbef[-1]
    tc = np.concatenate((tbef[:-1], t_control))
    vc1 = np.concatenate((solbef[0,:-1], sol[0,:]))
    vc2 = np.concatenate((solbef[13,:-1], sol[50,:]))
    vc3 = np.concatenate((solbef[26,:-1], sol[100,:]))
    vc4 = np.concatenate((solbef[39,:-1], sol[150,:]))
    vc5 = np.concatenate((solbef[52,:-1], sol[200,:]))
    
    vnd1 = solnd[0,:]
    vnd2 = solnd[12,:]
    vnd3 = solnd[24,:]
    
    g_ests = sol[126:130,:]
    
    # trnc = 1200000 # For controller plots
    # t_trnc = t_psd[:trnc]; v_trnc = v[:trnc]; v_nd_trnc = v_nd[:trnc] # Note t_psd
    # error_trnc = error[:trnc]
    
    # trnc2 = 400000 # For observer plots
    # t_trnc2 = t[:trnc2]; g_ests_trnc = g_ests[:,:trnc2]
    
    skp = 10 # Skip every 'skp'th index. As lualatex struggles with the memory requirements.
    t_control = t_control[0::skp]; tc = tc[0::skp]; tnd = tnd[0::skp]
    vc1 = vc1[0::skp]; vc2 = vc2[0::skp]; vc3 = vc3[0::skp]; vc4 = vc4[0::skp]
    vc5 = vc5[0::skp]; vnd1 = vnd1[0::skp]; vnd2 = vnd2[0::skp]
    vnd3 = vnd3[0::skp]; g_ests = g_ests[:,0::skp]
    
    exp3_control_data = np.vstack((tc, vc1, vc2, vc3, vc4, vc5,
                                   tnd, vnd1, vnd2, vnd3)).T
    exp3_observe_data = np.vstack((t_control, g_ests)).T
    
    np.savetxt("../reports/ifac-data/exp3_controller_performance.txt",exp3_control_data,delimiter=' ')
    np.savetxt("../reports/ifac-data/exp3_observer_performance.txt",exp3_observe_data,delimiter=' ')
    
    