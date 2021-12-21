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
    data = np.load("downloaded_data/exp1f.npz")
    t=data['t']
    sol=data['sol']
    
    # plt.plot(t[ps_idx:],sol[0,ps_idx:],t[ps_idx:],sol_ref[0,:])
    
    v = sol[0,:]
    v_ref = sol[142,:]
    g_ests = sol[22:31,:]
    error = v - v_ref
    
    # For observer performance plots. 
    skp_obs = 10 # Skip every 'skp'th index. As lualatex struggles with the memory requirements.
    t_obs = t[0::skp_obs]; g_ests = g_ests[:,0::skp_obs]
    
    # For controller performance plots.
    skp_ctrl = 5 
    t_ctrl = t[0::skp_ctrl]; v = v[0::skp_ctrl]; v_ref = v_ref[0::skp_ctrl]
    error = error[0::skp_ctrl]
    
    exp1_control_data = np.vstack((t_ctrl, v, v_ref, error)).T
    exp1_observe_data = np.vstack((t_obs, g_ests)).T
    np.savetxt("../reports/ifac-data/exp1_controller_performance.txt",exp1_control_data,delimiter=' ')
    np.savetxt("../reports/ifac-data/exp1_observer_performance.txt",exp1_observe_data,delimiter=' ')
    
prep_exp2 = False
if prep_exp2:
    data = np.load("downloaded_data/exp2f.npz")
    t=data['t']
    t_nodist=data['tnd']
    sol=data['sol']
    sol_nodist=data['solnd']
    ps=data['ps']
    ps_idx=data['ps_idx']
    
    # diff = sol[0,:ps_idx] - sol_nodist[0,:]
    # plt.plot(t[-ps_idx:],sol[0,:ps_idx],t_nodist,sol_nodist[0,:],
    #           t_nodist, diff)
    t_psd = t[ps_idx:] # Phase shifted
    sol_psd = sol[:,ps_idx:] # Phase shifted
    
    v = sol_psd[0,:]
    v_nd = sol_nodist[0,:]
    gsyn_hat = sol[24,:]
    error = v - v_nd

    syn_g = 2.; Esyn = -70; # Saving these parameters could be automated.
    Id = syn_g * sol[11,:] * (sol[0,:] - Esyn)
    Id_hat = gsyn_hat * sol[23,:] * (sol[0,:] - Esyn)
    
    # trnc = 75000 # As observer converges so quickly, can truncate time-axis.
    # t_trunc = t[:trnc]; g_ests = g_ests[:,:trnc]
    # Id = Id[:trnc]; Id_hat = Id_hat[:trnc]
    
    skp = 10 # Skip every 'skp'th index. As lualatex struggles with the memory requirements.
    t_psd = t_psd[0::skp]; v = v[0::skp]; v_nd = v_nd[0::skp]; error = error[0::skp]
    t = t[0::skp]; gsyn_hat = gsyn_hat[0::skp]; Id = Id[0::skp]; Id_hat = Id_hat[0::skp]
    
    exp2_control_data = np.vstack((t_psd, v, v_nd, error)).T
    exp2_observe_data = np.vstack((t, gsyn_hat, Id, Id_hat)).T
    
    np.savetxt("../reports/ifac-data/exp2_controller_performance.txt",exp2_control_data,delimiter=' ')
    np.savetxt("../reports/ifac-data/exp2_observer_performance.txt",exp2_observe_data,delimiter=' ')
    
prep_exp3 = True
if prep_exp3:
    # Note how controller plots use t_psd, observer plots use t.
    data = np.load("downloaded_data/exp3f.npz")
    t=data['t']
    t_nodist=data['tnd']
    sol=data['sol']
    sol_nodist=data['solnd']
    ps=data['ps']
    ps_idx=data['ps_idx']
    
    t_psd = t[ps_idx:] # Phase shifted
    sol_psd = sol[:,ps_idx:] # Phase shifted
    
    roll_idx = 85 # Have to shift 'nodist' back a little as phase shift code
    # in the simulation script didn't quite work for some reason.
    t_psd = t_psd[:-roll_idx]
    
    v = sol_psd[100,:-roll_idx] # Of hub neuron
    v_nd = np.roll(sol_nodist[0,:], -roll_idx)[:-roll_idx]
    error = v - v_nd
    
    g_ests = sol[126:130,:]
    
    trnc = 1200000 # For controller plots
    t_trnc = t_psd[:trnc]; v_trnc = v[:trnc]; v_nd_trnc = v_nd[:trnc] # Note t_psd
    error_trnc = error[:trnc]
    
    trnc2 = 400000 # For observer plots
    t_trnc2 = t[:trnc2]; g_ests_trnc = g_ests[:,:trnc2]
    
    skp = 10 # Skip every 'skp'th index. As lualatex struggles with the memory requirements.
    t_trnc = t_trnc[0::skp]; v_trnc = v_trnc[0::skp]; v_nd_trnc = v_nd_trnc[0::skp]
    error_trnc = error_trnc[0::skp]
    
    skp2 = 5
    t_trnc2 = t_trnc2[0::skp2]; g_ests_trnc = g_ests_trnc[:,0::skp2]
    
    exp3_control_data = np.vstack((t_trnc, v_trnc, v_nd_trnc, error_trnc)).T
    exp3_observe_data = np.vstack((t_trnc2, g_ests_trnc)).T
    
    np.savetxt("../reports/ifac-data/exp3_controller_performance.txt",exp3_control_data,delimiter=' ')
    np.savetxt("../reports/ifac-data/exp3_observer_performance.txt",exp3_observe_data,delimiter=' ')
    
    