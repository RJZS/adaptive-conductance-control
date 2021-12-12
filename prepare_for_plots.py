# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 16:21:18 2021

@author: Rafi
"""
import numpy as np

prep_exp1 = True
if prep_exp1:
    data = np.load("simulate_experiment_1.npz")
    t=data['t']
    t_ref=data['t_ref']
    sol=data['sol']
    sol_ref=data['sol_ref']
    ps=data['ps']
    ps_idx=data['ps_idx']
    
    # plt.plot(t[ps_idx:],sol[0,ps_idx:],t[ps_idx:],sol_ref[0,:])
    
    t = t[ps_idx:]
    sol = sol[:,ps_idx:]
    
    v = sol[0,:]
    v_ref = sol_ref[0,:]
    g_ests = sol[22:25,:]
    error = sol[0,:] - sol_ref[0,:]
    controller_start_time = 20. # The saving of this setting is currently not automated.
    
    trnc = 5000 # As observer converges so quickly, can truncate time-axis.
    t_trunc = t[:trnc]; g_ests = g_ests[:,:trnc]
    
    skp = 5 # Skip every 'skp'th index. As lualatex struggles with the memory requirements.
    t = t[0::skp]; v = v[0::skp]; v_ref = v_ref[0::skp]; error = error[0::skp]
    
    exp1_control_data = np.vstack((t, v, v_ref, error)).T
    exp1_observe_data = np.vstack((t_trunc, g_ests)).T
    np.savetxt("../reports/ifac-data/exp1_controller_performance.txt",exp1_control_data,delimiter=' ')
    np.savetxt("../reports/ifac-data/exp1_observer_performance.txt",exp1_observe_data,delimiter=' ')
    
prep_exp2 = False
if prep_exp2:
    data = np.load("simulate_experiment_2.npz")
    t=data['t']
    t_nodist=data['t_nodist']
    sol=data['sol']
    sol_nodist=data['sol_nodist']
    ps=data['ps']
    ps_idx=data['ps_idx']
    
    # diff = sol[0,:ps_idx] - sol_nodist[0,:]
    # plt.plot(t[-ps_idx:],sol[0,:ps_idx],t_nodist,sol_nodist[0,:],
    #           t_nodist, diff)
    t = t_nodist
    sol = sol[:,:ps_idx]
    
    v = sol[0,:]
    v_nd = sol_nodist[0,:]
    g_ests = sol[24:27,:]
    error = sol[0,:] - sol_nodist[0,:]
    controller_start_time = 2000. # The saving of this setting is currently not automated.

    syn_g = 2.; Esyn = -70; # Also could be automated.
    Id = syn_g * sol[11,:] * (v - Esyn)
    Id_hat = sol[26,:] * sol[23,:] * (v - Esyn)
    
    trnc = 75000 # As observer converges so quickly, can truncate time-axis.
    t_trunc = t[:trnc]; g_ests = g_ests[:,:trnc]
    Id = Id[:trnc]; Id_hat = Id_hat[:trnc]
    
    skp = 2 # Skip every 'skp'th index. As lualatex struggles with the memory requirements.
    t = t[0::skp]; v = v[0::skp]; v_nd = v_nd[0::skp]; error = error[0::skp]
    
    exp2_control_data = np.vstack((t, v, v_nd, error)).T
    exp2_observe_data = np.vstack((t_trunc, g_ests, Id, Id_hat)).T
    
    np.savetxt("../reports/ifac-data/exp2_controller_performance.txt",exp2_control_data,delimiter=' ')
    np.savetxt("../reports/ifac-data/exp2_observer_performance.txt",exp2_observe_data,delimiter=' ')
