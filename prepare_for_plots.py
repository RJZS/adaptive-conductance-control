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
    v_ref = sol[0,:]
    g_ests = sol[22:25,:]
    error = sol[0,:] - sol_ref[0,:]
    controller_start_time = 20. # The saving of this setting is currently not automated.
    
    exp1_plot_data = np.vstack((t, v, v_ref, g_ests, error)).T
    np.savetxt("../reports/ifac-data/exp1_data.txt",exp1_plot_data,delimiter=' ')