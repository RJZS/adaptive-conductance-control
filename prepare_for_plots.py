# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 16:21:18 2021

@author: Rafi
"""
import numpy as np

prep_exp1 = True
if prep_exp1:
    data = np.load("downloaded_data/exp1f.npz")
    t=data['t']; t_ref=data['t_ref']
    sol=data['sol']; sol_ref=data['sol_ref']
    # t = t[:-50000] # Truncate
    # sol = sol[:,:-50000]
    t = t / 1000; t_ref = t_ref / 1000
    print(t_ref[-1])
    print(t[-1])
    
    t_control = t # Define t = 0 as moment controller is switched on.
    tc = np.concatenate((t_ref[:-1] - t_ref[-1], t_control))
    vc = np.concatenate((sol_ref[0,:-1], sol[0,:]))
    v2c = np.concatenate((sol_ref[11,:-1], sol[102,:]))
    
    v = sol[0,:]
    v_hat = sol[11,:]
    error = v - v_hat # Large initial transient.
    logerror = np.log(error)
    
    v_ref = sol[102,:]
    v_ref_hat = sol[102+11,:]
    error_ref = v_ref - v_ref_hat

    tracking_error = v2c - vc
    
    k = 700000
    tc_trunc = tc[:k]; vc_trunc = vc[:k]; v2c_trunc = v2c[:k]
    tracking_error_trunc = tracking_error[:k]
    
    k = -600000
    t_control_trunc = t_control[:k]; error_trunc = error[:k]
    logerror_trunc = logerror[:k]
    
    skp = 10
    tc = tc[0::skp]; t_control = t_control[0::skp]
    vc = vc[0::skp]; v2c = v2c[0::skp]; tracking_error = tracking_error[0::skp]
    tc_trunc = tc_trunc[0::skp]; vc_trunc = vc_trunc[0::skp]
    tracking_error_trunc = tracking_error_trunc[0::skp]
    v2c_trunc = v2c_trunc[0::skp]
    v = v[0::skp]; v_hat = v_hat[0::skp]; error = error[0::skp]
    v_ref = v_ref[0::skp]; v_ref_hat = v_ref_hat[0::skp]; error_ref = error_ref[0::skp]
    
    exp1_control_data = np.vstack((tc, vc, v2c, tracking_error)).T
    exp1_control_data_zoomed = np.vstack((tc_trunc, vc_trunc, v2c_trunc, tracking_error_trunc)).T
    exp1_observe_data = np.vstack((t_control, v, v_hat, v_ref, v_ref_hat, error, error_ref)).T
    exp1_error_zoomed = np.vstack((t_control_trunc, error_trunc, logerror_trunc)).T
    
    np.savetxt("../reports/ifac-data/exp1_controller_performance.txt",exp1_control_data,delimiter=' ')
    np.savetxt("../reports/ifac-data/exp1_controller_zoomed.txt",exp1_control_data_zoomed,delimiter=' ')
    np.savetxt("../reports/ifac-data/exp1_observer_performance.txt",exp1_observe_data,delimiter=' ')
    np.savetxt("../reports/ifac-data/exp1_error_zoomed.txt",exp1_error_zoomed,delimiter=' ')
    
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
    error = Id - Id_hat # Note there's a large initial transient (down to -1323).
    
    Idbef = syn_g * solbef[11,:] * (solbef[0,:] - Esyn)
    Idc = np.concatenate((Idbef[:-1], Id))
    
    # For the 'zoomed in' error plot.
    # j=50000;k=1000000 # NB: this cuts out a large initial transient! Down to -1323...
    # t_trunc = t_control[j:k]; error_trunc = error[j:k]
    # trnc = 75000 # As observer converges so quickly, can truncate time-axis.
    # t_trunc = t[:trnc]; g_ests = g_ests[:,:trnc]
    # Id = Id[:trnc]; Id_hat = Id_hat[:trnc]
    
    skp = 10 # Skip every 'skp'th index. As lualatex struggles with the memory requirements.
    t_control = t_control[0::skp]; tc = tc[0::skp]; vc = vc[0::skp]
    tdc = tdc[0::skp]
    vdc = vdc[0::skp]; Id = Id[0::skp]; Id_hat = Id_hat[0::skp]
    Idc = Idc[0::skp]; gsyn_hat = gsyn_hat[0::skp]
    error = error[0::skp]; tnd = tnd[0::skp]
    
    exp2_control_data = np.vstack((tc, vc)).T
    exp2_dist_data = np.vstack((tdc, vdc, Idc)).T
    exp2_observe_data = np.vstack((t_control, gsyn_hat, Id, Id_hat, error)).T
    
    np.savetxt("../reports/ifac-data/exp2_controller_performance.txt",exp2_control_data,delimiter=' ')
    np.savetxt("../reports/ifac-data/exp2_dist_data.txt",exp2_dist_data,delimiter=' ')
    np.savetxt("../reports/ifac-data/exp2_observer_performance.txt",exp2_observe_data,delimiter=' ')
    
prep_exp3 = False
if prep_exp3:
    # Note how controller plots use t_psd, observer plots use t.
    data = np.load("downloaded_data/exp3f.npz")
    tbef = data['tbef']; t=data['t']; tnd=data['tnd']
    solbef = data['solbef']; sol=data['sol']; solnd=data['solnd']
    # t = t[:450000] # Truncate
    # sol = sol[:,:450000]
    
    t_control = t + tnd[-1] + tbef[-1]
    tc = np.concatenate((tnd[:-1], tbef[:-1] + tnd[-1], t_control))
    vc1 = np.concatenate((solnd[0,:-1], solbef[0,:-1], sol[0,:]))
    vc2 = np.concatenate((solnd[12,:-1], solbef[13,:-1], sol[50,:]))
    vc3 = np.concatenate((solnd[24,:-1], solbef[26,:-1], sol[100,:]))
    vc4 = np.concatenate((solnd[36,:-1], solbef[39,:-1], sol[150,:]))
    vc5 = np.concatenate((solnd[48,:-1], solbef[52,:-1], sol[200,:]))
    
    # tcd = np.concatenate((tbef[:-1] + tnd[-1], t_control)) # t without the first stage
    # vc1d = np.concatenate((solbef[0,:-1], sol[0,:]))
    # vc2d = np.concatenate((solbef[13,:-1], sol[50,:]))
    # vc3d = np.concatenate((solbef[26,:-1], sol[100,:]))
    # vc4 = np.concatenate((solbef[39,:-1], sol[150,:]))
    # vc5 = np.concatenate((solbef[52,:-1], sol[200,:]))
    
    g_ests = sol[126:130,:]
    
    # trnc = 1200000 # For controller plots
    # t_trnc = t_psd[:trnc]; v_trnc = v[:trnc]; v_nd_trnc = v_nd[:trnc] # Note t_psd
    # error_trnc = error[:trnc]
    
    # trnc2 = 400000 # For observer plots
    # t_trnc2 = t[:trnc2]; g_ests_trnc = g_ests[:,:trnc2]
    
    skp = 10 # Skip every 'skp'th index. As lualatex struggles with the memory requirements.
    t_control = t_control[0::skp]; tc = tc[0::skp]; tnd = tnd[0::skp]
    # tcd = tcd[0::skp]
    vc1 = vc1[0::skp]; vc2 = vc2[0::skp]; vc3 = vc3[0::skp]; vc4 = vc4[0::skp]
    vc5 = vc5[0::skp];
    # vc1d = vc1d[0::skp]; vc2d = vc2d[0::skp]; vc3d = vc3d[0::skp]; 
    g_ests = g_ests[:,0::skp]
    
    exp3_full_t_data = np.vstack((tc, vc1, vc2, vc3, vc4, vc5)).T
    # exp3_dist_t_data = np.vstack((tcd, vc1d, vc2d, vc3d, vc4, vc5)).T
    exp3_observe_data = np.vstack((t_control, g_ests)).T
    
    np.savetxt("../reports/ifac-data/exp3_full_t.txt",exp3_full_t_data,delimiter=' ')
    # np.savetxt("../reports/ifac-data/exp3_dist_t.txt",exp3_dist_t_data,delimiter=' ')
    np.savetxt("../reports/ifac-data/exp3_observer_performance.txt",exp3_observe_data,delimiter=' ')
    
    