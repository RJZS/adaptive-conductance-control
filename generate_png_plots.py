# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 16:21:18 2021

@author: Rafi
"""
import numpy as np
import matplotlib.pyplot as plt

# After running this script, on each file need to run 'convert -trim <fname of image> <fname of output img>' where the second
# fname can be the same as the first (ie to overwrite the file).
my_dpi=300 # Was originally 100 in example code.
lw = 0.9 # Linewidth. Default is 1.
def create_plot(x, y, fname, color='C0', vline_xs=[]):
    fig = plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi) # Change figsize?
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.plot(x, y, linewidth=lw, color=color)
    plt.vlines(vline_xs, y.min(), y.max(), linestyles='dashed', color='k', linewidth=lw)
    fig.savefig(f"../reports/ifac-raw-plots/{fname}.png", dpi=my_dpi, transparent=True)

## NEURON DEMO FIG
print("Neuron Demo")
data = np.load("data/neuron_demo.npz")
t=data['t']; sol=data['sol']; currs=data['currs']
t = t / 1000;

i=68000;j=82000 # Will 'zoom in' to a single burst.
t = t[i:j]
v = sol[0,i:j]
INa = currs[0,i:j]
IK = currs[4,i:j]
ICaT = currs[2,i:j]
print(f"T max - min: {t[-1] - t[0]}")
print(f"v min: {v.min()} max: {v.max()}")
print(f"INa min: {INa.min()} max: {INa.max()}")
print(f"IK min: {IK.min()} max: {IK.max()}")
print(f"ICaT min: {ICaT.min()} max: {ICaT.max()}")

create_plot(t, v, 'neuron_demo_i', 'C3')
create_plot(t, INa, 'neuron_demo_ii')
create_plot(t, IK, 'neuron_demo_iii')
create_plot(t, ICaT, 'neuron_demo_iv')

## NEURON DEMO FIG TWO BURSTS
print("Neuron Demo Two Bursts")
data = np.load("data/neuron_demo_twobursts.npz")
t=data['t']; sol=data['sol']; currs=data['currs']
t = t / 1000;

i=100000;j=165000 # Will 'zoom in' to two bursts.
t = t[i:j]
v = sol[0,i:j]
INa = currs[0,i:j]
IK = currs[4,i:j]
ICaT = currs[2,i:j]
print(f"T max - min: {t[-1] - t[0]}")
print(f"v min: {v.min()} max: {v.max()}")
print(f"INa min: {INa.min()} max: {INa.max()}")
print(f"IK min: {IK.min()} max: {IK.max()}")
print(f"ICaT min: {ICaT.min()} max: {ICaT.max()}")

create_plot(t, v, 'neuron_demo2_i', 'C3')
create_plot(t, INa, 'neuron_demo2_ii')
create_plot(t, IK, 'neuron_demo2_iii')
create_plot(t, ICaT, 'neuron_demo2_iv')

## REFTRACK FIGS
data = np.load("data/exp1_relu.npz")
t=data['t']; t_ref=data['t_ref']
sol=data['sol']; sol_ref=data['sol_ref']
# t = t[:-50000] # Truncate
# sol = sol[:,:-50000]
t = t / 1000; t_ref = t_ref / 1000

t_control = t # Define t = 0 as moment controller is switched on.
tc = np.concatenate((t_ref[:-1] - t_ref[-1], t_control))
vc = np.concatenate((sol_ref[0,:-1], sol[0,:]))
v2c = np.concatenate((sol_ref[11,:-1], sol[102,:]))

v = sol[0,:]
v_hat = sol[11,:]
error = v - v_hat # Large initial transient.

v_ref = sol[102,:]
v_ref_hat = sol[102+11,:]
error_ref = v_ref - v_ref_hat

tracking_error = v2c - vc

k = 1050000
tc_trunc = tc[:k]; vc_trunc = vc[:k]; v2c_trunc = v2c[:k]
tracking_error_trunc = tracking_error[:k]

k = -600000
t_control_trunc = t_control[:k]; error_trunc = error[:k]

# Truncate tc a little for the bottom plot.
j = 2250000
tc = tc[:j]
tracking_error = tracking_error[:j]
    
print("\n\nREFTRACK")
print(f"tc_trunc start: {tc_trunc[0]} end: {tc_trunc[-1]}")
print(f"v2c_trunc min: {v2c_trunc.min()} max: {v2c_trunc.max()}")
print(f"vc_trunc min: {vc_trunc.min()} max: {vc_trunc.max()}")
print(f"tc start: {tc[0]} end: {tc[-1]}")
print(f"tracking_error min: {tracking_error.min()} max: {tracking_error.max()}")

create_plot(tc_trunc, v2c_trunc, 'exp1_a_i', 'C3')
create_plot(tc_trunc, vc_trunc, 'exp1_a_ii')
create_plot(tc, tracking_error, 'exp1_b_i', 'C3')

## DISTREJ FIGS

data = np.load("data/exp2_relu.npz")
tbef = data['tbef']; t=data['t']; tnd=data['tnd']
solbef = data['solbef']; sol=data['sol']; solnd=data['solnd']
t = t / 1000; tbef = tbef / 1000; tnd = tnd / 1000
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
vdc = np.concatenate((solnd[11,:-1], solbef[12,:-1], sol[27,:]))

gsyn_hat = sol[24,:]

syn_g = 2.5; Esyn = -90; # Saving these parameters could be automated.
Id = syn_g * sol[11,:] * (sol[0,:] - Esyn)
Id_hat = gsyn_hat * sol[23,:] * (sol[12,:] - Esyn)
error = Id - Id_hat # Note there's a large initial transient (down to -1323).

k=140000
t_control_trunc = t_control[:k]; error_trunc = error[:k]

tdc = np.concatenate((tbef[:-1] + tnd[-1], t_control))
Idbef = syn_g * solbef[11,:] * (solbef[0,:] - Esyn)
Idc = np.concatenate((Idbef[:-1], Id))

print("\n\nDISTREJ")
print(f"tc start: {tc[0]} end: {tc[-1]}")
print(f"vc min: {vc.min()} max: {vc.max()}")
print(f"vdc min: {vdc.min()} max: {vdc.max()}")
print(f"tdc start: {tdc[0]} end: {tdc[-1]}")
print(f"Idc min: {Idc.min()} max: {Idc.max()}")
print(f"t_control_trunc start: {t_control_trunc[0]} end: {t_control_trunc[-1]}")
print(f"error_trunc min: {error_trunc.min()} max: {error_trunc.max()}")

create_plot(tc, vc, 'exp2_a_i', 'C3')
create_plot(tc, vdc, 'exp2_a_ii')
create_plot(tdc, Idc, 'exp2_b_i')
create_plot(t_control_trunc, error_trunc, 'exp2_b_ii', 'C3')

# ORCHESTRON FIGS

# Note how controller plots use t_psd, observer plots use t.
data = np.load("data/exp3_relu.npz")
tbef = data['tbef']; t=data['t']; tnd=data['tnd']
solbef = data['solbef']; sol=data['sol']; solnd=data['solnd']
t = t / 1000; tbef = tbef / 1000; tnd = tnd / 1000
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

print("\n\nORCHESTRON")
print(f"tc start: {tc[0]} end: {tc[-1]}")
print(f"vc1 min: {vc1.min()} max: {vc1.max()}")
print(f"vc2 min: {vc2.min()} max: {vc2.max()}")
print(f"vc3 min: {vc3.min()} max: {vc3.max()}")
print(f"vc4 min: {vc4.min()} max: {vc4.max()}")
print(f"vc5 min: {vc5.min()} max: {vc5.max()}")

create_plot(tc, vc1, 'exp3_i', 'C0')
create_plot(tc, vc2, 'exp3_ii', 'C3')#, [4,10]) # Removed lines as drawing on pgfplots.
create_plot(tc, vc3, 'exp3_iii', 'C4')#, [4,10])
create_plot(tc, vc4, 'exp3_iv', 'C0')
create_plot(tc, vc5, 'exp3_v', 'C3')#, [4,10])