# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 17:08:56 2021

@author: Rafi
"""
import numpy as np
from scipy.integrate import solve_ivp
import time

from network_and_neuron import Synapse, Neuron, Network
from network_odes import main, no_observer
 
# Playing with model to find bursting behaviour.
playing = True
if playing:
    Tfinal = 3500//2.
    dt=0.01
    syn1 = Synapse(0.6, 1) # 0.5 and 2 seem to have the same result.
    syn2 = Synapse(0.6, 0)
    syn3 = Synapse(0.6, 4)
    syn4 = Synapse(0.6, 3)
    
    # To hub neuron
    synhub1 = Synapse(2, 0) # 0.5 does nothing to stop spiking. 6 give very wide bursts.
    synhub2 = Synapse(2, 4)
    
    
    Iapp = lambda t : 2 + np.sin(2*np.pi/10*t)
    def Irb(t): # For rebound burster
        if t < 400 or t > 500:
            return -3.7
        else:
            return -7.5
    
    def Ioffset(t): # So two neurons in HCO don't burst simultaneously
        if t < 200:
            return -4.
        else:
            return -2.
        
    def Ioffset2(t): # So two neurons in HCO don't burst simultaneously
        if t < 300:
            return -4.
        else:
            return -2.
        
    Iconst = lambda t: -2.
    
    
    Iapps = [Iconst, Ioffset, lambda t: 38, Iconst, Ioffset2] 
    # Iapps = [lambda t: 34] # 34 gives bursts, length about 1600. 38 gives spiking.
    
    tspan = (0.,Tfinal)
    
    # neur_one_play = Neuron(1., [120.,0,5.,0,36.,0,0,0.03], []) # For rebound burster.
    # neur_two_nodist = Neuron(1., [120.,0,0,0,36.,0,0,0.3], [syn2])
    
    #x_0 = [0,0,0,0,0,0,0,0,0,0,0]; # V, m, h, mH, mT, hT, mA, hA, mKD, mL, Ca
    
    # neur_one_play = Neuron(0.1, [100.,0.08,3.5,0,70.,0.5,1.6,0.,0.1], [])  # gNa, gH, gT, gA, gKD, gL, gKCa, gKir, gleak
    # neur_two_play = Neuron(0.1, [120.,0.1,2.,0,80.,0.4,2.,0.,0.1], [])
    
    one = Neuron(0.1, [120.,0.1,2.,0,80.,0.4,2.,0.,0.1], [syn1], 0)
    two = Neuron(0.1, [120.,0.1,2.,0,80.,0.4,2.,0.,0.1], [syn2], 1)
    
    three = Neuron(0.1, [80.,0.1,0.2,0,30.,0.,1.,0.,0.1], [synhub1, synhub2], 2) # Hub neuron
    # three = Neuron(0.1, [120.,0.1,2.,0,80.,0.4,2.,0.,0.1], [synhub1], 1) # Hub neuron
    
    four = Neuron(0.1, [120.,0.1,1.,0,80.,0.4,2.,0.,0.1], [syn3], 1)
    five = Neuron(0.1, [120.,0.1,1.,0,80.,0.4,2.,0.,0.1], [syn4], 0)
    # Remember, order of currents is Na, H, T, A, KD, L, KCA, KIR, leak
    
    res_g = 0.005
    el_connects = np.array([[res_g, 1, 2],[res_g, 3, 2]])
    # el_connects = np.array([[res_g, 1, 2]])
    
    # v_0 = np.array([-70.])
    # starting_gate_vals = neur_one_play.initialise(v_0)
    x_0 = [0,0,0,0,0,0,0,0,0,0,0,0,0] # 2 syns
    
    # # Rebound burster
    # neur_one_nodist = Neuron(1., [120.,0,0,0,36.,0,0,0.3], [])
    # network_play = Network([one, two, three, four, five], np.zeros((5,5)))
    network_play = Network([one, two, three, four, five], el_connects)
    # network_play = Network([one, two, three], el_connects)
    # network_play = Network([four, five], np.zeros((2,2)))
    p_play = (Iapps, network_play)
    
    z_0_play = np.concatenate((x_0, x_0, x_0, x_0, x_0))
    # z_0_play = x_0
    # z_0_play[0] = -70
    print("Starting 'play' simulation. Tfinal = {}".format(Tfinal))
    start_time = time.time()
    out_play = solve_ivp(lambda t, z: no_observer(t, z, p_play), tspan, z_0_play, rtol=1e-2,atol=1e-2,
                    t_eval=np.linspace(0,Tfinal,int(Tfinal/dt)), method='LSODA')#, vectorized=True)
    end_time = time.time()
    print("'Play' simulation time: {}s".format(end_time-start_time))
    
    t = out_play.t
    sol = out_play.y
    
    # plt.plot(t,sol[0,:],t,sol[13,:],t,sol[26,:])
    # plt.plot(t_play,sol_play[0,:],t_play2,sol_play2[0,:])
    
    print("Converting and saving...")
    t=t.astype('float32')
    sol=sol.astype('float32')
    np.savez("simulate_experiment_3.npz", t=t, sol=sol)