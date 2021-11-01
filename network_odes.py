# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 19:16:03 2021

@author: Rafi
"""
from network_and_neuron import Neuron, Network

def main(t,z,p):
    Iapps = p[0]
    network = p[1]
    (α,γ) = p[3]
    to_estimate = p[4] # Which maximal conductances to estimate
    controller_law = p[5] # Control law to use for the neurons
    
    # Assuming all the neurons are of the same model:
    len_neur_state = network.neurons[0].NUM_GATES + 1
    
    dz = -z**2
    return dz

