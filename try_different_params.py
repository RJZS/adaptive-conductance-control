Tfinal= 3000.
dt=0.01
    
Iconst = lambda t: -3.2
Iconst2 = lambda t: -2.5

Iapps = [Iconst, Iconst, Iconst, Iconst2, Iconst2, Iconst2] 

tspan = (0.,Tfinal)

one = Neuron(0.1, [120.,0.1,1.6,0,80.,0.2,2.,0.,0.1], [], 0)
two = Neuron(0.1, [110.,0.1,1.6,0,80.,0.2,2.,0.,0.1], [], 0)
three = Neuron(0.1, [120.,0.1,1.4,0,80.,0.2,2.,0.,0.1], [], 0)

four = Neuron(0.1, [130.,0.1,3.2,0,80.,1.,2.,0.,0.1], [], 0)
five = Neuron(0.1, [135.,0.1,3.2,0,80.,1.,2.,0.,0.1], [], 0)
six = Neuron(0.1, [130.,0.1,3.2,0,80.,1.,2.,0.5,0.1], [], 0)
# Remember, order of currents is Na, H, T, A, KD, L, KCA, KIR, leak

x_0 = [0,0,0,0,0,0,0,0,0,0,0] # 0 syns

network_play = Network([one, two, three, four, five, six], [])
p_play = (Iapps, network_play)

z_0_play = np.concatenate((x_0, x_0, x_0, x_0, x_0, x_0))

print("Starting 'play' simulation. Tfinal = {}".format(Tfinal))
start_time = time.time()
out_play = solve_ivp(lambda t, z: no_observer(t, z, p_play), tspan, z_0_play, rtol=2e-3,atol=2e-3,
                t_eval=np.linspace(0,Tfinal,int(Tfinal/dt)), method='LSODA')#, vectorized=True)
end_time = time.time()
print("'Play' simulation time: {}s".format(end_time-start_time))

t = out_play.t
sol = out_play.y

t = t.astype('float32')
sol = sol.astype('float32')

np.savez("try_params.npz", t=t, sol=sol)