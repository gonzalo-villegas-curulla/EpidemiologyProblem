#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pylab as plt
import matplotlib
from   numpy.random import Generator, PCG64
from scipy.spatial.distance import pdist, squareform
import time
from numba.experimental import jitclass 


seed   = 42                  
rg     = Generator(PCG64(seed))  

# @jitclass
class GVCNX:
    def __init__(self, ini):
        
        Xax, Yax, N, positions, R = ini
        
        #  Initialisations 
        self.nodes = {}
        self.nodes['pos']        = dict()
        self.nodes['state']      = dict()
        self.nodes['propensity'] = dict()
        self.nodes['tau']        = dict()
        self.nodes['neighbors']  = dict()
        self.DistMX              = np.zeros((N,N),dtype=np.float64)
        self.positions           = positions
        
        
        #  Populate :
        for idx, (x, y) in enumerate(zip(Xax.ravel(), Yax.ravel())):
            self.nodes['pos'][idx] = np.array([x, y], dtype=np.float64)
            self.nodes['state'][idx]      = 0
            self.nodes['propensity'][idx] = 0
            self.nodes['tau'][idx]        = 0
                    
        
        # Compute adjancency:
        
        self.DistMX    = squareform(pdist(self.positions))
        # num_nodes = len(self.nodes)
        for idx in range(N):
            
            self.nodes['neighbors'][idx]={}
            ctr = 0
            for jdx in range(idx+1, N):
                if self.DistMX[idx, jdx] <= R:
                    if ctr==0:
                        self.nodes['neighbors'][idx] = {jdx}
                        ctr = 1
                    else:
                        self.nodes['neighbors'][idx].add(jdx)
                        # ctr += 1
           
        
#  =====================================================================
    
# @jitclass
class SIRnetwork:

    def __init__(self, N, beta, alpha, T, L, Gin):
        self.N         = N
        self.beta      = beta
        self.alpha     = alpha
        self.T         = T
        self.L         = L
        self.Gloc      = Gin
        self.SOL = []
        self.listS = list(np.arange(0,self.N,1))
        self.listI = list([])
        self.listR = list([])
        
        # Inject 1 infectious    
        IDX_INFEC = int(rg.random()*N)
        self.Gloc.nodes['state'][IDX_INFEC] = np.int32(1)
        
        self.listS.pop(IDX_INFEC)
        self.listI.append(IDX_INFEC)
        self.listI.sort()
        
        
    # +++++++++++++++++++++++
    
    def calc_propensities(self): # <<<<<<<<<<<<<<<<<<<<<<<<<
        
        # Set all lambdas_i to zero :        
        for idx in range(self.N):
            
            self.Gloc.nodes['propensity'][idx] = 0
            
            
        # Loop over all nodes
        for idx in range(self.N):
            
            # Node=susceptible? 
            # Calc neighboring propensities for infection
            if self.Gloc.nodes['state'][idx] == 0 :
                
                neigh_states = [self.Gloc.nodes['state'][jdx] for jdx in list(self.Gloc.nodes['neighbors'][idx]) ] 
                
                lambda_ith = self.beta * np.sum([state==1 for state in neigh_states])
                
                # Set node propensity and draw waiting time:
                if lambda_ith > 0.:
                    self.Gloc.nodes['propensity'][idx] = lambda_ith
                    
                    u_i = rg.random()
                    self.Gloc.nodes['tau'][idx] = -np.log(1. - u_i) / lambda_ith
                
            # Or node i-th = Infectious:
            elif self.Gloc.nodes['state'][idx] == 1:
                
                # Set node propensity to recovering:
                self.Gloc.nodes['propensity'][idx] = self.alpha    
                
                u_i = rg.random()
                self.Gloc.nodes['tau'][idx] = -np.log(1. - u_i) / self.alpha
            # print(self.Gloc.nodes['tau'])    
                
    def next_event(self):        
         
        # node_keys  = list(nx.get_node_attributes(G, 'tau'))
        IDXkeys = np.array(list(self.Gloc.nodes['tau'].keys()))
        
        # Get waiting times for active reaction channels from G:
        waiting_times = np.array(list(self.Gloc.nodes['tau'].values()))
        # print(waiting_times)
        
        
        # filtmask = np.nonzero(waiting_times>0)[0]
        filtmask = np.nonzero(waiting_times)[0]
        
        # Filter results to non-nul waiting times
        waiting_times = waiting_times[filtmask]
        IDXkeys       = IDXkeys[filtmask]
        
        
            
        # Select minimal waiting time:
        tau = np.min(waiting_times)
        # print(tau)
        # if tau<0:
        #     print('negative tau')

        i_selected = np.where(waiting_times == tau)[0][0]


        return(i_selected, tau, IDXkeys)
    
    # +++++++++++++++++++++++++++++++++++
    
    # def update_states(self):
    def update_states(self, X, tau, i_selected, IDXkeys):
        
        
        S, I, R = X
        
        # Get node indices with ACTIVE reaction channels:
        
        # node_indices = list(np.nonzero(list(self.Gloc.nodes['tau'].values())))
        # node_indices = list(node_indices[0])  
        
        node_indices = list(IDXkeys)

        # Get node id corresponding to reaction channel i_0
        i_node = node_indices[i_selected]


        # Update waiting times for remaining nodes:
        node_indices.pop(i_selected) # Remove this index from category

        
        
        # print('tau  {:.5f}'.format(tau))
        for jdx in node_indices:
            self.Gloc.nodes['tau'][jdx] -= tau
                    



        #===== Update network's state: =====
        
        # i_node TRIGGERED an event:
        state_before = self.Gloc.nodes['state'][i_node]


        # (i_node) TRIGGERED S->I 
        if state_before==0:             
            
            S -= 1
            I += 1
            
            # Update node i_selected (S -> I): 
            self.Gloc.nodes['state'][i_node] = 1

            # Update node's propensity:
            self.Gloc.nodes['propensity'][i_node] = self.alpha # = 1*alpha 
            
            # Draw new waiting time for i_selected:
            u_i = rg.random()
            self.Gloc.nodes['tau'][i_node] = -np.log(1. - u_i) / self.alpha

            # Update propensities of i's NEIGHBOURS:
            susceptible_neighbors = np.array([jdx for jdx in self.Gloc.nodes['neighbors'][i_node] if self.Gloc.nodes['state'][jdx]==0])

            if len(susceptible_neighbors)>0:
                for jdx in susceptible_neighbors: 
                    # Update j's propensity:
                    self.Gloc.nodes['propensity'][jdx] += self.beta
                       
                    # Draw new waiting time for j:
                    u_j = rg.random()
                    self.Gloc.nodes['tau'][jdx] = -np.log(1. - u_j) / (self.Gloc.nodes['propensity'][jdx]+1e-12)


        # (i_node) TRIGGERED I->R
        # If state_before=I : update it to R
        else: 
            
            I -= 1
            R += 1
            

            # Update state of node in graph: 
            self.Gloc.nodes['state'][i_node] = 2

            # Remove i from reaction channels:
            self.Gloc.nodes['propensity'][i_node] = 0.
            del self.Gloc.nodes['tau'][i_node]

            # Update propensities of i's neighbors:
            susceptible_neighbors = np.array([jdx for jdx in self.Gloc.nodes['neighbors'][i_node] if self.Gloc.nodes['state'][jdx]==0])

            if len(susceptible_neighbors)>0:
                for jdx in susceptible_neighbors:  
                    
                    # Update j's propensity:
                    self.Gloc.nodes['propensity'][jdx] -= self.beta
                                       
                    # If node propensity is zero remove channel:
                    if np.isclose(self.Gloc.nodes['propensity'][jdx], 0.):
                        del self.Gloc.nodes['tau'][jdx]
                    
                    # Else, draw new waiting time for j:
                    else:
                        u_j = rg.random()
                        self.Gloc.nodes['tau'][jdx] = -np.log(1. - u_j) / self.Gloc.nodes['propensity'][jdx]
                        
        

            
        return([S, I, R])
    
    
    
    
    # +++++++++++++++++++++++++++++++++++    
    
    def simul(self):
        # G, beta, alpha, T)
        
        #--- Initialization: ---
        node_states = self.Gloc.nodes['state']        
        
        N = len(node_states)
        S = sum(X==0 for X in node_states.values())
        I = sum(X==1 for X in node_states.values())
        R = N - S - I 
        
        # Calculate node propensities:
        self.calc_propensities()    

        t = 0

        # Vector to save time solution: 
        self.SOL.append([t, S, I, R])
        
        while t < T:
            # Check if no waiting times are left (no more reactions can happen):
            
            # if len(nx.get_node_attributes(self.Gloc,'tau')) == 0:
            #     X_t.append([T, S, I, R])
            #     break
        
        
            # if len(self.Gloc.nodes['tau'].values())==1:
            # if len(self.Gloc.nodes['tau'].values())==1:
            if  np.sum(list(GSIR.Gloc.nodes['tau'].values())) == 0:
                self.SOL.append([T, S, I, R])
                break
            
            # First reaction event sampling step
            i, tau, IDXk = self.next_event()
            
            # Advance time:
            t += tau
            # print('Time {:.5f}'.format(t))


            #===== Update: =====
            
            [S, I, R] = self.update_states([S, I, R], tau, i, IDXk)
            
            
            self.SOL.append([t, S, I, R])    
            
        
        return(np.transpose(self.SOL))
        


#####################################################################
#####################################################################
#               End of Function Definitions
#####################################################################
#####################################################################

    
#--- Simulation parameters: ---
seed   = 42                     # Set seed of PRNG state 
rg     = Generator(PCG64(seed))  # Initialize bit generator (here PCG64) with seed

beta   = np.float64(0.1)   # Infection rate
alpha  = np.float64(0.03)  # Recovery rate

# k      = np.int32(3)     # Degree of nodes   ????
N      = np.int32(1000)   # Number of nodes
T      = np.float64(300)   # Time of simulation
L      = np.float64(100)  # Length of the square domain 


A   = L**2                # Area
dh  = np.sqrt(A/N)        # regularized grid step
R   = 1.2*dh  *np.sqrt(2) # R-distance for node-neighbors

NumSimuls = np.int32(10)

# ~~~~~~~~~~ Regular grid ~~~~~~~~~~~~~~~

xax = np.arange(0,L,dh)
yax = np.arange(0,L,dh)

# Readjust N
N = len(xax)*len(yax)

Xax,Yax = np.meshgrid(xax,yax)
Xax = np.float64(np.reshape(Xax, (N,1)))
Yax = np.float64(np.reshape(Yax, (N,1)))

Xax = Xax + rg.random((N,1))*0.5*dh
Yax = Yax + rg.random((N,1))*0.5*dh
pos = np.transpose(np.squeeze([Xax, Yax]))




X_array = []
for IdxSimul in range(NumSimuls):
    # G = G0.copy()
    G0   = GVCNX([Xax, Yax, N, pos, R])
    GSIR = SIRnetwork(N=N, beta=beta, alpha=alpha, T=T, L=L, Gin=G0)

    tmp = GSIR.simul() 

    X_array.append(tmp)
    

colors  = ['b','r','g']
legends = [r'$N_S$',r'$N_I$',r'$N_R$']


fig = plt.figure(4)
ax  = plt.subplot()

TIMAX = []

for X_t in X_array:
    [ax.plot(X_t[0,:],X_,'o',c=c_,alpha=1/NumSimuls) for X_, c_ in zip(X_t[1:,:], colors)]
    # [ax.plot(X_t[0,:],X_,'o',c='r',alpha=1/NumSimuls) for X_, c_ in zip(X_t[2:3,:], colors)]
    # [ax.plot(), for ]   
    tImax = np.argmax(X_t[2,:])
    tvec = X_t[0,:]
    thisT = tvec[tImax]
    TIMAX.append(thisT)
    # plt.plot([thisT,thisT],[0,N],':k', alpha=0.125)    

ax.legend(legends)
plt.xlabel(r"Temps [jours]")
plt.ylabel(r"Numéro de cas")


