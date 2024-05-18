#!/usr/bin/env python
# coding: utf-8


import numpy as np
import networkx as nx
import matplotlib.pylab as plt
import timeit, time
from numpy.random import Generator, PCG64, SeedSequence
from networkx import *
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import Voronoi, voronoi_plot_2d


seed = 42                     # Set seed of PRNG state 
rg   = Generator(PCG64(seed))  # Initialize bit generator (here PCG64) with seed
print(f"PRNG: {rg.bit_generator}")
    

# :::::::::::::::::::::::::::::::::::
# Functions
# :::::::::::::::::::::::::::::::::::


def calculate_node_propensities(G, beta, alpha):
    
    N = len(G)
    
    nx.set_node_attributes(G, 0., 'lambda')
    
    for i in range(N):
        
        # If node=susceptible: calculate total propensity for infection:
        if G.nodes[i]['state'] == 0:
            # Get i's neighbors:
            neighbor_states = [G.nodes[j]['state'] for j in G.neighbors(i)] 
            lens = [distances[i, jjdx] for jjdx in G.neighbors(i)]
            # Total node rate from number of neighbors that are infectious:
            if flag_edglen:
                lambda_i = beta * np.sum([state==1 for state in neighbor_states]) * np.sum(lens) # <<<<<<<
            else:
                lambda_i = beta * np.sum([state==1 for state in neighbor_states])
            
            # Set node propensity and draw waiting time:
            if lambda_i > 0.:
                G.nodes[i]['lambda'] = lambda_i
                
                u_i = rg.random()
                G.nodes[i]['tau'] = -np.log(1. - u_i) / lambda_i
            
        # If node=infectious: set propensity to alpha: ---
        
        elif G.nodes[i]['state'] == 1:
            # Set node propensity:
            G.nodes[i]['lambda'] = alpha    
            
            u_i = rg.random()
            G.nodes[i]['tau'] = -np.log(1. - u_i) / alpha
    return()

######################################################

def draw_next_event_first_reaction(G):
    
    '''Input: the network G.
    Output: selected reaction channel, i_selected, and the waiting time until the event, tau.'''
     
    # Get waiting times for active reaction channels from G:
        
    node_indices  = list(nx.get_node_attributes(G, 'tau'))
    waiting_times = list(nx.get_node_attributes(G, 'tau').values()) 
        
    # Select reaction with minimal waiting time:
    tau = np.min(waiting_times)
    
    i_selected = np.where(waiting_times == tau)[0][0]

    return(i_selected, tau)

########################################################

def update_states(G, X, tau, i_selected):
    # State counts:
    S, I, R = X
    
    # Get node indices for active reaction channels from G:
    node_indices = list(nx.get_node_attributes(G, 'tau'))

    # Get node id corresponding to reaction channel i_0
    i_node = node_indices[i_selected] 

    #--- Update waiting times for remaining nodes: ---
    
    node_indices.pop(i_selected) # Remove this index from category
    
    # "Advance" time 
    for jdx in node_indices:
        G.nodes[jdx]['tau'] -= tau

    #===== Update network state: =====
    
    # i_node TRIGGERED an event
    state_before = G.nodes[i_node]['state']

    # (i_node) TRIGGERED S->I   (If state_before=S: update to I)
    if state_before==0:             
        # Update state counts: 
        S -= 1
        I += 1

        # Update state of node i_selected (S -> I): 
        G.nodes[i_node]['state'] = 1

        # Update node i_selected's propensity:
        G.nodes[i_node]['lambda'] = alpha
        
        # Draw new waiting time for i_selected:
        u_i = rg.random()
        G.nodes[i_node]['tau'] = -np.log(1. - u_i) / alpha

        # Update propensities of i's NEIGHBOURS:
            
        susceptible_neighbors = np.array([j for j in G.neighbors(i_node) if G.nodes[j]['state']==0])

        if len(susceptible_neighbors)>0:
            for jdx_nei in susceptible_neighbors: 
                
                if flag_edglen:
                    edglen = distances[i_node, jdx_nei]
                else:
                    edglen = 1
                # Update j's propensity:
                G.nodes[jdx_nei]['lambda'] += beta/edglen # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                   
                # Draw new waiting time for j:
                u_j = rg.random()
                G.nodes[jdx_nei]['tau'] = -np.log(1. - u_j) / G.nodes[jdx_nei]['lambda']

    # (i_node) TRIGGERED I->R
    # If state_before=I : update to R ---
    else: # (Note that this is a simple else statement since state_before is either S or I)
        # Update state counts: 
        I -= 1
        R += 1

        # Update state of node in graph: 
        G.nodes[i_node]['state'] = 2

        # Remove i from reaction channels:
        G.nodes[i_node]['lambda'] = 0.
        del G.nodes[i_node]['tau']

        # Update propensities of i's neighbors:
        susceptible_neighbors = np.array([j for j in G.neighbors(i_node) if G.nodes[j]['state']==0])

        if len(susceptible_neighbors)>0:
            for j_nei in susceptible_neighbors:            
                
                # Update j's propensity:
                if flag_edglen:
                    edglen = distances[i_node, j_nei]
                else:
                    edglen = 1
                G.nodes[j_nei]['lambda'] -= beta/edglen ### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                   
                # If node propensity is zero remove channel:
                if np.isclose(G.nodes[j_nei]['lambda'], 0.):
                    del G.nodes[j_nei]['tau']
                
                # Else, draw new waiting time for j:
                else:
                    u_j = rg.random()
                    G.nodes[j_nei]['tau'] = -np.log(1. - u_j) / G.nodes[j_nei]['lambda']
                    
    ## =========  VISUALIZE  =======================
    if flag_visual:
        nx.draw_networkx_edges(G, position_dict, edge_color="black", width=0.2)
        options = {"edgecolors": "tab:gray", "node_size": 200, "alpha": 0.9, "label":"true"}
        
        # get status of all ndoes
        node_states = nx.get_node_attributes(G, 'state')
        status = list(node_states.values())
        idxS = list(np.nonzero(status==np.int64(0)))[0]
        idxI = np.nonzero(status==np.int64(1))[0]
        idxR = np.nonzero(status==np.int64(2))[0]
                
        nx.draw_networkx_nodes(G, pos=position_dict,nodelist=idxS, node_color="tab:blue", **options)
        nx.draw_networkx_nodes(G, pos=position_dict,nodelist=idxI, node_color="tab:red", **options)
        nx.draw_networkx_nodes(G, pos=position_dict,nodelist=idxR, node_color="tab:green", **options)
        plt.show()
        time.sleep(1)                    
                    

    
    return([S, I, R])

########################################################


def first_reaction_method_SIR_graph_linear(G, beta, alpha, T):
    
    #--- Initialization: ---
    node_states = nx.get_node_attributes(G, 'state')
    
    Sinf = []
    N = len(node_states)
    S = sum(X==0 for X in node_states.values())
    I = sum(X==1 for X in node_states.values())
    R = N - S - I 
    
    # Calculate and store node propensities:
    calculate_node_propensities(G, beta, alpha)    

    # Set initial time t = 0:
    t = 0

    # Vector to save temporal evolution of state numbers over time: 
    X_t = [] 
    X_t.append([t, S, I, R])
    
    #  Keep drawing events until t >= T: 
    while t < T:
        # Check if no putative waiting times are left (no more reactions can happen):
        if len(nx.get_node_attributes(G,'tau')) == 0:
            X_t.append([T, S, I, R])
            break
        
        #===== First reaction event sampling step: =====
        i, tau = draw_next_event_first_reaction(G) 
       
        # Advance time
        t += tau

        #===== Update: =====

        [S, I, R] = update_states(G, [S, I, R], tau, i)
        
        #--- Save current state numbers to X_t: ---
        X_t.append([t, S, I, R])       


    Sinf = S
    return(np.array(X_t).transpose(), Sinf) 

#####################################################################
#####################################################################
#               End of Function Definitions
#####################################################################
#####################################################################


#--- Simulation parameters: ---
beta   = np.float64(0.1)   # Infection rate
alpha  = np.float64(0.03)  # Recovery rate
k      = np.int32(3)     # Degree of nodes
N      = np.int32(100)   # Number of nodes
T      = np.float64(300)   # Time of simulation

NumSimuls = np.int32(20)

## PARMETERS grid
L   = np.float64(10)  # Length of the square domain 
A   = L**2 # Area
dh  = np.sqrt(A/N) # regularized grid step
R   = 1.*dh  *np.sqrt(2) # R-distance for node-neighbors

flag_visual = 1
flag_edglen = 1

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



# Generate Graph =======================
myG = nx.Graph()
myG.add_nodes_from(range(N))

# Dictionary of node positions =======================
position_dict = dict()
for i, (x, y) in enumerate(zip(Xax.ravel(), Yax.ravel())):
    position_dict[i] = np.array([x, y], dtype=np.float64)

myG.add_nodes_from(position_dict.keys())

# SET EDGES based on R-distance =======================
positions = np.array(list(position_dict.values()))
distances = squareform(pdist(positions))
num_nodes = len(position_dict)
for i in range(num_nodes):
    for j in range(i+1, num_nodes):
        if distances[i, j] <= R:
            myG.add_edge(i, j)

degs = np.array(myG.degree())
degs = degs[:,1]
print('Av degree: ', np.mean(degs))



# ~~~~~~~~~~  ~~~~~~~~~~~~~~~

X_array = []

Sinf = np.zeros(NumSimuls)

for IdxSimul in range(NumSimuls):
    
    G = myG.copy()

    I_nodes = [0]
    R_nodes = []

    states = np.zeros(N, dtype=int)
    states[I_nodes] = 1
    states[R_nodes] = 2

    nx.set_node_attributes(G, 'state', 0)
    for i,state in enumerate(states):
        G.nodes[i]['state'] = state


    X_t, Sinf[IdxSimul] = first_reaction_method_SIR_graph_linear(G, beta, alpha, T)

    X_array.append(X_t)

#==========================
# Plot:
#==========================


colors = ['b','r','g']
legends = [r'$N_S$',r'$N_I$',r'$N_R$']

fig = plt.figure(4)
ax  = plt.subplot()

TIMAX = []

for X_t in X_array:
    [ax.plot(X_t[0,:],X_,'o',c=c_,alpha=0.125) for X_, c_ in zip(X_t[1:,:], colors)]
    # [ax.plot(), for ]   
    tImax = np.argmax(X_t[2,:])
    tvec = X_t[0,:]
    thisT = tvec[tImax]
    TIMAX.append(thisT)
    plt.plot([thisT,thisT],[0,N],':k', alpha=0.125)
    
    
ax.legend(legends)
plt.xlabel(r"Temps [jours]")
plt.ylabel(r"Numéro de cas")
plt.title(r"Average degree: {:.2f}".format(np.mean(degs)))




