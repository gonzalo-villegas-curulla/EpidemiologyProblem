#!/usr/bin/env python
# coding: utf-8

import numpy as np
import networkx as nx
import matplotlib.pylab as plt
import time#, timeit, 
from numpy.random import Generator, PCG64 #, SeedSequence
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import Voronoi#, voronoi_plot_2d
import scipy.special
from shapely import Polygon#,LineString, Point
from collections import defaultdict
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as pltPolygon
import os    
from networkx_robustness import networkx_robustness
from Model01_func import Model01_func 

    
#####################################################################
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
#  1. Function definitions                                          #
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
#####################################################################

# from rk4 import rk4

def calc_propensities(G, beta, alpha, adj_matrix, edge_lengths):
    
    N = len(G)
    
    nx.set_node_attributes(G, 0., 'lambda')
    
    for i in range(N):
        
        # If node=susceptible: calculate total propensity for infection:
        if G.nodes[i]['state'] == 0:
            # Get i's neighbors:
            neighbor_states = [G.nodes[j]['state'] for j in G.neighbors(i)] 

            neighbors = np.array([j for j in G.neighbors(i)])
            
            
            total_contact_surface = 0
            for idx in range(len(neighbors)):
                total_contact_surface += edge_lengths.get((i,neighbors[idx]))


            infective_neighbors = np.array([j for j in G.neighbors(i) if G.nodes[j]['state']==1])
            infectant_surface = 0
            for idx in range(len(infective_neighbors)):
                infectant_surface += edge_lengths.get((i,infective_neighbors[idx]))

            
            # Total node rate from number of neighbors that are infectious:
            if use_VDcontact:
                lambda_i = beta * np.sum([state==1 for state in neighbor_states])* (1 + 1*infectant_surface/(total_contact_surface+1e-12)) 
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

def next_event(G):
    
    '''Input: the network G.
    Output: selected reaction channel, i_selected, and the waiting time until the event, tau.'''
     
    # Get waiting times from G:
        
    node_indices  = list(nx.get_node_attributes(G, 'tau'))
    waiting_times = list(nx.get_node_attributes(G, 'tau').values()) 
        
    # Select reaction with min waiting time:
    tau = np.min(waiting_times)
    
    i_selected = np.where(waiting_times == tau)[0][0]

    return(i_selected, tau)

########################################################

def update_states(G, X, tau, i_selected, adj_matrix, edge_lengths):
    # global global_iteration_count
    # State counts:
    global PROPENS
    S, I, R = X
    
    # Get node indices for active reaction channels from G:
    node_indices = list(nx.get_node_attributes(G, 'tau'))

    # Get node id corresponding to reaction channel i_0
    TRIG_node = node_indices[i_selected] # next node triggered/event'ed

    # == Update waiting times for remaining node =====
    
    node_indices.pop(i_selected) # Remove this index from group
    
    for j in node_indices:
        G.nodes[j]['tau'] -= tau

    # ===== Update network state: =====
    
    # i_node TRIGGERED event
    prev_state = G.nodes[TRIG_node]['state']


    if prev_state==0: # S-> I
 
        S -= 1
        I += 1

        # +1 (S -> I): 
        G.nodes[TRIG_node]['state'] = 1

        # n*Beta --> alpha
        G.nodes[TRIG_node]['lambda'] = alpha
        
        # Draw new waiting time for i_selected::i_node :
        u_i = rg.random()
        G.nodes[TRIG_node]['tau'] = -np.log(1. - u_i) / alpha

        # Update propensities of i's NEIGHBOURS:
        susceptible_neighbors = np.array([j for j in G.neighbors(TRIG_node) if G.nodes[j]['state']==0])


        # (TRIG_node) can infect (susceptible_neighbors)

        neighbors = np.array([j for j in G.neighbors(TRIG_node)])
        total_contact_surface = 0
        for idx in range(len(neighbors)):
            total_contact_surface += edge_lengths.get((TRIG_node,neighbors[idx]))

        if len(susceptible_neighbors)>0:
            for j in susceptible_neighbors: 
                # Update j's propensity:
                common_surface = edge_lengths.get((TRIG_node , j))
                if use_VDcontact:
                    G.nodes[j]['lambda'] += (betaMODIF )* ( 1+ 1*common_surface/total_contact_surface)
                    # PROPENS.append(  1+ 1*common_surface/total_contact_surface)
                    PROPENS.append(  G.nodes[j]['lambda'])
                    
                    
                    
                else:
                    G.nodes[j]['lambda'] += betaMODIF 
                    PROPENS.append(  G.nodes[j]['lambda'])
                   
                # Draw new waiting time for j:
                u_j = rg.random()
                G.nodes[j]['tau'] = -np.log(1. - u_j) / G.nodes[j]['lambda']

    # I->R
    else:
        I -= 1
        R += 1

        # Update state of node in graph: 
        G.nodes[TRIG_node]['state'] = 2

        # Remove i from reaction channels:
        G.nodes[TRIG_node]['lambda'] = 0.
        del G.nodes[TRIG_node]['tau']

        # Update propensities of i's neighbors:
        susceptible_neighbors = np.array([j for j in G.neighbors(TRIG_node) if G.nodes[j]['state']==0])

        if len(susceptible_neighbors)>0:
            for j in susceptible_neighbors:            
                # Update j's propensity:
                G.nodes[j]['lambda'] -= betaMODIF
                                   
                # If node propensity is zero remove channel:
                if np.isclose(G.nodes[j]['lambda'], 0.):
                    del G.nodes[j]['tau']
                
                # Else, draw new waiting time for j:
                else:
                    u_j = rg.random()
                    G.nodes[j]['tau'] = -np.log(1. - u_j) / G.nodes[j]['lambda']
                    
    ## =========  VISUALIZE  =======================
    if visual_time:
        global counter
        
        SAVINGFIG  = plt.figure(figsize=(8, 6)) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # counter = 0
        
        node_states = nx.get_node_attributes(G, 'state')
        status = list(node_states.values())
        idxS = list(np.nonzero(status==np.int64(0)))[0]
        idxI = np.nonzero(status==np.int64(1))[0]
        idxR = np.nonzero(status==np.int64(2))[0]
        
        patches = []
        colors  = []
        for ii, ptr in enumerate(boundedpols):
            x, y = ptr.exterior.xy
            polygon = pltPolygon(np.column_stack([x, y]), closed=True)
            patches.append(polygon)
            if ii in idxS:
                colors.append("cornflowerblue")
            elif ii in idxI:
                colors.append("r")
            else: 
                colors.append("lime")
        collection = PatchCollection(patches, edgecolor='k', linewidth=0.5, facecolor=colors)
        # fig, ax = plt.subplots(1)
        ax = SAVINGFIG.add_subplot(1, 1, 1)
        ax.add_collection(collection)
        ax.autoscale()
        ax.axis('equal')
        if visual_SaveVideo:
            if counter > 1485:
                save_figure(SAVINGFIG)
        plt.show()
        counter += 1
        
    return([S, I, R])
    


# Function to save the current figure
def save_figure(SAVINGFIG):
    
    # Define directory to save
    # directory = "Model11images"
    directory = "Images"
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filename = os.path.join(directory, f"image_{counter:04d}.png")
    SAVINGFIG.savefig(filename, dpi=100)    
    
    
    


########################################################


def FirstReac_SIR(G, beta, alpha, T, adj_matrix, edge_lengths):
        
    # Initialization
    node_states = nx.get_node_attributes(G, 'state')
    
    Sinf = []
    N = len(node_states)
    S = sum(X==0 for X in node_states.values())
    I = sum(X==1 for X in node_states.values())
    R = N - S - I 
    
    # Intialise node propensities:
    calc_propensities(G, beta, alpha, adj_matrix, edge_lengths)    

    # Set time t=0:
    t = 0

    # Vector to save temporal evolution of states over time: 
    X_t = [] 
    X_t.append([t, S, I, R])
    
    #  Keep producing events until t >= T: 
    while t < T:
        # Check if no waiting times are left
        # => a.k.a., no more reactions can happen
        if len(nx.get_node_attributes(G,'tau')) == 0:
            X_t.append([T, S, I, R])
            break
        
        # First reaction event sampling ++++++++++++++++++++
        i, tau = next_event(G) 
        
        # Advance wall time:
        t += tau

        # Perform update:++++++++++++++++++++++++++++++++++++++
        [S, I, R] = update_states(G, [S, I, R], tau, i, adj_matrix, edge_lengths)
        
        # Store results: +++++++++++++++++++++
        X_t.append([t, S, I, R])

    Sinf = S
    return(np.array(X_t).transpose(), Sinf) 

########################################################

# Fragments from stackexchange and github (links in PDF document)
def voronoi_polygons(voronoi, diameter):
    
    centroid = voronoi.points.mean(axis=0)

    # Map from Voronoi point index to list of unit vectors in the 
    # direction of infinite ridges
    # Start at the Voronoi point and neighbours
    
    
    ridge_direction = defaultdict(list)
    for (p, q), rv in zip(voronoi.ridge_points, voronoi.ridge_vertices):
        u, v = sorted(rv)
        if u == -1:
            # Infinite ridge starting at ridge point with index v,
            # equidistant from input points with indexes p and q.
            
            t = voronoi.points[q] - voronoi.points[p]       # tangentential
            n = np.array([-t[1], t[0]]) / np.linalg.norm(t) # normal
            
            
            midpoint = voronoi.points[[p, q]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - centroid, n)) * n
            ridge_direction[p, v].append(direction)
            ridge_direction[q, v].append(direction)

    for i, r in enumerate(voronoi.point_region):
        region = voronoi.regions[r]
        if -1 not in region:
            # Finite region:
            yield Polygon(voronoi.vertices[region])
            continue
        # Infinite region:
        inf = region.index(-1)              # Index of vertex at infinity.
        j = region[(inf - 1) % len(region)] # Index of previous vertex.
        k = region[(inf + 1) % len(region)] # Index of next vertex.
        if j == k:
            # Region has one Voronoi vertex with two ridges.
            dir_j, dir_k = ridge_direction[i, j]
        else:
            # Region has two Voronoi vertices, each with one ridge.
            dir_j, = ridge_direction[i, j]
            dir_k, = ridge_direction[i, k]

        # Length of ridges needed for the extra edge to lie at least
        # 'diameter' away from all Voronoi vertices.
        length = 2 * diameter / np.linalg.norm(dir_j + dir_k)

        # Polygon consists of finite part plus an extra edge.
        finite_part = voronoi.vertices[region[inf + 1:] + region[:inf]]
        extra_edge = [voronoi.vertices[j] + dir_j * length,
                      voronoi.vertices[k] + dir_k * length]
        yield Polygon(np.concatenate((finite_part, extra_edge)))

########################################################

def compute_adjacency_matrix(positions, R_threshold):
    N = len(positions)
    distances = squareform(pdist(positions))
    adj_matrix = np.zeros((N, N), dtype=bool)

    for i in range(N):
        for j in range(i + 1, N):
            if distances[i, j] <= R_threshold:
                adj_matrix[i, j] = True
                adj_matrix[j, i] = True

    return adj_matrix


########################################################


def compute_edge_lengths(positions, boundedpols, adj_matrix):
    N = len(positions)
    edge_lengths = {}

    for i in range(N):
        polygon = boundedpols[i]
        exterior = polygon.exterior

        # Get neighbors of node i-th based on adjacency matrix
        neighbors = np.where(adj_matrix[i])[0]

        for neighbor in neighbors:
            
            try:
                edge_length, touchflag = find_shared_edge(polygon, boundedpols[neighbor], i, neighbor)
                if touchflag:
                    edge_lengths[(i, neighbor)] = edge_length
            except:
                try:
                    myG.remove_edge(i, neighbor)
                except:
                    myG.remove_edge(neighbor,i)
                adj_matrix[i,neighbor]  = False
                adj_matrix[neighbor, i] = False
                
    
    return edge_lengths, adj_matrix


########################################################
def find_shared_edge(polygon1, polygon2, i, neighbor):
    
    # Impose condition de voisinage
    touchflag = 0
    # edgelen   = 0
    if polygon1.touches(polygon2):
        touchflag = 1
        
        
        # They don't necessarily have the same number of edges
        co1  = np.array(polygon1.boundary.coords.xy) # OR polygon1.boundary.coords.xy       
        co1x = np.array(co1[0])
        co1y = np.array(co1[1])
        
         
        co2  = np.array(polygon2.boundary.coords.xy)
        co2x = np.array(co2[0])
        co2y = np.array(co2[1])
        
        xMX = np.zeros((len(co2x),len(co1x)))
        
        for idx in range(len(co1x)):
            x1 = co1x[idx]
            for jdx in range(len(co2x)):
                x2 = co2x[jdx]
                xMX[jdx,idx]  = np.abs(x1-x2)
                
        yMX = np.zeros((len(co2y),len(co1y)))
        for idx in range(len(co1y)):
            y1 = co1y[idx]
            for jdx in range(len(co2y)):
                y2 = co2y[jdx]
                yMX[jdx,idx] = np.abs(y1-y2)
          
                
        thr = 1e-3    # Threshold for spatial identity of VD corners
        boolx = xMX<thr
        booly = yMX<thr
        coin_idx = boolx&booly
        coin_idx = np.sum(coin_idx,axis=0)
        coin_idx = list(np.where(coin_idx>0))
        exes = co1x[coin_idx]
        ys   = co1y[coin_idx]
        edgelen = np.sqrt(  (exes[0][0]-exes[0][1])**2 + (ys[0][0]-ys[0][1])**2)
    
        
    return(edgelen, touchflag)

########################################################

def do_tessellation(Lx_, Ly_, positions_, R_):
    
    boundary = np.array([[0, 0], [Lx_,0], [Lx_,Ly_], [0,Ly_]])
    x_, y_ = boundary.T
    
    if visual_VDtessellation:
        plt.figure(1)
        plt.clf()
        plt.xlim(round(x_.min() - 1), round(x_.max() + 1))
        plt.ylim(round(y_.min() - 1), round(y_.max() + 1))
        plt.plot(*positions_.T, 'b.')

    boundary_polygon = Polygon(boundary)
    boundedpols      = []
    diameter_        = np.linalg.norm(boundary.ptp(axis=0))

    for p_ in voronoi_polygons(Voronoi(positions_), diameter_):
        x, y = zip(*p_.intersection(boundary_polygon).exterior.coords)
        
        if visual_VDtessellation:
            plt.plot(x, y, 'r-')
        
        tmpverts = []
        for x_,y_ in zip(np.array(x).ravel(),np.array(y).ravel()):
            tmpverts.append([x_,y_])
            
        # Create a list of VD polygons bounded by Domain Boundary
        boundedpols.append( Polygon(tmpverts) )

    if visual_VDtessellation:
        ax = plt.gca()
        ax.axis('equal')
        plt.xlabel(r"Domaine [m]")
        plt.ylabel(r"Domaine [m]")
        plt.show()

    adj_matrix_ = compute_adjacency_matrix(positions_, R_)

    # Compute shared adjacent edge lengths 
    # Correct adjacency matrix if needed based on direct neighbor contact
    edge_lengthsOUT, adj_matrixOUT = compute_edge_lengths(positions_, boundedpols, adj_matrix_)
    
    return edge_lengthsOUT, adj_matrixOUT, boundedpols

 
#####################################################################
#####################################################################
#               End of Function Definitions                         #
#####################################################################
#####################################################################


#####################################################################
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
#                         Custom Parameters                         #
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
#####################################################################

counter = 0
PROPENS = []
    
# +++++++++++++++++++++ SIMUALTION PARAMETERS: +++++++++++++++++++++++++++++
seed = 42                     # Set seed of PRNG state 
rg   = Generator(PCG64(seed))  # Initialize bit generator (here PCG64) with seed

beta   = np.float64(0.1)   # Infection ratio

alpha  = np.float64(0.03)  # Recovery rate

N      = np.int32(1220)    # Number of nodes
T      = np.float64(300)   # Time of simulation

NumSimuls   = np.int32(20)

visual_time           = 0 # Visualize network infection in time steps
visual_SaveVideo      = 0 # [CAREFULL!!!], it takes hours
visual_NTWX           = 1 # View networkx object
visual_VDtessellation = 1 # See Voronoi tessellation of domain
use_VDcontact         = 0 # Use contact surface/edge for transmission propensity
visual_AllGroups      = 0 # See S-I-R (1), only (I) if (0), (-1) for none 


## +++++++++++++++++++++ PARMETERS DOMAIN GRID +++++++++++++++++++++

Lx = np.float64(40)  
Ly = np.float64(40)  
A  = Lx*Ly # Area
dh = np.sqrt(A/N) # Regularized grid step

Rthr      = 3*dh  *1.045*np.sqrt(2) # R-distance for node-neighbors
Excent    = 10*dh # Randmzd. particle movement from homogeneous distribution post.
BoundDist = 0.9 # Distance of particles to boundaries of domain

Rthr   = 1.045*np.sqrt(2)*dh
Excent = 0.001*dh


#####################################################################
#####################################################################
#                    End of  Custom Parameters                      #
#####################################################################
#####################################################################


# +++++++++++++++++++++ Build Grid +++++++++++++++++++++

xax = np.linspace( 0+BoundDist, Lx-BoundDist, np.int32(np.floor(Lx/dh)), endpoint=True  )
yax = np.linspace( 0+BoundDist, Ly-BoundDist, np.int32(np.floor(Ly/dh)), endpoint=True  )

# Readjust N
N = len(xax)*len(yax)

Xax,Yax = np.meshgrid(xax,yax)

Xax = np.float64(np.reshape(Xax, (N,1)))
Yax = np.float64(np.reshape(Yax, (N,1)))

Xax = Xax + rg.random((N,1))*Excent*dh
Yax = Yax + rg.random((N,1))*Excent*dh

Lx = np.max(Xax)+BoundDist
Ly = np.max(Yax)+BoundDist

# +++++++++++++++++++++ Generate reference Graph +++++++++++++++++++++
myG = nx.Graph()

position_dict = dict()
for i, (x, y) in enumerate(zip(Xax.ravel(), Yax.ravel())):
    position_dict[i] = np.array([x, y], dtype=np.float64)

myG.add_nodes_from(position_dict)


# +++++++++++++++++++++ SET EDGES based on R-distance +++++++++++++++++++++
positions = np.array(list(position_dict.values()))
distances = squareform(pdist(positions))
num_nodes = len(position_dict)
for i in range(num_nodes):
    for j in range(i+1, num_nodes):
        if distances[i, j] <= Rthr:
            # myG.add_edge(i, j)
            myG.add_edge(i, j, minlen=distances[i,j])


# +++++++++++++++++++++ Tessellate domain +++++++++++++++++++++
edge_lengths, adj_matrix, boundedpols = do_tessellation(Lx, Ly, positions, Rthr)

if visual_NTWX:
    plt.figure(2)
    plt.clf()
    nx.draw(myG, pos=position_dict, with_labels=True)
    ax = plt.gca()
    ax.axis('equal')
    plt.title("Réseau arpès correction")
    plt.show()

kmean = np.mean(np.sum(adj_matrix,axis=1))
print('(AdjMX) kmean: ', kmean)


#  ++++++++++++++++++++ Rectify Beta for conservation ++++++++++++++++++++++

degs = np.array(myG.degree())
degs = degs[:,1]

betaMODIF = beta
betaMODIF = 0.88*beta * np.mean(degs)/np.max(degs) 


# +++++++++++++++++++++ Do Simulations +++++++++++++++++++++

# =====================  SIR ODE's =================================

tdet, Sdet, Idet, Rdet = Model01_func( T, beta, alpha, N)

if 0:
    plt.figure()
    plt.plot(tdet, Sdet,'b')
    plt.plot(tdet, Idet,'r')
    plt.plot(tdet, Rdet,'g')


# ===================  SIR Stockastic ============================
X_array = []

Sinf = np.zeros(NumSimuls)

flagonce = 1
init  = time.time()
for IdxSimul in range(NumSimuls):
    
    G = myG.copy()

    I_nodes = [int(rg.random()*N)]
    # I_nodes = int(N/2)
    I_nodes = [0]
    R_nodes = []

    states = np.zeros(N, dtype=int) # Default to State=0 everywhere
    states[I_nodes] = 1
    states[R_nodes] = 2

    nx.set_node_attributes(G, 'state', 0)
    for i,state in enumerate(states):
        G.nodes[i]['state'] = state


    X_t, Sinf[IdxSimul] = FirstReac_SIR(G, betaMODIF, alpha, T, adj_matrix, edge_lengths)
    X_array.append(X_t)

    if flagonce:
        total = (time.time() - init)
        print('First Run exec-time: ' + str(total) + ' [s]')
        flagonce = 0


total = (time.time() - init)/NumSimuls
print('Total avrg. 1Run exec-time: ' + str(total) + ' [s]')

plt.figure(3)
plt.clf()
plt.plot(PROPENS, 'o')
plt.grid()
plt.title(r"Beta Basic: {:.3f}".format(beta))
plt.plot((0,len(PROPENS)),(beta,beta),'--r')


# +++++++++++++++++++++ Plot Simulation Results +++++++++++++++++++++

IMAX  = []
TIMAX = []
Sdata = []
Idata = []
Rdata = []
Tvec  = []

# TIME DOMAIN Simulation Series +++++++++++++++++++++++++

if visual_AllGroups!=-1:
    fig = plt.figure(5)
    plt.clf()
    ax  = plt.subplot()
    for X_t in X_array:
        
        Tvec.append(X_t[0:1,:])
        Sdata.append(np.array(X_t[1:2,:]) )
        Idata.append(X_t[2:3,:])
        Rdata.append(X_t[3:4,:])
        
        if visual_AllGroups==1: # Plot All data sets
            tImax = np.argmax(X_t[2,:])
            tvec = X_t[0,:]
            thisT = tvec[tImax]
            TIMAX.append(thisT)
            IMAX.append(np.max(X_t[2,:]))
            colors  = ['b','r','g']
            [ax.plot(X_t[0,:],X_,'o',c=c_,alpha=1/NumSimuls) for X_, c_ in zip(X_t[1:,:], colors)] 
            legends = [r'$N_S$',r'$N_I$',r'$N_R$']
            ax.legend(legends)
        
        elif visual_AllGroups==0: # Plot Only infected I(t)
            tImax = np.argmax(X_t[2,:])
            tvec = X_t[0,:]
            thisT = tvec[tImax]
            TIMAX.append(thisT)
            IMAX.append(np.max(X_t[2,:]))
            colors  = ['r']
            [ax.plot(X_t[0,:], X_ ,'o',c='r',alpha=1/NumSimuls) for X_, c_ in zip(X_t[2:3,:],colors)] 
            
            # plt.plot([thisT,thisT],[0,N],':k', alpha=0.125) # <<<<<<<<< Imax
            legends = [r'$N_I$']     
            ax.legend(legends)
    print('T(Imax) moyen: ', np.mean(TIMAX))   
    print('Imax moyen: ', np.mean(IMAX), " (Imax/N: ",np.mean(IMAX)/N,")" )

     
    plt.plot(tdet, Idet, 'k')   
    plt.xlabel(r"Temps [jours]")
    plt.ylabel(r"Numéro de cas")
    degs = np.array(myG.degree())
    degs = degs[:,1]
    plt.title(r"Solutions SIR Gillespie. Degré moyen: {:.2f}".format(np.mean(degs)))
    plt.xlim((0,T))
    plt.show()


# plt.figure(4)
# plt.clf()
# plt.plot(degs, 'o')
# plt.title(r"Degree")


print('PROPENS. Mean: ', np.mean(PROPENS), '. STD: ', np.std(PROPENS))
print('PropensMean/Beta: ', np.mean(PROPENS)/beta)
# 29/05/2024

vpAdj, VPAdj = np.linalg.eig(adj_matrix)


lapl = nx.laplacian_matrix(myG)
lapl = lapl.toarray()
vplapl, VPlapl = np.linalg.eig(lapl)
vplapl_sort = np.sort(vplapl)

SG = np.max(vplapl) - vplapl_sort[1]
print('Spectral gap: ', SG)

# plt.pcolor( np.log( ) )
plt.figure(6)
plt.clf()
plt.spy(adj_matrix)
plt.title('adj MX')

plt.figure(7)
plt.clf()
plt.pcolor(lapl)
plt.title(r"lapl")


plt.figure(8)
plt.clf()
plt.plot(np.real(vpAdj),'o')
plt.plot(np.real(vplapl),'o')
plt.legend(['adj','lapl'])
plt.grid()

plt.figure(9)
plt.clf()
plt.pcolor(abs(VPAdj))
plt.title(r"vec prop ADJ MX")
plt.colorbar()

plt.figure(10)
plt.clf()
plt.pcolor(abs(VPlapl))
plt.title(r"vec prop Lapl")
plt.colorbar()
plt.show()

# %%


Len = len(Tvec[0][0])
for idx in range(NumSimuls):
    tmp = len(Tvec[idx][0])
    if tmp>Len:
        Len=tmp
        
T =  np.zeros((NumSimuls,Len))
S =  np.zeros((NumSimuls,Len))
I =  np.zeros((NumSimuls,Len))
R =  np.zeros((NumSimuls,Len))
        
for idx in range(NumSimuls):
    T[idx,range(len(Tvec[idx][0]))]  = Tvec[idx][0]
    S[idx,range(len(Sdata[idx][0]))] = Sdata[idx][0]
    I[idx,range(len(Idata[idx][0]))] = Idata[idx][0]
    R[idx,range(len(Rdata[idx][0]))] = Rdata[idx][0]
    
    

# ESPACE DE PHASE  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

plt.figure()

fits = np.nan*np.ones((NumSimuls,2))
for idx in range(NumSimuls):
    idxmax = np.argmax(I[idx,:])
    idxstop = np.where(I[idx,0:idxmax] < 0.5 * np.max(I[idx,:]) )
    
    try:
        maskfit = range(idxstop[0][-1])
        fits[idx,:] = np.polyfit( S[idx,maskfit] , I[idx,maskfit] ,  1)  
        
        maskplot = range(idxmax)
        plt.plot(S[idx,:], I[idx,:], 'ob')
        pp = np.poly1d(fits[idx,:])
        plt.plot( S[idx,maskfit] , pp(S[idx,maskfit]), '--r'  )
    except:
        continue
            
plt.title(r"Espace de phase")    
plt.xlabel(r"S(t)")
plt.ylabel(r"I(t)")

plt.show()



Stot = np.array([])
Itot = np.array([])
Ttot = np.array([])


for idx in range(NumSimuls):
    Stot = np.append(Stot, (Sdata[idx]))
    Itot = np.append(Itot, np.squeeze(Idata[idx]))
    Ttot = np.append(Ttot, np.squeeze(Tvec[idx]))
    

# I sur S =================================

IsurS  = I/(S+1e-12)
IDXsel = 0
selec  = IsurS[IDXsel,:]
idx_zero = np.where(selec==0)
selec[idx_zero] = selec[0]


T[IDXsel,0] = 0.5

plt.figure()
plt.plot(   np.log( T[IDXsel,:]),  np.log(selec),'o')


plt.ylim((-7.5,7.5))
plt.grid()
plt.xlabel(r" Temps log")
plt.ylabel(r"I/S log")
plt.title(r"I sur S")


# EIGENVALUES: adjMX et Laplacian(adjMX) ================================

vpAdj, VPAdj = np.linalg.eig(adj_matrix)
plt.figure()

plt.plot(vpAdj, 'bo')
plt.title('Valeurs propres')

lapl = nx.laplacian_matrix(myG)
lapl = lapl.toarray()
vpLapl, VPLapl = np.linalg.eig(lapl)


plt.plot(vpLapl,'ro')
plt.legend([r'Adj MX',r'Laplacien de AdjMX'])


# EIGENVECS Adj MX ====================================
plt.figure()

vpreal = np.real(VPAdj)
vpimag = np.imag(VPAdj)

vecdata = np.abs(VPAdj)

plt.pcolor( np.log(vecdata/np.mean(vecdata) + 1e-6) )
plt.colorbar()
plt.clim((-2, 2))
plt.title(r"Vecteurs propres Matrice Adjacence")


# EIGENVECS Lapl(Adj) MX ====================================
plt.figure()

vpreal = np.real(vpLapl)
vpimag = np.imag(vpLapl)

vecdata = np.abs(VPLapl)

plt.pcolor( np.log(vecdata/np.mean(vecdata) + 1e-6) )
plt.colorbar()
plt.clim((-4, 2))
plt.title(r"Vects. propres Laplacien de l'adjacence")


# QUIVER EIGENVECS ===================================================
if 0:
    plt.figure() 
    
    xmesh, ymesh = np.meshgrid(Xax, Yax)
    
    U = np.abs(vpreal + vpimag*j) * np.cos( np.arctan(vpimag/vpreal) )
    V = np.abs(vpreal + vpimag*j) * np.sin( np.arctan(vpimag/vpreal) )
    plt.quiver ( xmesh, ymesh, U, V ) 
    plt.xlim((30, 80))
    plt.ylim((-20, 60))


# +++++++++++++++++++++ Some statistics +++++++++++++++++++++

meandeg = np.mean(degs)
Ro_estim = ((np.var(degs)+(np.mean(degs))**2))/np.mean(degs) - 1
print('(NTWRK) Av Node k-degree: ', meandeg)
print('Ro estime: ', Ro_estim)
print('Fitted slope I/S: ', np.nanmean(fits[:,0]))

S_o = N-1

betw = nx.betweenness_centrality(myG)
betw = list(betw.values())
print("\nBETWEENNESS CENTRALITY. Mean: {0:.4E}".format( np.mean(betw)),". STD: {0:.4E}".format(np.std(betw))  )

clu = nx.clustering(myG) 
clu = list(clu.values())

print('CLUSTERING. Mean: ', nx.average_clustering(myG), '. STD: ', np.std(clu))


print('Imax: ', np.max(I), '. Imax/N:', np.max(I)/N,'\n')

vpLapl_sort = np.sort(vpLapl)
SG = np.max(vpLapl) - vpLapl_sort[-2]
print('Spectral gap: ', SG)

###########################
#    NETWORK ANALYSIS     #
###########################

ISCONNECTED = nx.is_connected(myG)

# Clustering = the proba of i's neighbors to be linked
clust = nx.clustering(myG)
plt.figure()
plt.plot(clust.values(),'o')
plt.title('Réseau: clustering coefficients')

np.mean(list(clust.values()))
avclust = nx.average_clustering(myG)
betw = nx.betweenness_centrality(myG)
betw = list(betw.values())

plt.figure() 
plt.plot(list(betw),'o')
plt.title('Réseau: betweenness centrality')


# Propensity history ====================================
plt.figure()
plt.plot(PROPENS/beta,'o')
plt.ylabel(r"Propensité/beta")
plt.title(r"Propensités par rapport à $\beta$")

# Adj Matrix ==========================
plt.figure()
plt.spy(adj_matrix)
plt.title(r"Matrice Adjacence")



# NETWORK'S DEGREE DISTRIBUTION PLOT =============================================

degrees = dict(myG.degree())
degree_values       = list(degrees.values())
degree_distribution = nx.degree_histogram(myG)

# plt.bar(range(len(degree_distribution)), degree_distribution, align='center')
xx = np.double(list(range(len(degree_distribution))))
yy = degree_distribution 

idx_zero = np.where( np.double(yy) == 0  )
# yy[np.squeeze(idx_zero)] = 1e-3# np.nan

plt.figure()
np.seterr(divide = 'ignore') 
# numpy.seterr(divide = 'warn') 
plt.plot(np.log10(xx), np.log10(yy/np.sum(yy)),'--o')


kvec    = np.linspace(0,15, 16).astype(np.int32)
kmean   = np.mean(degree_values)
poisson = np.exp(-kmean) * (kmean**kvec) / scipy.special.factorial(kvec)

plt.plot( np.log10(kvec), np.log10(poisson),'r'  )

plt.legend([r'Réseau generé',r'Distribution Poisson'])
plt.xlabel('Degré $k$ (log10)')
plt.ylabel('$p_k$ (log 10)')
plt.title(r"Distribution de degré pour N={}".format(N) )
plt.grid()

plt.gca().set_xticks([0, 0.5, 1])
plt.gca().set_xticklabels([r'$10^{0}$', r'$10^{0.5}$', r'$10^{1}$'])
plt.gca().set_yticks([-3.,  -2., -1., -0.5])
plt.gca().set_yticklabels([r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$',r'$10^{-0.5}$'])
plt.show()
