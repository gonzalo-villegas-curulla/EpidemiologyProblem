#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from numba import njit#, prange 


########### FONCTIONS ###############

@njit
def deriv(t_loc, y,params):
        
    beta  = params[0]
    alpha = params[1]
    N = params[3]
    dydt = np.zeros(3,dtype=np.float64)
    
    dydt[0] = -beta * y[1]*y[0]/N             # -beta * I(t) * S(t)
    dydt[1] =  y[1] * (beta*y[0]/N - alpha)   # beta*I(t)*S(t) - alpha*I(t)
    dydt[2] = alpha * y[1]                      # alpha*I(t)
    return dydt 

@njit
def rk4(x, dx, y, deriv,params):
    ddx = dx/2.    
    d1 = deriv(x,y,params)  
    yp = y + d1*ddx
    d2 = deriv(x+ddx,yp,params)
    yp = y + d2*ddx    
    d3 = deriv(x+ddx,yp,params) 
    yp = y + d3*dx    
    d4 = deriv(x+dx,yp,params)  
    return y + dx*( d1 + 2*d2 + 2*d3 + d4 )/6  

@njit
def simul_fonc(tvec, dt, yk, deriv, PARAMS,NumPas, Ndist):
    
    fmouv = PARAMS[2]
    N = PARAMS[3]
    S = np.zeros((Ndist,NumPas),dtype=np.float64)
    I = np.zeros((Ndist,NumPas),dtype=np.float64)
    R = np.zeros((Ndist,NumPas),dtype=np.float64)
    
        
    for idx in range(NumPas-1): # Time steps
        X = np.random.random()
        if X < 0.33:
            my_fac = np.float64(1.)
        elif np.logical_and(X>=0.33, X<0.66):
            my_fac = np.float64(-1)
        else:
            my_fac = 0 
        
        M = 1*my_fac
        
        
        # First ODE evaluation
        for jdx in range(Ndist): # Loop over the Ndist districts
            
            yk[:,jdx] = rk4(tvec[idx], dt, yk[:,jdx], deriv,PARAMS)
            yk[:,jdx] = np.maximum(yk[:,jdx], 0)
            
        
        # Random transfer            
        yk[1,0] = yk[1,0]-M*fmouv#*N
        yk[1,1] = yk[1,1]+M*fmouv#*N
        
        # Normalize
        if np.sum(yk[:,0])>N:
            yk[:,0] = yk[:,0]/np.sum(yk[:,0])*N
            
        if np.sum(yk[:,1])>N:
            yk[:,1] = yk[:,1]/np.sum(yk[:,1])*N
            
        
        # Second ODE evaluation
        for jdx in range(Ndist): # Loop over the Ndist districts
            
            yk[:,jdx] = rk4(tvec[idx], dt, yk[:,jdx], deriv,PARAMS)
            yk[:,jdx] = np.maximum(yk[:,jdx], 0)
        
        # Normalize
        if np.sum(yk[:,0])>N:
            yk[:,0] = yk[:,0]/np.sum(yk[:,0])*N
            
        if np.sum(yk[:,1])>N:
            yk[:,1] = yk[:,1]/np.sum(yk[:,1])*N
            
        S[:,idx+1] = yk[0,:]
        I[:,idx+1] = yk[1,:]
        R[:,idx+1] = yk[2,:]
        # print(np.sum(yk))
        
    return S,I,R


############ PARAMETRES #############
dt      = np.float64(1e-4)  # jours
Tinit   = np.float64(0.)
Tmax    = np.float64(400.)  # jours
PI      = np.float64(np.pi) 

N = np.float64(1000.) # Population totale


NumPas = np.int32( np.ceil( (Tmax-Tinit)/dt ) + 1 ) 
tvec   = np.linspace(Tinit,Tmax, NumPas).astype(np.float64) 

beta   = np.float64( (500)**(-1) ) 
alpha  = np.float64( (1e-0)**(-1) )

beta   = np.float64( 0.1 ) 
alpha  = np.float64( 0.03 )


 
fmouv  = np.float64(0.3) # fraction infectieux qui bougent à une autre districte


Ncompart   = np.int32(3) # S, I, R = 3
Ndist      = np.int32(2) # Numéro de sub-compart (districtes)




PARAMS = [beta,alpha, fmouv, N]
PARAMS = np.float64(PARAMS)

########## INITIALISATIONS ###########

# Conditions initiales (identiques pour chaque districte):
    
y0_1 = np.float64(N-1)
y0_2 = np.float64(N-y0_1)
y0_3 = np.float64(0.)

# Allocations mémoire
yk      = np.zeros((Ncompart,Ndist),dtype=np.float64)
yk[0,:] = y0_1 
yk[1,:] = y0_2
yk[2,:] = y0_3


########## SIMULATION ###########
Ro = beta/alpha*y0_1

S,I,R = simul_fonc(tvec, dt, yk, deriv, PARAMS,NumPas, Ndist)
for idx in range(Ndist):
    S[idx,0] = y0_1
    I[idx,0] = y0_2
    R[idx,0] = y0_3



########## VISUALISATION RÉSULTATS ###########

plt.figure(figsize=(8.27,5.85))

plt.subplot(3,1,1)
plt.plot(tvec, S[0,:])
plt.plot(tvec, I[0,:])
plt.plot(tvec, R[0,:])
plt.ylim((0,1.2*N))
plt.ylabel(r"No. Cas Compart. 1")

plt.subplot(3,1,2)
plt.plot(tvec, S[1,:])
plt.plot(tvec, I[1,:])
plt.plot(tvec, R[1,:])
plt.ylim((0,1.2*N))
plt.ylabel(r"No. Cas Compart. 2")
plt.xlabel(r"Temps [jours]")

plt.show()

