#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from numba import njit#, prange 

########### FONCTIONS ###############

@njit
def deriv(t, y, params):
        
    q = params[5]
    T = params[6]
    if t>T:
        a = params[0]/q
    else:
        a  = params[0]
    b  = params[1]
    c  = params[2]
    f  = params[3]
    N  = params[4]
    
    dydt  = np.zeros(5,dtype=np.float64)
    
    dydt[0] = -a*y[0]*y[2]/N
    dydt[1] = a*y[0]/N - b*y[1]
    dydt[2] = b*y[1] - c*y[2]
    dydt[3] = f*c*y[2]
    dydt[4] = (1-f)*c*y[2]
    return dydt # @t

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
def simul_fonc(tvec, dt, yk, deriv, PARAMS,NumPas):
    
    S = np.zeros(NumPas-1,dtype=np.float64)
    E = np.zeros(NumPas-1,dtype=np.float64)
    I = np.zeros(NumPas-1,dtype=np.float64)
    R1 = np.zeros(NumPas-1,dtype=np.float64)
    R2 = np.zeros(NumPas-1,dtype=np.float64)
    
    idx = np.int32(0)
    for idx in range(NumPas-1):
        
        yk = rk4(tvec[idx], dt, yk, deriv,PARAMS)
        S[idx]  = yk[0]
        E[idx]  = yk[1]
        I[idx]  = yk[2]
        R1[idx] = yk[3]
        R2[idx] = yk[4]
        
    return S,E,I,R1,R2


############ PARAMETRES #############
dt      = np.float64(0.1)  # jours
Tinit   = np.float64(0.)
Tmax    = np.float64(100)  # jours
PI      = np.float64(np.pi) 


NumPas = np.int32( np.ceil( (Tmax-Tinit)/dt ) + 1 ) 
tvec   = np.linspace(Tinit,Tmax, NumPas).astype(np.float64) 

N = np.int32(1e3)     # Population totale

a = np.float64(1.3)# contact effectif: S->E 
b = np.float64(0.8)# cas latent devient infectieux E->I
c = np.float64(0.08)# retiré R(t) de la chaîne  I-> R_k 


f = np.float64(0.8)   # fraction des retirés avec infct positive R1

q = np.float64(1.3) # fraction de réduction de taux de contact 'a' pour t>T
T = np.float64(50) # temps d'action (mesures de réduction du contact)


Ncompart = np.int32(5)

PARAMS = [a,b,c,f, N, q, T]
PARAMS = np.float64(PARAMS)

########## INITIALISATIONS ###########


# Conditions initiales :
    
y0_1 = np.float64(N*0.7)
y0_2 = np.float64(N*0.3)
y0_3 = np.float64(0.)
y0_4 = np.float64(0.)
y0_5 = np.float64(0.)

# Allocations mémoire
yk    = np.zeros(Ncompart,dtype=np.float64)
yk[0] = y0_1 
yk[1] = y0_2
yk[2] = y0_3
yk[3] = y0_4
yk[4] = y0_5


########## SIMULATION ###########

S,E,I,R1,R2 = simul_fonc(tvec, dt, yk, deriv, PARAMS,NumPas)

########## VISUALISATION RÉSULTATS ###########

plt.figure(figsize=(8.27,5.85))

tvec = tvec[0:-1]

plt.plot(tvec, S)
plt.plot(tvec, E)
plt.plot(tvec, I)
plt.plot(tvec, R1)
plt.plot(tvec, R2)

plt.legend(["S(t)","E(t)","I(t)","R1(t)","R2(t)"])
plt.xlabel("Temps  [jours]");
plt.ylabel(r"Nombre de cas")
plt.title(r"Résultat simulation temporelle pour tous les groupes")



