#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from numba import njit#, prange 


########### FONCTIONS ###############

@njit
def deriv(t_loc, y_loc,params):
        
    beta  = params[0]
    gamma = params[1]
    dydt = np.zeros(3,dtype=np.float64)
    
    dydt[0] = -beta * y_loc[1]*y_loc[0]             # -beta * I(t) * S(t)
    dydt[1] =  y_loc[1] * (beta*y_loc[0] - gamma)   # beta*Y(t)*S(t) - gamma*I(t)
    dydt[2] = gamma * y_loc[1]                      # gamma*I(t)
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
    
    S = np.zeros(NumPas,dtype=np.float64)
    I = np.zeros(NumPas,dtype=np.float64)
    R = np.zeros(NumPas,dtype=np.float64)
    
    idx = np.int32(0)
    for idx in range(NumPas-1):
        
        yk = rk4(tvec[idx], dt, yk, deriv,PARAMS)
        S[idx+1] = yk[0]
        I[idx+1] = yk[1]
        R[idx+1] = yk[2]
        
    return S,I,R


############ PARAMETRES #############
dt      = np.float64(0.05)  # jours
Tinit   = np.float64(0.)
Tmax    = np.float64(20.)  # jours
PI      = np.float64(np.pi) 


NumPas = np.int32( np.ceil( (Tmax-Tinit)/dt ) + 1 ) 
tvec   = np.linspace(Tinit,Tmax, NumPas).astype(np.float64) 

beta   = np.float64( (50)**(-1) ) # Défaut 7 jours
gamma  = np.float64( (1.)**(-1) ) # Déf. 4 jours 


Ncompart = np.int32(3)

PARAMS = [beta,gamma]
PARAMS = np.float64(PARAMS)

########## INITIALISATIONS ###########


N = np.float64(100.) # Population totale

# Conditions initiales :
    
y0_1 = np.float64(0.9999*N)
y0_2 = np.float64(N-y0_1)
y0_3 = np.float64(0.)

# Allocations mémoire
yk    = np.zeros(Ncompart,dtype=np.float64)
yk[0] = y0_1 
yk[1] = y0_2
yk[2] = y0_3



########## SIMULATION ###########

S,I,R = simul_fonc(tvec, dt, yk, deriv, PARAMS,NumPas)

S[0] = y0_1 
I[0] = y0_2 
R[0] = y0_3



########## ESTIMATIONS ###########

Ro = beta*S[0]/gamma

# Find the inflection point of I as last S>alpha/beta

idxinfl = np.nonzero(S>1.0*gamma/beta)
tinfl   = tvec[idxinfl]
tinfl   = tinfl[-1]

########## Fit Iinit like exp() ###########


idxForFit = np.nonzero(S>1.3*gamma/beta)

rate      = np.polyfit(tvec[idxForFit], np.log(I[idxForFit]), 1)
Iinit     = np.exp(np.polyval(rate, tvec[idxForFit] ))
BetaEstim = (rate[0]+gamma)/N


########## PRINT results ###########

print(f"{Ro    = }")
print(f"{S[-1] = }")
print(f"{I[-1] = }")
print(f"{R[-1] = }")
print(f"{N     = }")
print(r"Imax  = {:.1f}".format(np.max(I)))
print(r"Beta actual        : {:.3f}".format(beta))
print(r"Beta meas from Init: {:.3f}".format(BetaEstim))



########## VISUALISATION RÉSULTATS ###########

plt.figure(figsize=(8.27,5.85))
plt.plot(tvec, S)
plt.plot(tvec, I)
plt.plot(tvec, R)

plt.plot( [tinfl, tinfl] ,[0.,N],'--r')
plt.plot(tvec[idxForFit], Iinit, ':k')
plt.ylim((0,N))




plt.legend(["S(t)","I(t)","R(t)","I$_{max}$","I$_{init}$ $\propto \ exp()$"])
plt.xlabel("Temps  [jours]");
plt.ylabel(r"Nombre de cas")

plt.title(r"S(0)={:.1f}, I(0)={:.2E}, R(0)={:.1f} // Beta={:.2f}, Gamma={:.2f}".format(y0_1, y0_2, y0_3,beta, gamma) )

plt.show()

#  S-I plane =================================
plt.figure(figsize=(8.27,5.85))

idxImax = np.argmax(I)
idxImax = int(0.8*idxImax)
mask = np.arange(0,idxImax,1,dtype=int)
plt.plot(  (S[mask]),   (I[mask]))


rate2     = np.polyfit(S[mask], I[mask], 1)
slopeSI   =  (np.polyval(rate2, S[mask] ))
BetaEstim = (rate[0]+gamma)/N


plt.plot(S[mask], slopeSI, '--r')

plt.xlabel('S(t)')
plt.ylabel('I(t)')
plt.title(r"Espace de phase S-I (segment initial)")
plt.grid()
plt.show()




########################### function g(x) ############

Rp = Ro # np.float64(1.1)
So = np.float64(S[0])
x  = np.arange(0.,800.,1.e-4).astype(np.float64)


g = np.log(So/x) - Rp*(1-x/N)


plt.figure(figsize=(8.27,5.85))
plt.plot(x, g)
plt.xlim((0.*N,1.7*N))
plt.ylim((-0.4,0.4))
plt.plot([N,N],[-3,3],'--r')
plt.plot([N/Ro,N/Ro],[-3,3],'--m')
plt.grid()
plt.legend(["g(x)","N","N/Ro"])
plt.title(r"Fnction g(x). Existence et unicité de solution $S_{\infty}$")
# plt.savefig("PROJ_Model01_foncGx.eps",format="eps",bbox_inches='tight')
plt.show()


