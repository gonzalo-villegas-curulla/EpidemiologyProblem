#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 00:15:44 2024

@author: organ
"""



def Model01_func( TT, BB, AA, NN):

    import numpy as np
    
    def deriv(t_loc, y_loc,params, N):
            
        beta  = params[0]
        gamma = params[1]
        dydt = np.zeros(3,dtype=np.float64)
        
        dydt[0] = -beta * y_loc[1]*y_loc[0]/N             # -beta * I(t) * S(t)
        dydt[1] =  y_loc[1] * (beta*y_loc[0]/N - gamma)   # beta*Y(t)*S(t) - gamma*I(t)
        dydt[2] = gamma * y_loc[1]                      # gamma*I(t)
        return dydt # @t
    
    def rk4(x, dx, y, deriv,params, N):
        ddx = dx/2.    
        d1 = deriv(x,y,params, N)  
        yp = y + d1*ddx
        d2 = deriv(x+ddx,yp,params, N)
        yp = y + d2*ddx    
        d3 = deriv(x+ddx,yp,params, N) 
        yp = y + d3*dx    
        d4 = deriv(x+dx,yp,params, N)  
        return y + dx*( d1 + 2*d2 + 2*d3 + d4 )/6  
    
    
    def simul_fonc(tvec, dt, yk, deriv, PARAMS,NumPas, N):
        
        S = np.zeros(NumPas,dtype=np.float64)
        I = np.zeros(NumPas,dtype=np.float64)
        R = np.zeros(NumPas,dtype=np.float64)
        
        idx = np.int32(0)
        for idx in range(NumPas-1):
            
            yk = rk4(tvec[idx], dt, yk, deriv,PARAMS, N)
            S[idx+1] = yk[0]
            I[idx+1] = yk[1]
            R[idx+1] = yk[2]
            
        return S,I,R
    
    
    ############ PARAMETRES #############
    dt      = np.float64(0.005)  # jours
    Tinit   = np.float64(0.)
    Tmax    = TT
    PI      = np.float64(np.pi) 
    
    
    NumPas = np.int32( np.ceil( (Tmax-Tinit)/dt ) + 1 ) 
    tvec   = np.linspace(Tinit,Tmax, NumPas).astype(np.float64) 
    
    beta   = BB
    gamma  = AA
    
    Ncompart = np.int32(3)
    
    PARAMS = [beta,gamma]
    PARAMS = np.float64(PARAMS)
    
    ########## INITIALISATIONS ###########
    
    
    N = NN
    
    # Conditions initiales :
        
    y0_1 = np.float64(N-1)
    y0_2 = np.float64(N-y0_1)
    y0_3 = np.float64(0.)
    
    # Allocations mémoire
    yk    = np.zeros(Ncompart,dtype=np.float64)
    yk[0] = y0_1 
    yk[1] = y0_2
    yk[2] = y0_3
    
    
    ########## SIMULATION ###########
    
    S,I,R = simul_fonc(tvec, dt, yk, deriv, PARAMS,NumPas, N)
    
    S[0] = y0_1 
    I[0] = y0_2 
    R[0] = y0_3
    
    return (tvec, S, I, R)

