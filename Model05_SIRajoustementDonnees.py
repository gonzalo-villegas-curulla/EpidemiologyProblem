#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from datetime import datetime
from scipy.special import lambertw



######  H1N1 Mexique
daysMex  = np.arange(75,195,1, dtype=np.int32)

casesMex = np.array([2,1,3,2,3,3,4,4,5,7,3,1,2,5,7,4,10,11,13,4,
    4,11,5,7,4,4,4,11,17,26,20,12,26,33,44,107,114,155,227,280,
    318,399,412,305,282,227,212,187,212,237,231,237,176,167,139,152,162,138,117,100,
    83,75,87,98,71,73,78,67,68,69,65,85,55,67,75,71,97,168,126,148,
    152,138,159,186,222,204,257,208,198,193,243,231,225,239,219,199,215,309,346,332,
    328,298,335,330,375,366,291,251,215,242,223,317,305,228,251,207,159,155,214,237])

plt.figure()
plt.plot(daysMex, casesMex,'-o')

# First wave of cases

idxinit = np.array(np.nonzero(daysMex<98))
idxinit = idxinit[0,-1]
mask1 = np.arange( idxinit,  np.argmax(casesMex)+1, 1, dtype=np.int32)
rateMex1    = np.polyfit(daysMex[mask1], np.log(casesMex[mask1]), 1)
Mex1        =  np.exp(np.polyval(rateMex1, daysMex[mask1] ))
plt.plot(daysMex[mask1], Mex1, '--r')
# BetaMex1   = (rate[0]+gamma)/N # We don't know N or gamma 



# Second wave of cases
idxinit = np.array(np.nonzero(daysMex<140))
idxinit = idxinit[0,-1]
idxend  = np.array(np.nonzero(daysMex<180))
idxend = idxend[0,-1]

mask2    = np.arange(idxinit, idxend+1, 1, dtype=np.int32)
rateMex2 = np.polyfit( daysMex[mask2], np.log(casesMex[mask2]), 1 )
Mex2     = np.exp( np.polyval(rateMex2, daysMex[mask2] ))

plt.plot( daysMex[mask2], Mex2, '--r' )

plt.title('H1N1 à Mexique (2003)')
plt.ylabel(r"Nombre de cas infectieux")
plt.xlabel(r"Temps [jours]")



# COVID-19 FRANCE ============================================



data = pd.read_excel("CovidData75000.ods") 

data['jour'] = pd.to_datetime(data['jour'])
jour     = data['jour']
hosp     = data['hosp']
hospconv = data['HospConv']  
rad      = data['rad'] # retournés à domicile
dc       = data['dc'] # décedées

I = hosp.fillna(0) #+ hospconv.fillna(0)
R = rad+dc

dates = [ts.to_pydatetime() for ts in jour]



data = pd.read_excel("COVID-VACC_75000_purged.ods") 

data['jour'] = pd.to_datetime(data['jour'])
jour      = data['jour']
datesvacc = [ts.to_pydatetime() for ts in jour]

dose1     = data['n_dose1']
complet   = data['n_complet']  

plt.figure()

plt.plot(dates, I,'r')
plt.plot(dates[0:-1], np.diff(R)*10, 'g')



plt.plot(datesvacc, 0.1*dose1,'b')
plt.plot(datesvacc, 0.1*complet, 'k')


plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=80))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gcf().autofmt_xdate()

plt.xlabel(r"Date")
plt.ylabel(r"Numéro de personnes")
plt.legend([r"Hospitalisation $\approx$ I(t)", r"10x(Domicile+Décés) $\approx$ R(t)",r"(Vacc. 1 dose)/10",r"(Vacc. Complète)/10"])
plt.ylim((0,np.max(I)*1.1))

plt.show()


# %% Schlickeiser2024 @email tm, td

mask = np.array(range(160))

datesmask = dates[mask[0]:mask[-1]+1]


D  = np.diff( R[mask[0]:mask[-1]+2] )  # Décés
J  = I[mask] # CUMSUM ????  Inféct
J  = list(J)

plt.figure()


plt.plot( datesmask ,  J )
plt.plot( datesmask ,  D*10 )
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=20))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gcf().autofmt_xdate()

Jinf = J[-1]
Dinf = D[-1]

Jdot = np.diff(J)
Ddot = np.diff(D)


t_o = dates[0]
t_m = datesmask[np.argmax(J)] - t_o
t_m = t_m.days
t_d = datesmask[np.argmax(D)] - t_o
t_d = t_d.days



Jinf = 0.1*Jinf
Dinf = 100*Dinf

G = (np.max(Ddot) * Jinf )/( np.max(Jdot)*Dinf * np.exp(1)  )

Wo = lambertw(G*np.log(G), k=0)

U = np.log(G)/Wo

kappa0 = U**(-t_d/(t_d-t_m))

q0 = kappa0*Dinf/Jinf
k0 = kappa0-q0

b0 = kappa0**(t_m/t_d) - kappa0

a0 = -(np.log(kappa0 + b0))/(t_m*b0)# BETA
mu0 = a0*k0 # alpha/gamma:: recvoery rate
phi = a0*q0 # death rate
print('beta ', a0)
print('alpha: ', mu0)




