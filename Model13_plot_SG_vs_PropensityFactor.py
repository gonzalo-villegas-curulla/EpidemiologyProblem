#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:02:18 2024

@author: organ
"""

import numpy as np
import matplotlib.pylab as plt

# N=220 (Increasing Excent:0.001,...,)

SG = np.array([9.881412697527876, 9.881518078157088,9.881038768951022,	9.87186760432617,	10.22759093507363,	13.266262947520675,	11.655603355329365,11.695073150969186,10.545397861835433,	10.627250018664032])
 
factor = np.array([2.060012634238787,2.050568900126423,2.064005069708492,2.0643939393939394,2.0932835820895526,2.1121495327102804,2.0769720101781166,2.0579328505595784,1.944933920704846,1.91304347826087])




plt.figure()
plt.plot( (SG), (factor), 'o')