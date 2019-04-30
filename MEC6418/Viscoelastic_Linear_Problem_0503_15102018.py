# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 15:56:18 2018

@author: momoe
"""

from Tensors_Functions_S1_S2_S3_21082018 import *
from Viscoelastic_Linear_Function_S6_Euler_NC_Rel_Flu_14102018 import *

import numpy as np
import scipy
import scipy.linalg   # SciPy Linear Algebra Library
from numpy.linalg import inv
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from array import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure



def epsilon_fluage_P0503(total_time,n):
    (I,J,K)=I_J_K_4orten_isotropes()
    J_V=tensor4_to_Voigt_tensor2(J)
    K_V=tensor4_to_Voigt_tensor2(K)
    
    def S(t):
        S= (1./3.)*( 3. + 2.*(1-exp(-2.*t)) + 1.*(1.-exp(-t/130.))  )*J_V + (1./2.)*(7.+ 2.*(1.-exp(-2.*t)) + 3.*(1-exp(-t/130)) ) *K_V 
        return S
    
    epsilon=np.zeros((6,n))
    time=np.zeros((n,1))
    sigma0=np.zeros((6,1))
    sigma0[0,0]=10.
    delta_t=total_time/n
    for i in range(0,n):
        t=i*delta_t
        time[i]=t
        St=S(t)
        for j in range(0,6):
            e=np.matmul(St,sigma0)
            epsilon[j,i]=e[j]
            
    return time, epsilon



(time_10, epsilon_10)=epsilon_fluage_P0503(100,10)
(time_100, epsilon_100)=epsilon_fluage_P0503(100,100)
(time_1000, epsilon_1000)=epsilon_fluage_P0503(100,1000)

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 

plt.plot(time_10, epsilon_10[0,:], 'r-')
plt.plot(time_100, epsilon_100[0,:], 'b-')
plt.plot(time_1000, epsilon_1000[0,:], 'g-')
plt.xlabel('time [second]')
plt.ylabel('strain [-]')
plt.title('Creep test')
plt.legend(('e11 n=10', 'e11 n=100', 'e11 n=1000'),
           shadow=True, loc=(0.6, 0.7), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([0, 100, 0., 60])
plt.show()



D=6
k=[3,2,1]
mu=[7,2,3]
landa=[2.,1/130.]
excel_file='05-03-sigma.xls'
(t, epsilon_euler)=fluage_Euler_implicite(excel_file,k,mu,landa,D)
(t, epsilon_Crank_Nicholson)=fluage_Crank_Nicholson(excel_file,k,mu,landa,D)

#S=Sv_fluage_test(k,mu)

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
plt.plot(time_100, epsilon_100[0,:], 'r-')
plt.plot(t[0:len(t)], epsilon_euler[0,:], 'bx')
plt.plot(t[0:len(t)], epsilon_Crank_Nicholson[0,:], 'g.')
nt=len(t)-1

plt.xlabel('time [seconds]')
plt.ylabel('Strain [-]')
plt.title('fluage unixial')
plt.legend(('Analytical', 'Euler implicite n=%d'%nt, 'Crank_Nicholson n=%d'%nt),
           shadow=True, loc=(0.4, 0.2), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([0, 100, 25, 45])
plt.show()


