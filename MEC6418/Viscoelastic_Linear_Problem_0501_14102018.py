# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:13:03 2018

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



def sigma_relaxation_P0502(total_time,n):
    (I,J,K)=I_J_K_4orten_isotropes()
    J_V=tensor4_to_Voigt_tensor2(J)
    K_V=tensor4_to_Voigt_tensor2(K)
    
    def C(t):
        C=( 3.*(10.+2.*exp(-t)+5.*exp(-t/35.))*J_V + 2.*(4.+ 1.*exp(-t)+3.*exp(-t/35.))*K_V )
        return C
    
    sigma=np.zeros((6,n))
    time=np.zeros((n,1))
    epsilon0=np.zeros((6,1))
    epsilon0[0]=0.01
    delta_t=total_time/n
    for i in range(0,n):
        t=i*delta_t
        time[i]=t
        Ct=C(t)
        for j in range(0,6):
            s=np.matmul(Ct,epsilon0)
            sigma[j,i]=s[j]
            
    return time, sigma

    
(time_10, sigma_10)=sigma_relaxation_P0502(100,10)
(time_100, sigma_100)=sigma_relaxation_P0502(100,100)
(time_1000, sigma_1000)=sigma_relaxation_P0502(100,1000)


plt.plot(time_10, sigma_10[0,:], 'r-')
plt.plot(time_100, sigma_100[0,:], 'b-')
plt.plot(time_1000, sigma_1000[0,:], 'g-')
plt.xlabel('time [second]')
plt.ylabel('stress [GPa]')
plt.title('Relaxation test')
plt.legend(('s11 n=10', 's11 n=100', 's11 n=1000'),
           shadow=True, loc=(0.6, 0.7), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([0, 100, 0, 0.4])
plt.show()



k=[10.,2.,5.]
mu=[4.,1.,3.]
w=[1.,1/35.]
excel_file='05-01-epsilon.xls'
D=6
#relaxation_Euler_implicite(excel_file,k,mu,w,D)


C=Cv_relaxation_test(k,mu)
(B,L1,L2_matrix,L3,N,D)=B_L1_L2_L3(C,w)


(t,sigma_euler)=relaxation_Euler_implicite(excel_file,k,mu,w,D) 
(t,sigma_crank_nicholson)=relaxation_Crank_Nicholson(excel_file,k,mu,w,D)

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
plt.plot(time_1000, sigma_1000[0,:], 'r-')
plt.plot(t[0:len(t)], sigma_euler[0,:], 'bx')
plt.plot(t[0:len(t)], sigma_crank_nicholson[0,:], 'g.')
nt=len(t)-1

plt.xlabel('time [seconds]')
plt.ylabel('stress [GPa]')
plt.title('Relaxation test')
plt.legend(('Analytical', 'Euler implicite n=%d'%nt, 'Crank_Nicholson n=%d'%nt),
           shadow=True, loc=(0.4, 0.7), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([0, 100, 0.1, 0.3])
plt.show()










