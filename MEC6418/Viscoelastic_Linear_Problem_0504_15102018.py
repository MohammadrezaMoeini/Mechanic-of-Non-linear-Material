# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 17:12:21 2018

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


excel_file='05-06-histoire-contrainte.xls'

D=6
(t,total_se)=import_time_Dcolumns(excel_file,D)
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 


plt.plot(t, total_se[0,:], 'b.')
nt=len(t)-1

plt.xlabel('time [seconds]')
plt.ylabel('stress [GPa]')
plt.title('histoire-contrainte')
plt.legend(('Stress'),
           shadow=True, loc=(0.4, 0.7), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([0, 80, 0, 30])
plt.show()






k=[3,2,1]
mu=[7,2,3]
landa=[1./5.,1./70.]
(t, epsilon_euler)=fluage_Euler_implicite(excel_file,k,mu,landa,D)
(t, epsilon_Crank_Nicholson)=fluage_Crank_Nicholson(excel_file,k,mu,landa,D)

#S=Sv_fluage_test(k,mu)

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 


plt.plot(t[0:len(t)], epsilon_euler[0,:], 'bx')
plt.plot(t[0:len(t)], epsilon_Crank_Nicholson[0,:], 'g--')
nt=len(t)-1

plt.xlabel('time [seconds]')
plt.ylabel('stress [GPa]')
plt.title('Relaxation test')
plt.legend(('Analytical', 'Euler implicite n=%d'%nt, 'Crank_Nicholson n=%d'%nt),
           shadow=True, loc=(0.6, 0.7), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([0, 80, 0, 70])
plt.show()

S=Sv_fluage_test(k,mu)
(B,A1,A2_matrix,A3,N,D)=B_A1_A2_A3(S,landa)
