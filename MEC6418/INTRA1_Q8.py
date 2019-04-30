# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 16:50:22 2018

@author: momoe
"""

from numpy import *
import numpy as np
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter
from scipy.optimize import least_squares
from scipy.optimize import fmin_slsqp
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


from Tensors_Functions_S1_S2_S3_21082018 import *
from Viscoelastic_Linear_Function_S6_Euler_NC_Rel_Flu_14102018 import *
from Viscoelastic_Linear_Function_S5_01_09102018 import *



D=6
k=[23,12,14]
mu=[80,60,70]
landa=[1./35., 10/100.]
excel_file='Intra1_Question8.xls'
(t, epsilon_euler)=fluage_Euler_implicite(excel_file,k,mu,landa,D)
(t, epsilon_Crank_Nicholson)=fluage_Crank_Nicholson(excel_file,k,mu,landa,D)


figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
plt.plot(t[:], epsilon_Crank_Nicholson[0,:], 'g.')
nt=len(t)-1

plt.xlabel('time [seconds]')
plt.ylabel('Strain [-]')
plt.title('fluage unixial')
plt.legend(('Crank_Nicholson n=%d'%nt, 'Crank_Nicholson n=%d'%nt),
           shadow=True, loc=(0.4, 0.2), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([0, 70, 0, 40])
plt.show()





