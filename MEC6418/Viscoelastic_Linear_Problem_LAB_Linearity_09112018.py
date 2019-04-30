# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 13:28:31 2018

@author: momoe
"""

from numpy import *
from Tensors_Functions_S1_S2_S3_21082018 import *
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

'''Function for calculate S for just sigma=sigma_0*H(t) [where H(t) is Heviside function]'''

fluage_recouvrance_2MPa='fluage-recouvrance-2MPa.xls'
fluage_recouvrance_5MPa='fluage-recouvrance-5MPa.xls'
fluage_recouvrance_10MPa='fluage-recouvrance-10MPa.xls'
fluage_recouvrance_13MPa='fluage-recouvrance-13MPa.xls'
fluage_recouvrance_16MPa='fluage-recouvrance-16MPa.xls'
fluage_recouvrance_20MPa='fluage-recouvrance-20MPa.xls'
fluage_recouvrance_22MPa='fluage-recouvrance-22MPa.xls'
fluage_recouvrance_25MPa='fluage-recouvrance-25MPa.xls'
fluage_recouvrance_30MPa='fluage-recouvrance-30MPa.xls'
fluage_recouvrance_35MPa='fluage-recouvrance-35MPa.xls'



def import_LAB_Date_t_sigma_e11_e22(excel_file):

    df = pd.read_excel(excel_file, sheet_name='Sheet1')
    c0,c1,c2,c3=df
    column_0 = df[c0]
    column_1 = df[c1]
    column_2 = df[c2]
    column_3 = df[c3]

    t=  np.zeros((len(column_0),1))  
    sigma=np.zeros((len(column_0),1))
    e11_exp=np.zeros((len(column_0),1))
    e22_exp=np.zeros((len(column_0),1))
    
    for i in range(0,len(column_0)):
        t[i]=column_0[i]
        sigma[i]=column_1[i]
        e11_exp[i]=column_2[i]
        e22_exp[i]=column_3[i]

    return t, sigma, e11_exp, e22_exp


(t_2MP, sigma_2MP, e11_exp_2MP, e22_exp_2MP)=import_LAB_Date_t_sigma_e11_e22(fluage_recouvrance_2MPa)

(t_5MP, sigma_5MP, e11_exp_5MP, e22_exp_5MP)=import_LAB_Date_t_sigma_e11_e22(fluage_recouvrance_5MPa)

(t_10MP, sigma_10MP, e11_exp_10MP, e22_exp_10MP)=import_LAB_Date_t_sigma_e11_e22(fluage_recouvrance_10MPa)

(t_13MP, sigma_13MP, e11_exp_13MP, e22_exp_13MP)=import_LAB_Date_t_sigma_e11_e22(fluage_recouvrance_13MPa)

(t_16MP, sigma_16MP, e11_exp_16MP, e22_exp_16MP)=import_LAB_Date_t_sigma_e11_e22(fluage_recouvrance_16MPa)

(t_20MP, sigma_20MP, e11_exp_20MP, e22_exp_20MP)=import_LAB_Date_t_sigma_e11_e22(fluage_recouvrance_20MPa)

(t_22MP, sigma_22MP, e11_exp_22MP, e22_exp_22MP)=import_LAB_Date_t_sigma_e11_e22(fluage_recouvrance_22MPa)

(t_25MP, sigma_25MP, e11_exp_25MP, e22_exp_25MP)=import_LAB_Date_t_sigma_e11_e22(fluage_recouvrance_25MPa)

(t_30MP, sigma_30MP, e11_exp_30MP, e22_exp_30MP)=import_LAB_Date_t_sigma_e11_e22(fluage_recouvrance_30MPa)

(t_35MP, sigma_35MP, e11_exp_35MP, e22_exp_35MP)=import_LAB_Date_t_sigma_e11_e22(fluage_recouvrance_35MPa)


       
## =============================================================================
## Drawing Graph - Data
## =============================================================================
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 

plt.plot(t_2MP, sigma_2MP, 'b-')
plt.plot(t_5MP, sigma_5MP, 'b-')
plt.plot(t_10MP, sigma_10MP, 'b-')
plt.plot(t_13MP, sigma_13MP, 'b-')
plt.plot(t_16MP, sigma_16MP, 'b-')
plt.plot(t_20MP, sigma_20MP, 'b-')
plt.plot(t_22MP, sigma_22MP, 'b-')
plt.plot(t_25MP, sigma_25MP, 'b-')
plt.plot(t_30MP, sigma_30MP, 'b-')
plt.plot(t_35MP, sigma_35MP, 'b-')


plt.xlabel('time [seconds]')
plt.ylabel('Poisson Ratio [-]')
plt.title('"Make a comparison')
plt.legend(('sigma_2MP','sigma_5MP','sigma_10MP','sigma_13MP','sigma_16MP','sigma_20MP'
            'sigma_22MP','sigma_25MP','sigma_30MP','sigma_35MP'),
               shadow=True, loc=(0.5, 0.4), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([0, 4000, 0.0, 50])
plt.show()





figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 

plt.plot(t_2MP, e11_exp_2MP, 'b-')
plt.plot(t_5MP, e11_exp_5MP, 'b-')
plt.plot(t_10MP, e11_exp_10MP, 'b-')
plt.plot(t_13MP, e11_exp_13MP, 'b-')
plt.plot(t_16MP, e11_exp_16MP, 'b-')
plt.plot(t_20MP, e11_exp_20MP, 'b-')
plt.plot(t_22MP, e11_exp_22MP, 'b-')
plt.plot(t_25MP, e11_exp_25MP, 'b-')
plt.plot(t_30MP, e11_exp_30MP, 'b-')
plt.plot(t_35MP, e11_exp_35MP, 'b-')


plt.xlabel('time [seconds]')
plt.ylabel('Poisson Ratio [-]')
plt.title('"Make a comparison')
plt.legend(('e11_exp_2MP','e11_exp_5MP','e11_exp_10MP','e11_exp_13MP','e11_exp_16MP','e11_exp_20MP'
            'e11_exp_22MP','e11_exp_25MP','e11_exp_30MP','e11_exp_35MP'),
               shadow=True, loc=(0.5, 0.4), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([0, 4000, 0.0, 0.02])
plt.show()






#
#
#figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
#
#plt.plot(t_5MP, e11_exp_5MP+e11_exp_5MP, 'r--')
#plt.plot(t_10MP, e11_exp_10MP, 'b-')
#
#
#plt.xlabel('time [seconds]')
#plt.ylabel('Poisson Ratio [-]')
#plt.title('"Make a comparison')
#plt.legend(('e11_exp_5MP+e11_exp_5MP','e11_exp_10MP'),
#               shadow=True, loc=(0.5, 0.4), handlelength=1.5, fontsize=12)
#plt.grid(True)
#plt.axis([0, 4000, 0.0, 0.02])
#plt.show()


#
#figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
#
#plt.plot(t_5MP, 1.3*e11_exp_10MP, 'r--')
#plt.plot(t_10MP, e11_exp_13MP, 'b-')
#
#plt.xlabel('time [seconds]')
#plt.ylabel('Poisson Ratio [-]')
#plt.title('"Make a comparison')
#plt.legend(('1.3*e11_exp_10MP','e11_exp_13MP'),
#               shadow=True, loc=(0.5, 0.4), handlelength=1.5, fontsize=12)
#plt.grid(True)
#plt.axis([0, 4000, 0.0, 0.02])
#plt.show()
#



#figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
#plt.plot(t_5MP, 1.6*e11_exp_10MP, 'r--')
#plt.plot(t_10MP, e11_exp_16MP, 'b-')
#
#plt.xlabel('time [seconds]')
#plt.ylabel('Poisson Ratio [-]')
#plt.title('"Make a comparison')
#plt.legend(('1.6*e11_exp_10MP','e11_exp_16MP'),
#               shadow=True, loc=(0.5, 0.4), handlelength=1.5, fontsize=12)
#plt.grid(True)
#plt.axis([0, 4000, 0.0, 0.02])
#plt.show()
#
#
#
#print('Maximum up to 16 MPa is linear viscoelastic')
#
#figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
#plt.plot(t_5MP, e11_exp_10MP+e11_exp_10MP, 'r--')
#plt.plot(t_10MP, e11_exp_20MP, 'b-')
#
#plt.xlabel('time [seconds]')
#plt.ylabel('Poisson Ratio [-]')
#plt.title('"Make a comparison')
#plt.legend(('e11_exp_10MP+e11_exp_10MP','e11_exp_20MP'),
#               shadow=True, loc=(0.5, 0.4), handlelength=1.5, fontsize=12)
#plt.grid(True)
#plt.axis([0, 4000, 0.0, 0.02])
#plt.show()



#figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
#plt.plot(t_5MP, 4.0*e11_exp_5MP, 'r--')
#plt.plot(t_10MP, e11_exp_20MP, 'b-')
#
#plt.xlabel('time [seconds]')
#plt.ylabel('Poisson Ratio [-]')
#plt.title('"Make a comparison')
#plt.legend(('4.0*e11_exp_5MP','e11_exp_20MP'),
#               shadow=True, loc=(0.5, 0.4), handlelength=1.5, fontsize=12)
#plt.grid(True)
#plt.axis([0, 4000, 0.0, 0.02])
#plt.show()

e11_exp_2MP
gamma_2=np.zeros(shape(e11_exp_2MP))
gamma_5=np.zeros(shape(e11_exp_2MP))
gamma_10=np.zeros(shape(e11_exp_2MP))
gamma_13=np.zeros(shape(e11_exp_2MP))
gamma_16=np.zeros(shape(e11_exp_2MP))
gamma_20=np.zeros(shape(e11_exp_2MP))
gamma_22=np.zeros(shape(e11_exp_2MP))
gamma_25=np.zeros(shape(e11_exp_2MP))
gamma_30=np.zeros(shape(e11_exp_2MP))
gamma_35=np.zeros(shape(e11_exp_2MP))

for i in range(0,len(e11_exp_2MP)):
    gamma_2[i]=e11_exp_2MP[i]/((2./2.)*e11_exp_2MP[i])
    gamma_5[i]=e11_exp_5MP[i]/((5./2.)*e11_exp_2MP[i])
    gamma_10[i]=e11_exp_10MP[i]/((10./2.)*e11_exp_2MP[i])
    gamma_13[i]=e11_exp_13MP[i]/((13./2.)*e11_exp_2MP[i])
    gamma_16[i]=e11_exp_16MP[i]/((16./2.)*e11_exp_2MP[i])
    gamma_20[i]=e11_exp_20MP[i]/((20./2.)*e11_exp_2MP[i])
    gamma_22[i]=e11_exp_22MP[i]/((22./2.)*e11_exp_2MP[i])
    gamma_25[i]=e11_exp_25MP[i]/((25./2.)*e11_exp_2MP[i])
    gamma_30[i]=e11_exp_30MP[i]/((30./2.)*e11_exp_2MP[i])
    gamma_35[i]=e11_exp_35MP[i]/((35./2.)*e11_exp_2MP[i])


figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')

#plt.plot(t_2MP, gamma_2, 'r-')
#plt.plot(t_2MP, gamma_5, 'b-')
#plt.plot(t_2MP, gamma_10, 'g-')
#plt.plot(t_2MP, gamma_13, 'rx-')
#plt.plot(t_2MP, gamma_16, 'rx-')
#plt.plot(t_2MP, gamma_20, 'bx-')
#plt.plot(t_2MP, gamma_22, 'rx-')
#plt.plot(t_2MP, gamma_25, 'bx-')
#plt.plot(t_2MP, gamma_30, 'yx-')
#plt.plot(t_2MP, gamma_35, 'gx-')
#
#
#

plt.plot(t_2MP, gamma_2,'-')
plt.plot(t_2MP, gamma_5, '-')
plt.plot(t_2MP, gamma_10, '-')
plt.plot(t_2MP, gamma_13, '.-')
plt.plot(t_2MP, gamma_16, '.-')
plt.plot(t_2MP, gamma_20, 'x-')
plt.plot(t_2MP, gamma_22, 'x-')
plt.plot(t_2MP, gamma_25, 'x-')
plt.plot(t_2MP, gamma_30, 'x-')
plt.plot(t_2MP, gamma_35, 'x-')


num_plots = 20
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
x = np.arange(10)

plt.xlabel('time [seconds]')
plt.ylabel('gamma [-]')
plt.title('Le domaine de linéarité')
plt.legend(('gamma_2','gamma_5','gamma_10','gamma_13','gamma_16'
            ,'gamma_20','gamma_22','gamma_25','gamma_30','gamma_35'), ncol=5, loc='upper center',
               shadow=True, handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([0, 4000, 0.0, 6.])
plt.show()

