# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 14:23:13 2018

@author: momoe
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:21:28 2018

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
import time

job_date_time='optimization'+time.strftime('[%X - %x]')
print(job_date_time, 'Start time')

from Tensors_Functions_S1_S2_S3_21082018 import *
from Viscoelastic_Linear_Problem_LAB_Euler_NC_Rel_Flu_31102018 import * #(the difference is just for importing)
from Viscoelastic_Linear_Function_S5_01_09102018 import *
from Viscoelastic_Linear_Function_LAB_interconversion_25102018 import *
from Viscoelastic_Linear_Function_LAB_1D_interconversion_09112018 import *



'''
My note about this function:
    I wrote one big loop include p,q,- i,j,k,l in order to calculate all values of 
    Green function (eq. 27). my calculation it was ok for long elipsodal.
    (I think because of convergence in that long direction)
    However, for spherical the answer was not correct. (some elements were not even close)
    (As you can see in the previous revision of this file in Temp folder)
    So I saw other people code (Ilyas and Rolland, in Python and Matlab)
    They used two more orders [p,q] in all tensors. 
    I mean for every values. So for example Green function would be 6-order tensor, etc
    but I used list in previous revision. Which I think was not correct. 
    I this revision. I did the same and I wrote [p,q]
    Don't think about the dimentions. It's complecated. Just consider that you have two differen
    values for each points of w and zeta3
    which should be multiplied by the weights Wpq AND calculated in the eq. 29

'''





def import_Guass_points_x_Wx(excel_file):
    'Importing omega'
    'this function is for when we do not have any title for each column, I used header= None'    
    df = pd.read_excel(excel_file,header=None, sheet_name='Sheet1')
    column_1 = df[0]
    column_2 = df[1]
    x=  np.zeros((len(column_1),1))  
    Wx = np.zeros((len(column_1),1))
    for i in range(0,len(column_1)):
        x[i]=column_1[i]
        Wx[i]=column_2[i]
    return x,Wx

excel_file_omega='omega_32points.xls'
(x_omega,W_omega)=import_Guass_points_x_Wx(excel_file_omega)

excel_file_zeta3='zeta3_4points.xls'
(x_zeta3,W_zeta3)=import_Guass_points_x_Wx(excel_file_zeta3)




# =============================================================================
# Weights of omega and zeta3
# =============================================================================
def Wpq(W_omega,W_zeta3):
    P=len(W_zeta3)
    Q=len(W_omega)
    Wpq=zeros((P, Q))
    for p in range(0,P):
        for q in range(0,Q):
            Wpq[p,q]= W_zeta3[p] * W_omega[q]
    return Wpq

Wpq = Wpq(W_omega, W_zeta3)




# =============================================================================
# Fibre longue alignée selon l’axe x3
# =============================================================================
#P=len(W_zeta3)
#Q=len(W_omega)                
#a1=1. ; a2=1. ; a3=1.e4
#k=1. ; mu=1.
#(I,J,K)=I_J_K_4orten_isotropes()
#C0= 3*k*J + 2*mu*K



kesi=np.zeros((3,P,Q))
for p in range(0,P):
    zeta3=x_zeta3[p]
    for q in range(0,Q):
        omega=x_omega[q]
        
        zeta1=sqrt(1-pow(zeta3,2.))*cos(omega)
        zeta2=sqrt(1-pow(zeta3,2.))*sin(omega)
                        
        kesi[0][p][q]= zeta1/a1
        kesi[1][p][q]= zeta2/a2 
        kesi[2][p][q]= zeta3/a3
        
        
   

K=zeros((3,3,P,Q))
    
for p in range(0,P):
    zeta3=x_zeta3[p]
    for q in range(0,Q):
        omega=x_omega[q]

        for i in range(0,3):
            for j in range(0,3):
                for k in range(0,3):
                    for l in range(0,3):
                        K[i][k][p][q] =( C0[i][j][k][l] * kesi[j][p][q] * kesi[l][p][q] ) + K[i][k][p][q]


N=zeros((3,3,P,Q))
for p in range(0,P):
    zeta3=x_zeta3[p]
    for q in range(0,Q):
        omega=x_omega[q]
        for i in range(0,3):
            for j in range(0,3):
                for k in range(0,3):
                    for l in range(0,3):
                        for m in range(0,3):
                            for n in range(0,3):
                                N[i][j][p][q]= ( (1./2.) * epsilon_ijk(i,k,l) * epsilon_ijk(j,m,n) * K[k][m][p][q] * K[l][n][p][q] ) +  N[i][j][p][q]


D=zeros((P,Q))
for p in range(0,P):
    for q in range(0,Q):
        for m in range(0,3):
            for n in range(0,3):
                for l in range(0,3):                    
                    D[p][q] = epsilon_ijk(m,n,l)*K[m][0][p][q]*K[n][1][p][q]*K[l][2][p][q] + D[p][q]


G=np.zeros((3,3,3,3,P,Q))

for p in range(0,P):
    for q in range(0,Q):
        for i in range(0,3):
            for j in range(0,3):
                for k in range(0,3):
                    for l in range(0,3):
                        G[i][j][k][l][p][q]= (kesi[k][p][q] * kesi[l][p][q] * N[i][j][p][q]/D[p][q]) + G[i][j][k][l][p][q]



SE=np.zeros((3,3,3,3))
for i in range(0,3):
    for j in range(0,3):
        for k in range(0,3):
            for l in range(0,3):
                
                for m in range(0,3):
                    for n in range(0,3):
                        
                        for p in range(0,P):
                            for q in range(0,Q):
                               
                                SE[i][j][k][l] = (C0[m][n][k][l]/(8.*pi)) * ( G[i][m][j][n][p][q] + G[j][m][i][n][p][q] ) * Wpq[p][q] + SE[i][j][k][l]
                                




print('========================================================')  
print('Problème 07 – 01 : Calculs sur le tenseur d’Eshelby')                                                               
print('========================================================')  

# =============================================================================
# Result: Inclusions sphériques
# =============================================================================
print('Inclusions sphériques ') 
print('P = 4 et Q = 32, où P est le nombre de quadratures') 
print('k0 = mu0 = 1') 
print('...............')                              
                                
k0=mu0=1.
(I,J,K)=I_J_K_4orten_isotropes()
SE_sph = (3*k0)/(3*k0+4*mu0) * J + (6*(k0+2*mu0)) / (5*(3*k0+4*mu0)) * K


for i in range(0,3):
    for j in range(0,3):
        for k in range(0,3):
            for l in range(0,3):
                print('ijkl',i,j,k,l,SE[i][j][k][l],'-----',SE_sph[i][j][k][l])
                print('-----------')


# =============================================================================
# Inclusions sphériques
# =============================================================================
P=len(W_zeta3)
Q=len(W_omega)                
a1=1. ; a2=1. ; a3=1.
k=1. ; mu=1.
(I,J,K)=I_J_K_4orten_isotropes()
C0= 3*k*J + 2*mu*K



##v=(3.*k - 2.*mu) / (2.*mu + 6.*k)
#v=1./8.
#SE_3333=0.
#SE_1111 = SE_2222 = (5.-4.*v) / (8.*(1.-v))
#SE_1212 = (3. - 4.*v)/(8.*(1.-v))
#SE_3131 = SE_3232 = 1./4.
#SE_1122 = SE_2211 = (4.*v - 1.) /  (8.*(1.-v))
#SE_1133 = SE_2233= v/(2.*(1.-v))
#SE_3311=SE_3322=0.
#
#
#print('-----------------------------------')
#print(SE[2][2][2][2], 'S_Anl=', SE_3333)
#print('-----------------------------------')
#print(SE[0][0][0][0], 'S_Anl=', SE_1111)
#print('-----------------------------------')
#print(SE[1][1][1][1], 'S_Anl=', SE_2222)
#print('-----------------------------------')
#print(SE[0][1][0][1], 'S_Anl=', SE_1212)
#print('-----------------------------------')
#print(SE[2][0][2][0], 'S_Anl=', SE_3131)
#print('-----------------------------------')
#print(SE[2][1][2][1], 'S_Anl=', SE_3232)
#print('-----------------------------------')
#print(SE[0][0][1][1], 'S_Anl=', SE_1122)
#print('-----------------------------------')
#print(SE[1][1][0][0], 'S_Anl=', SE_2211)
#print('-----------------------------------')
#print(SE[0][0][2][2], 'S_Anl=', SE_1133)
#print('-----------------------------------')
#print(SE[1][1][2][2], 'S_Anl=', SE_2233)
#print('-----------------------------------')
#print(SE[2][2][0][0], 'S_Anl=', SE_3311)
#print('-----------------------------------')
#print(SE[2][2][1][1], 'S_Anl=', SE_3322)
#print('\n')
#print('-----------------------------------')
#print('-----------------------------------')
#
#job_date_time='optimization'+time.strftime('[%X - %x]')
#print(job_date_time, 'Start time')
#
