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



# =============================================================================
# =============================================================================
# # Fibre longue alignée selon l’axe x3
# =============================================================================
# =============================================================================

# Weights of omega and zeta3    
def Eshelby_Wpq(W_omega,W_zeta3,P,Q):
    'Weights of omega and zeta3'
    Wpq=np.zeros((P, Q))
    for p in range(0,P):
        for q in range(0,Q):
            Wpq[p,q]= W_zeta3[p] * W_omega[q]
    return Wpq


def Eshelby_kesi(x_zeta3,x_omega,a1,a2,a3,P,Q):
    kesi=np.zeros((3,P,Q))
    for p in range(0,P):
        zeta3=x_zeta3[p]
        for q in range(0,Q):
            omega=x_omega[q]
        
            zeta1=np.sqrt(1-pow(zeta3,2.))*np.cos(omega)
            zeta2=np.sqrt(1-pow(zeta3,2.))*np.sin(omega)
                        
            kesi[0][p][q]= zeta1/a1
            kesi[1][p][q]= zeta2/a2 
            kesi[2][p][q]= zeta3/a3
    return kesi
        
        
   
def Eshelby_K(kesi,C0,P,Q): 
    K=np.zeros((3,3,P,Q))    

    for i in range(0,3):
        for k in range(0,3):
            for p in range(0,P):
                for q in range(0,Q):
                               
                    for j in range(0,3):
                        for l in range(0,3):
                            
                                        
                            K[i][k][p][q] =( C0[i][j][k][l] * kesi[j][p][q] * kesi[l][p][q] ) + K[i][k][p][q]
    return K



def Eshelby_N(x_zeta3,x_omega,K,P,Q):
    N=np.zeros((3,3,P,Q))

    for i in range(0,3):
        for j in range(0,3):
            
            for k in range(0,3):
                for l in range(0,3):
                    for m in range(0,3):
                        for n in range(0,3):
                                    
                            for p in range(0,P):
                                for q in range(0,Q):

                                    N[i][j][p][q]= ( (1./2.) * epsilon_ijk(i,k,l) * epsilon_ijk(j,m,n) * K[k][m][p][q] * K[l][n][p][q] ) +  N[i][j][p][q]
    return N


def Eshelby_D(x_zeta3,x_omega,K,P,Q):
    D=np.zeros((P,Q))

    for m in range(0,3):
        for n in range(0,3):
            for l in range(0,3): 
                for p in range(0,P):
                    for q in range(0,Q):
                        
                        D[p][q] = epsilon_ijk(m,n,l)*K[m][0][p][q]*K[n][1][p][q]*K[l][2][p][q] + D[p][q]                        
    return D


def Eshelby_G(D,N,kesi,P,Q):
    G=np.zeros((3,3,3,3,P,Q))

    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                for l in range(0,3):
                    for p in range(0,P):
                        for q in range(0,Q):
                            G[i][j][k][l][p][q]= (kesi[k][p][q] * kesi[l][p][q] * N[i][j][p][q]/D[p][q]) + G[i][j][k][l][p][q]
    return G


def Eshelby_SE(C0,G,Wpq,P,Q):
    SE=np.zeros((3,3,3,3))
            
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                for l in range(0,3):
                
                    for m in range(0,3):
                        for n in range(0,3):
                            for p in range(0,P):
                                for q in range(0,Q):
                                                      
                                    SE[i][j][k][l] = (C0[m][n][k][l]/(8.*np.pi)) * ( G[i][m][j][n][p][q] + G[j][m][i][n][p][q] ) * Wpq[p][q] + SE[i][j][k][l]
    return SE
                                


def Eshelby_zeros_modifier(SE):
    'if abs(SE) < 1.0e-7 ----> SE=0'
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                for l in range(0,3):
                    if abs(SE[i][j][k][l]) < 1.0e-7:
                        SE[i][j][k][l]=0.
    return SE



def Eshelby_tensor_ellipsoid(excel_file_zeta3, excel_file_omega, k0, mu0, a1, a2, a3):

    #Points de Gauss 
    (x_omega,W_omega)=import_Guass_points_x_Wx(excel_file_omega)
    (x_zeta3,W_zeta3)=import_Guass_points_x_Wx(excel_file_zeta3)
    
    P=len(x_zeta3)
    Q=len(x_omega)
#    print('Eshelby calculator (fibres ellipsoïdales)')
#    print( 'Gauss points:','P =', P,'and','Q =', Q)
    Wpq = Eshelby_Wpq(W_omega, W_zeta3,P,Q)
    
    (I0,J0,K0)=I_J_K_4orten_isotropes()
    C0= 3.*k0*J0 + 2.*mu0*K0
    
    kesi=Eshelby_kesi(x_zeta3,x_omega,a1,a2,a3,P,Q)
    
    K=Eshelby_K(kesi,C0,P,Q)
    
    N=Eshelby_N(x_zeta3,x_omega,K,P,Q)
    
    D=Eshelby_D(x_zeta3,x_omega,K,P,Q)
    
    G=Eshelby_G(D,N,kesi,P,Q)
    
    SE=Eshelby_SE(C0,G,Wpq,P,Q)
    
    SE_modified=Eshelby_zeros_modifier(SE)
    
    return SE_modified
    




# =============================================================================
# =============================================================================
# # Fibre longue alignée selon l’axe x3
# =============================================================================
# =============================================================================

def Eshelby_tensor_ellipsoid_C0Input(excel_file_zeta3, excel_file_omega, C0, a1, a2, a3):
    'For using auto-coherant, we input C which is not isotropic'

    #Points de Gauss 
    (x_omega,W_omega)=import_Guass_points_x_Wx(excel_file_omega)
    (x_zeta3,W_zeta3)=import_Guass_points_x_Wx(excel_file_zeta3)
    
    P=len(x_zeta3)
    Q=len(x_omega)
#    print('Eshelby calculator (fibres ellipsoïdales)')
#    print( 'Gauss points:','P =', P,'and','Q =', Q)
    Wpq = Eshelby_Wpq(W_omega, W_zeta3,P,Q)
        
    kesi=Eshelby_kesi(x_zeta3,x_omega,a1,a2,a3,P,Q)
    
    K=Eshelby_K(kesi,C0,P,Q)
    
    N=Eshelby_N(x_zeta3,x_omega,K,P,Q)
    
    D=Eshelby_D(x_zeta3,x_omega,K,P,Q)
    
    G=Eshelby_G(D,N,kesi,P,Q)
    
    SE=Eshelby_SE(C0,G,Wpq,P,Q)
    
    SE_modified=Eshelby_zeros_modifier(SE)
    
    return SE_modified         

# =============================================================================
# =============================================================================
# Inclusions: sphériques
# =============================================================================
# =============================================================================
def Eshelby_tensor_Spherical(k0,mu0):
    (I,J,K)=I_J_K_4orten_isotropes()
    SE_sph = (3.*k0)/(3.*k0+4.*mu0) * J + (6.*(k0+2.*mu0)) / (5.*(3.*k0+4.*mu0)) * K
    return SE_sph
    
def Eshelby_tensor_Spherical_alpha_betha(k0,mu0):
    alpha_SE=(3.*k0)/(3.*k0 + 4.*mu0)
    betha_SE=(6.*(k0 + 2.*mu0)) / (5.*(3.*k0 + 4.*mu0))
    return alpha_SE, betha_SE

