# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 11:07:36 2018

@author: momoe
"""


import scipy
import scipy.linalg   # SciPy Linear Algebra Library
from numpy.linalg import inv
from array import *

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

def Cv_relaxation_test(k,mu):
    Cv=[]
    (I,J,K)=I_J_K_4orten_isotropes()
    J_V=tensor4_to_Voigt_tensor2(J)
    K_V=tensor4_to_Voigt_tensor2(K)
    for i in range(0,len(k)):       
        Ci= 3.*k[i]*J_V + 2.*mu[i]*K_V
        Cv.append(Ci)
        
    return Cv    



def interconversion_C_to_S(C,w):
    (B,L1,L2,L3,N,D) = B_L1_L2_L3(C,w)

    A1=inv(L1)
    A2=np.matmul(inv(L1),L2)
    A3=L3 - np.matmul( transpose(L2), np.matmul( inv(L1),L2 ) )

    (U,d,V)=np.linalg.svd(A3)
    Dp=diag(d)
    P=U

    B_star = np.matmul( transpose(P), np.matmul( B, P ) )
    A3_star = Dp
    A2_star = np.matmul( A2, P )

    S0=A1

    landa=[]
    for i in range(0,len(A3_star)):
        landa.append(A3_star[i,i])
    
    Sm=[]
    for m in range(0,D*N):
        S=np.zeros((D,D))
        for i in range(0,D):
            for j in range(0,D):
                S[i,j]=A2_star[i,m]*A2_star[j,m]/landa[m]
        Sm.append(S)
        
    print('''Pay attention, If you have mu and k; put individually.
          They are coeficients of orthogonal tensors and they are independent.
          Then
          mu0 = 2.*S0, mu_i = 2.*S[i]
          ku0 = 3.*S0, ku_i = 3.*S[i]
          
          mu(t) = mu0 + mu1 (1.-exp(-landa[1]t)) + mu2 (1.-exp(-landa[2]t)) + ...
          k(t) = k0 + k1 (1.-exp(-landa[1]t)) + k2 (1.-exp(-landa[2]t)) + ...
          
          Check Problem 05_06 for writing mu(t) & k(t)
          
          ''')
     
    return S0, Sm, landa
    



def interconversion_S_to_C(S,landa):
    (B,A1,A2,A3,M,D)=B_A1_A2_A3(S,landa)
    
    L1=inv(A1)
    L2=np.matmul(inv(A1),A2)
    L3=A3 + np.matmul( transpose(A2), np.matmul( inv(A1),A2 ) )
    
    (U,d,V)=np.linalg.svd(L3)
    Dp=diag(d)
    P=U
    
    B_star = np.matmul( transpose(P), np.matmul( B, P ) )
    L3_star = Dp
    L2_star = np.matmul( L2, P )
        
    w=[]
    for i in range(0,len(L3_star)):
        w.append(L3_star[i,i])    
    
    Cn=[]
    for n in range(0,D*M):
        C=np.zeros((D,D))
        for i in range(0,D):
            for j in range(0,D):
                C[i,j]=L2_star[i,n]*L2_star[j,n]/w[n]
        Cn.append(C)
    
    C_total=np.zeros((D,D))
    for n in range(0,D*M):
        C_total=Cn[n] + C_total
    
    C0=L1 - C_total
    
    print('''Pay attention, If you have mu and k; put individually.
          They are coeficients of orthogonal tensors and they are independent.
          Then
          mu0 = 2.*C0, mu_i = 2.*C[i]
          ku0 = 3.*C0, k_i = 3.*C[i]
          
          mu(t) = mu0 + mu1 (1.-exp(-landa[1]t)) + mu2 (1.-exp(-landa[2]t)) + ...
          k(t) = k0 + k1 (1.-exp(-landa[1]t)) + k2 (1.-exp(-landa[2]t)) + ...
          
          Check Problem 05_06 for writing mu(t) & k(t)
          
          ''')      
    return C0, Cn, w
    
        

# =============================================================================
# Calculate alpha and betha for (S or C)
# =============================================================================

# =============================================================================
# Using 2-order tensor
# =============================================================================
def alpha_betha_2orten_isotropes(A):
    '''tenseur isotrope  ---------> alph et betha [2 constantes indépendantes]'''
    (I,J,K)=I_J_K_4orten_isotropes()
    J_V=tensor4_to_Voigt_tensor2(J)
    K_V=tensor4_to_Voigt_tensor2(K)
    
    alpha=tensor2_contract2_tensor2(J_V,A)
    betha=(1./5.)*(tensor2_contract2_tensor2(K_V,A))
    
    return(alpha, betha)

def alphai_bethai_for_S2_or_C2(C2_S2):
    alpha=[]; betha=[];
    for i in range(0,len(C2_S2)):
        C2i_S2i=C2_S2[i]
        (alphai, bethai)=alpha_betha_2orten_isotropes(C2i_S2i)
        alpha.append(alphai)
        betha.append(bethai)
    return alpha, betha


def alphs0_betha0_for_S02_or_C02(C0_S0):
    (alpha0, betha0)=alpha_betha_2orten_isotropes(C0_S0)
    
    return alpha0, betha0
            


# =============================================================================
# Using 4-order tensor
# =============================================================================
def alphai_bethai_for_S_or_C(C_S):
    alpha=[]; betha=[];
    for i in range(0,len(C_S)):
        C4_S4=tensor2_Voigt_to_tensor4(C_S[i])
        (alphai, bethai)=alpha_betha_4orten_isotropes(C4_S4)
        alpha.append(alphai)
        betha.append(bethai)
    return alpha, betha
        


def alphs0_betha0_for_S0_or_C0(C0_S0):
    C04_S04=tensor2_Voigt_to_tensor4(C0_S0)
    (alpha0, betha0)=alpha_betha_4orten_isotropes(C04_S04)
    
    return alpha0, betha0
    
    




# =============================================================================
# Testing Fuctions
# =============================================================================
#       Interconversion of linearly viscoelastic material functions 
#       expressed as Prony series: a closure''')
#       Jacques Luk-Cyr, Thibaut Crochon, Chun Li, Martin Lévesque


#               print('C(t) --------------> S(t)')

C0=0.1*np.array([[136.8,1.546,0.996,1.209,0.408,0.026],
                 [1.546,100.9,-3.455,-4.792,5.692,1.367],
                 [0.996,-3.455,51.18,48.78,-20.04,-15.27],
                 [1.209,-4.792,48.78,75.10,-37.96,4.752],
                 [0.408,5.692,-20.04,-37.96,235.3,12.08],
                 [0.026,1.367,-15.27,4.752,12.08,189.1]])

(alpha, betha)=alpha_betha_2orten_isotropes(C0)


C1=0.1*np.array([[44.76,0.560,1.741,0.173,2.033,2.467],
                 [0.560,116.9,-27.11,35.68,11.87,-10.81],
                 [1.741,-27.11,55.8,-3.398,5.29,-9.681],
                 [0.173,35.68,-3.398,51.97,-30.22,18.29],
                 [2.033,11.87,5.285,-30.22,95.46,-50.67],
                 [2.467,-10.81,-9.681,18.29,-50.67,88.94]])


C2=0.1*np.array([[25.73,-12.29,-9.826,-6.928,-1.113,-2.627],
                 [-12.29,119.2,-53.04,-57.93,-4.033,18.40],
                 [-9.826,-53.04,147.4,-31.91,-56.97,-16.33],
                 [-6.928,-57.93,-31.91,198.1,61.87,-32.13],
                 [-1.113,-4.033,-56.97,61.87,155.0,23.37],
                 [-2.627,18.40,-16.33,-32.13,23.37,97.59]])

C=[]
C.append(C0)
C.append(C1)
C.append(C2)

w=[14.514, 368.88]


(S0,Sm,landa)=interconversion_C_to_S(C,w)

(alpha, betha)=alphai_bethai_for_S2_or_C2(Sm)
(alpha0, betha0)= alphs0_betha0_for_S0_or_C0(S0)


#(alpha, betha)=alphai_bethai_for_S_or_C(Sm)
#(alpha0, betha0)= alphs0_betha0_for_S0_or_C0(S0)



print('S(t) --------------> C(t)')

S0=0.1*np.array([[210.174,48.162,2.118,0.423,0.537,0.071],
                 [48.162,195.498,7.986,1.111,1.036,-1.379],
                 [2.118,7.986,18.481,-15.804,-6.611,17.307],
                 [0.423,1.111,-15.804,228.036,-67.011,33.105],
                 [0.537,1.036,-6.611,-67.011,177.019,35.857],
                 [0.071,-1.379,17.307,33.105,35.857,147.184]])



S1=0.1*np.array([[16.997,1.452,3.006,0.409,0.749,0.059],
               [1.452,102.517,-39.342,8.715,-22.003,28.447],
                 [3.006,-39.342,32.762,-3.286,1.264,-14.781],
                 [0.409,8.715,-3.286,95.569,-45.699,-36.615],
                 [0.749,-22.003,1.264,-45.699,74.399,18.321],
                 [0.059,28.447,-14.781,-36.615,18.321,87.530]])


S=[]
S.append(S0)
S.append(S1)
landa=[0.043145]


(C0, Cn, w)=interconversion_S_to_C(S,landa)


#
