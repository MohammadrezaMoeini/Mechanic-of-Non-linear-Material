# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 14:09:58 2018

@author: hp
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
from numpy.linalg import inv



from Tensors_Functions_S1_S2_S3_21082018 import *
from Viscoelastic_Linear_Problem_LAB_Euler_NC_Rel_Flu_31102018 import * #(the difference is just for importing)
from Viscoelastic_Linear_Function_S5_01_09102018 import *
from Viscoelastic_Linear_Function_LAB_interconversion_25102018 import *
from Viscoelastic_Linear_Function_LAB_1D_interconversion_09112018 import *

from Homogenization_Function_S11_Eshelby_16112018py import *
from Homogenization_Function_S12_diluée_18112018py import *


def C_Voigt_k_mu_homogen(k,mu):
    (I,J,K)=I_J_K_4orten_isotropes()
    J_V=tensor4_to_Voigt_tensor2(J)
    K_V=tensor4_to_Voigt_tensor2(K)    
    C_V=3.*k*J_V + 2.*mu*K_V
    return C_V

   

# =============================================================================
# =============================================================================
# # Mori_Tanaka_Spherical
# =============================================================================
# =============================================================================
    

# =============================================================================
# Mori_Tanaka_sph
# =============================================================================
'Mori_Tanaka: isotrope renforcée par des particules sphériques isotropes'

def Mori_Tanaka_sph_Tr(k0,mu0,kfr,mufr,cfr):
    'MULTI-Reinforcements: Write kfr,mufr,cfr as list'
      
    n=len(cfr) #number of phases
    Tr_alpha = np.zeros((n))
    Tr_betha = np.zeros((n))
    (alpha_sph_Ef, betha_sph_Ef)=SE_sphériques_alpa_betha(k0,mu0)

    for i in range(0,n):
        Tr_alpha[i]= k0 / (k0 +  (alpha_sph_Ef)*(kfr[i]-k0) )
        Tr_betha[i]= mu0 / (mu0 + (betha_sph_Ef)*(mufr[i]-mu0) )
        
    return Tr_alpha, Tr_betha


def Mori_Tanaka_sph_crTr_inv(Tr_alpha, Tr_betha, cfr):
    n=len(cfr)
    c0= 1. - sum(cfr)
    sum_Tr_alpha=0.
    sum_Tr_betha=0.
    for i in range(0,n):
        sum_Tr_alpha= cfr[i]*Tr_alpha[i] + sum_Tr_alpha
        sum_Tr_betha= cfr[i]*Tr_betha[i] + sum_Tr_alpha
    
    sum_crTr_inv_alpha=pow(c0*1. + sum_Tr_alpha,-1)
    sum_crTr_inv_betha=pow(c0*1. + sum_Tr_betha,-1)
    
    return sum_crTr_inv_alpha, sum_crTr_inv_betha


def Mori_Tanaka_sph_Ar(Tr_alpha, Tr_betha, cfr):
    n=len(cfr)    
    (sum_crTr_inv_alpha, sum_crTr_inv_betha)=Mori_Tanaka_sph_crTr_inv(Tr_alpha, Tr_betha, cfr)
    Ar_alpha = np.zeros((n))
    Ar_betha = np.zeros((n))

    for r in range(0,n):
        Ar_alpha[r]=Tr_alpha[r]*(sum_crTr_inv_alpha)
        Ar_betha[r]=Tr_betha[r]*(sum_crTr_inv_betha)
        
    return Ar_alpha,Ar_betha

def Mori_Tanaka_sph_C_homogen(Ar_alpha,Ar_betha,k0,mu0,kfr,mufr,cfr):
    n=len(cfr)
    sum_temp_alpha=0.
    sum_temp_betha=0.
    for r in range(0,n):
        sum_temp_alpha=cfr[r]*(3.*kfr[r] - 3.*k0)*Ar_alpha[r] + sum_temp_alpha
        sum_temp_betha=cfr[r]*(2.*mufr[r] - 2.*mu0)*Ar_betha[r] + sum_temp_betha
        
    k_homogen = (1./3.)* ( sum_temp_alpha + 3.*k0)
    mu_homogen = (1./2.)* (sum_temp_betha + 2.*mu0)
    
    return k_homogen, mu_homogen


def Mori_Tanaka_sphériques(k0,mu0,kfr,mufr,cfr):
    'Mori_Tanaka: isotrope renforcée par des particules sphériques isotropes'
    
    if type(kfr) is not list or type(mufr) is not list or type(cfr) is not list:
        print('********************************************************')
        print('ERROR: MULTI-Reinforcements: Write kfr,mufr,cfr as list')
        print('********************************************************')
        
    (Tr_alpha, Tr_betha) = Mori_Tanaka_sph_Tr(k0,mu0,kfr,mufr,cfr)
    (sum_crTr_inv_alpha, sum_crTr_inv_betha)= Mori_Tanaka_sph_crTr_inv(Tr_alpha, Tr_betha, cfr)
    (Ar_alpha, Ar_betha) =  Mori_Tanaka_sph_Ar(Tr_alpha, Tr_betha, cfr)
    (k_homogen, mu_homogen) =  Mori_Tanaka_sph_C_homogen(Ar_alpha,Ar_betha,k0,mu0,kfr,mufr,cfr)
    
    C_Voigt_homogen=C_Voigt_k_mu_homogen(k_homogen, mu_homogen)
    
    return k_homogen, mu_homogen, C_Voigt_homogen




# =============================================================================
# =============================================================================
# # Mori_Tanaka_elp
# =============================================================================
# =============================================================================


# =============================================================================
# Mori_Tanaka_elp: 2 phases: Matrix + Reinforement 
# =============================================================================
'Mori_Tanaka: fibres ellipsoïdales alignées selon un axe'
'This function is for just one reinforcement'    

def Mori_Tanaka_elp_ONEPHASE_Cr_C0(k0,mu0,kf,muf,cf):

    
    Cr_V=np.zeros((6,6))
    
    (I,J,K)=I_J_K_4orten_isotropes()
    I_V=tensor4_to_Voigt_tensor2(I)
    J_V=tensor4_to_Voigt_tensor2(J)
    K_V=tensor4_to_Voigt_tensor2(K)
    
    C_V=3.*k0*J_V + 2.*mu0*K_V
    Cr_V= 3.*kf*J_V + 2.*muf*K_V
        
    return Cr_V, C_V, I_V, J_V, K_V

def Mori_Tanaka_elp_ONEPHASE_Tr(k0,mu0,kfr,mufr,cfr,excel_file_zeta3, excel_file_omega,a1,a2,a3):
    (Cr_V, C_V, I_V, J_V, K_V)= Mori_Tanaka_elp_ONEPHASE_Cr_C0(k0,mu0,kfr,mufr,cfr) 

    SE_elp_numerical=Eshelby_tensor_ellipsoid(excel_file_zeta3, excel_file_omega, k0, mu0,a1,a2,a3)
    SE_elp_numerical_Voigt=tensor4_to_Voigt_tensor2_NOT_Major_Sym(SE_elp_numerical)
             
    Tr=inv ( I_V +  np.matmul ( np.matmul( SE_elp_numerical_Voigt,inv(C_V) ) , Cr_V-C_V) )
        
    return Tr

def Mori_Tanaka_elp_ONEPHASE_crTr_inv(Tr,k0,mu0,kfr,mufr,cfr):
    c0= 1. - cfr
      
    (Cr_V, C_V, I_V, J_V, K_V)= Mori_Tanaka_elp_ONEPHASE_Cr_C0(k0,mu0,kfr,mufr,cfr)
    
    sum_crTr_inv=inv(c0*I_V + cfr*Tr)
    
    return sum_crTr_inv

def Mori_Tanaka_elp_ONEPHASE_Ar(Tr,k0,mu0,kfr,mufr,cfr):
    sum_crTr_inv=Mori_Tanaka_elp_ONEPHASE_crTr_inv(Tr,k0,mu0,kfr,mufr,cfr)    
    Ar=  np.matmul (Tr,sum_crTr_inv)
        
    return Ar

def Mori_Tanaka_elp_ONEPHASE_C_homogen(Ar,k0,mu0,kfr,mufr,cfr):
    (Cr_V, C_V, I_V, J_V, K_V)= Mori_Tanaka_elp_ONEPHASE_Cr_C0(k0,mu0,kfr,mufr,cfr)
    
    C_homogen = C_V  + np.matmul(cfr*(Cr_V - C_V), Ar)
      
    return C_homogen


def Mori_Tanaka_ellipsoidales_ONEPHASE(k0,mu0,kfr,mufr,cfr,excel_file_zeta3, excel_file_omega,a1,a2,a3):
    'Mori_Tanaka: fibres ellipsoïdales alignées selon un axe'
    'This function is for just one reinforcement'  
    
    Tr = Mori_Tanaka_elp_ONEPHASE_Tr(k0,mu0,kfr,mufr,cfr,excel_file_zeta3, excel_file_omega,a1,a2,a3)
        
    Ar =  Mori_Tanaka_elp_ONEPHASE_Ar(Tr,k0,mu0,kfr,mufr,cfr)
    
    C_homogen = Mori_Tanaka_elp_ONEPHASE_C_homogen(Ar,k0,mu0,kfr,mufr,cfr)
        
    return C_homogen




# =============================================================================
# Mori_Tanaka_elp: 2 phases: Matrix + Reinforement  INPT Cf
# =============================================================================
'Mori_Tanaka: fibres ellipsoïdales alignées selon un axe'
'In this function input is C (6x6 Voigt notation) for fiber'
'this function has been written specifically for problem 07-04 case III'

def Mori_Tanaka_elp_ONEPHASE_Cr_C0_INPUT_Cf(k0,mu0,Cfr,cfr):

    
    Cr_V=np.zeros((6,6))
    
    (I,J,K)=I_J_K_4orten_isotropes()
    I_V=tensor4_to_Voigt_tensor2(I)
    J_V=tensor4_to_Voigt_tensor2(J)
    K_V=tensor4_to_Voigt_tensor2(K)
    
    C_V=3.*k0*J_V + 2.*mu0*K_V
    
    Cr_V=Cfr    
#    Cr_V= tensor2_Voigt_to_tensor4_NOT_Major_Sym(Cfr)
        
    return Cr_V, C_V, I_V, J_V, K_V

def Mori_Tanaka_elp_ONEPHASE_Tr_INPUT_Cf(k0,mu0,Cfr,cfr,excel_file_zeta3, excel_file_omega,a1,a2,a3):
    (Cr_V, C_V, I_V, J_V, K_V)= Mori_Tanaka_elp_ONEPHASE_Cr_C0_INPUT_Cf(k0,mu0,Cfr,cfr) 

    SE_elp_numerical=Eshelby_tensor_ellipsoid(excel_file_zeta3, excel_file_omega, k0, mu0,a1,a2,a3)
    SE_elp_numerical_Voigt=tensor4_to_Voigt_tensor2_NOT_Major_Sym(SE_elp_numerical)
             
    Tr=inv ( I_V +  np.matmul ( np.matmul( SE_elp_numerical_Voigt,inv(C_V) ) , Cr_V-C_V) )
        
    return Tr

def Mori_Tanaka_elp_ONEPHASE_crTr_inv_INPUT_Cf(Tr,k0,mu0,Cfr,cfr):
    c0= 1. - cfr
      
    (Cr_V, C_V, I_V, J_V, K_V)= Mori_Tanaka_elp_ONEPHASE_Cr_C0_INPUT_Cf(k0,mu0,Cfr,cfr)
    
    sum_crTr_inv=inv(c0*I_V + cfr*Tr)
    
    return sum_crTr_inv

def Mori_Tanaka_elp_ONEPHASE_Ar_INPUT_Cf(Tr,k0,mu0,Cfr,cfr):
    sum_crTr_inv=Mori_Tanaka_elp_ONEPHASE_crTr_inv_INPUT_Cf(Tr,k0,mu0,Cfr,cfr)    
    Ar=  np.matmul (Tr,sum_crTr_inv)
        
    return Ar

def Mori_Tanaka_elp_ONEPHASE_C_homogen_INPUT_Cf(Ar,k0,mu0,Cfr,cfr):
    (Cr_V, C_V, I_V, J_V, K_V)= Mori_Tanaka_elp_ONEPHASE_Cr_C0_INPUT_Cf(k0,mu0,Cfr,cfr)
    
    C_homogen = C_V  + np.matmul(cfr*(Cr_V - C_V), Ar)
      
    return C_homogen


def Mori_Tanaka_ellipsoidales_ONEPHASE_INPUT_Cf(k0,mu0,Cfr,cfr,excel_file_zeta3, excel_file_omega,a1,a2,a3):
    '''Mori_Tanaka: fibres ellipsoïdales alignées selon un axe'
    'In this function input is C (6x6 Voigt notation) for fiber'
    'this function has been written specifically for problem 07-04 case III'''
            
    Tr = Mori_Tanaka_elp_ONEPHASE_Tr_INPUT_Cf(k0,mu0,Cfr,cfr, excel_file_zeta3, excel_file_omega,a1,a2,a3)
        
    Ar =  Mori_Tanaka_elp_ONEPHASE_Ar_INPUT_Cf(Tr,k0,mu0,Cfr,cfr)
    
    C_homogen = Mori_Tanaka_elp_ONEPHASE_C_homogen_INPUT_Cf(Ar,k0,mu0,Cfr,cfr)
        
    return C_homogen






# =============================================================================
# Mori_Tanaka_elp: MORE THAN 2PHASES (Using for and .append as list)
# =============================================================================
'Mori_Tanaka: fibres ellipsoïdales alignées selon un axe'
'This function is for more than one reinforcement'
' kfr,mufr,cfr,a1,a2,a3 ---- > list or np.array'

def Mori_Tanaka_elp_Cr_C0(k0,mu0,kfr,mufr,cfr):
            
    n=len(cfr) #number of phases
    Cr_V=[]
    
    (I,J,K)=I_J_K_4orten_isotropes()
    I_V=tensor4_to_Voigt_tensor2(I)
    J_V=tensor4_to_Voigt_tensor2(J)
    K_V=tensor4_to_Voigt_tensor2(K)
    
    C_V=3.*k0*J_V + 2.*mu0*K_V

    for i in range(0,n):
        Cr_Vi= 3.*kfr[i]*J_V + 2.*mufr[i]*K_V
        Cr_V.append(Cr_Vi)
        
    return Cr_V, C_V, I_V, J_V, K_V



def Mori_Tanaka_elp_Tr(k0,mu0,kfr,mufr,cfr,excel_file_zeta3, excel_file_omega,a1,a2,a3):
    ' kfr,mufr,cfr,a1,a2,a3 ---- > list or np.array'  
    (Cr_V, C_V, I_V, J_V, K_V)= Mori_Tanaka_elp_Cr_C0(k0,mu0,kfr,mufr,cfr) 
    n=len(cfr) #number of phases
    Tr = []
    SE_sph_numerical= []
    SE_sph_numerical_Voigt= []
    

    for i in range(0,n):
        SE_sph_numericali=Eshelby_tensor_ellipsoid(excel_file_zeta3, excel_file_omega, k0, mu0,a1[i],a2[i],a3[i])
        SE_sph_numerical_Voigti=tensor4_to_Voigt_tensor2_NOT_Major_Sym(SE_sph_numericali)
        
        SE_sph_numerical.append(SE_sph_numericali)
        SE_sph_numerical_Voigt.append(SE_sph_numerical_Voigti)
        
        
    for i in range(0,n):
        SEi=SE_sph_numerical_Voigt[i]
        Cr_Vi=Cr_V[i]
        Tri=inv ( I_V +  np.matmul ( np.matmul( SEi,inv(C_V) ) , Cr_Vi-C_V) )
        Tr.append(Tri)
        
    return Tr


def Mori_Tanaka_elp_crTr_inv(Tr,k0,mu0,kfr,mufr,cfr):
    n=len(cfr)
    c0= 1. - sum(cfr)
      
    (Cr_V, C_V, I_V, J_V, K_V)= Mori_Tanaka_elp_Cr_C0(k0,mu0,kfr,mufr,cfr)

    sum_Tr=np.zeros((6,6))
    for i in range(0,n):
        sum_Tr= cfr[i]*Tr[i] + sum_Tr
    
    sum_crTr_inv=inv(c0*I_V + sum_Tr)
    
    return sum_crTr_inv


def Mori_Tanaka_elp_Ar(Tr,k0,mu0,kfr,mufr,cfr):
    n=len(cfr)    
    sum_crTr_inv=Mori_Tanaka_elp_crTr_inv(Tr,k0,mu0,kfr,mufr,cfr)
    
    Ar = []

    for r in range(0,n):
        Ari=  np.matmul (Tr[:][:][r],sum_crTr_inv)
        Ar.append(Ari)
        
    return Ar


def Mori_Tanaka_elp_C_homogen(Ar,k0,mu0,kfr,mufr,cfr):
    n=len(cfr)
    (Cr_V, C_V, I_V, J_V, K_V)= Mori_Tanaka_elp_Cr_C0(k0,mu0,kfr,mufr,cfr)
    sum_temp=np.zeros((6,6))
    
    for r in range(0,n):
        Cr_Vi=Cr_V[r]
        sum_temp= np.matmul(cfr[r]*(Cr_Vi - C_V), Ar[r]) + sum_temp
        
    C_homogen = C_V  + sum_temp
    
    return C_homogen




def Mori_Tanaka_ellipsoidales(k0,mu0,kfr,mufr,cfr,excel_file_zeta3, excel_file_omega,a1,a2,a3):
    '''Mori_Tanaka: fibres ellipsoïdales alignées selon un axe'
    'This function is for more than one reinforcement'
    ' kfr,mufr,cfr,a1,a2,a3 ---- > list or np.array'''
    
    
    print('---------------------------------------------------------')
    print('MULTI-Reinforcements: Write kfr,mufr,cfr,a1,a2,a3 as list')
    print('---------------------------------------------------------')

    
    Tr = Mori_Tanaka_elp_Tr(k0,mu0,kfr,mufr,cfr,excel_file_zeta3, excel_file_omega,a1,a2,a3)
        
    Ar =  Mori_Tanaka_elp_Ar(Tr,k0,mu0,kfr,mufr,cfr)
    
    C_homogen = Mori_Tanaka_elp_C_homogen(Ar,k0,mu0,kfr,mufr,cfr)
        
    return C_homogen










