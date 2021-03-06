# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 16:26:46 2018

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

from Homogenization_Function_S11_Eshelby_16112018py import *
from Homogenization_Function_S12_diluée_18112018py import *
from Homogenization_Function_S12_Mori_Tanaka_18112018py import *

from Homogenization_Function_S12_Voigt_Reuss_19112018py import *

print('**********************************************')
print('Problème 07 – 04 : Les nano-composites à base de nano-tubes de carbone - défis et potentiel')
print('**********************************************')




print('===========================  1  =============================')
print('Le cas des agrégats')
print('========================================================')
job_date_time='optimization'+time.strftime('[%X - %x]')
print(job_date_time, 'Start time - Case I')
print('\n')



C_ntc_V= np.array([[40.7, 39.3, 12.4, 0., 0., 0.],
                      [39.3, 40.7, 12.4, 0., 0., 0.],
                      [12.4, 12.4, 625.7, 0., 0., 0.],
                      [0., 0., 0., 2.44, 0., 0.],
                      [0., 0., 0., 0., 2.44, 0.],
                      [0., 0., 0., 0., 0., 1.36]])

C4_ntc = tensor2_Voigt_to_tensor4_NOT_Major_Sym(C_ntc_V)



def P_theta_phi_betha_ntc(b,p,t):
    P_theta_phi_betha = np.zeros((3,3))

    P_theta_phi_betha[0][0] = cos(t)*cos(p)*cos(b) - sin(p)*sin(b)
    P_theta_phi_betha[0][1] = -cos(t)*cos(p)*sin(b) - sin(p)*cos(b)
    P_theta_phi_betha[0][2] = sin(t)*cos(p)

    P_theta_phi_betha[1][0] = cos(t)*sin(p)*cos(b) + cos(p)*sin(b)
    P_theta_phi_betha[1][1] = -cos(t)*sin(p)*sin(b) + cos(p)*cos(b)
    P_theta_phi_betha[1][2] = sin(t)*sin(p)

    P_theta_phi_betha[2][0] = -sin(t)*cos(b)
    P_theta_phi_betha[2][1] = sin(t)*sin(b)
    P_theta_phi_betha[2][2] = cos(t)
    
    return P_theta_phi_betha



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


excel_file_0pi_theta='0pi_16points.xls'
excel_file_02pi_phi='02pi_16points.xls'
excel_file_02pi_beta='02pi_16points.xls'


(x_0pi_theta,  W_0pi_theta)=import_Guass_points_x_Wx(excel_file_0pi_theta)
(x_02pi_phi,   W_02pi_phi)=import_Guass_points_x_Wx(excel_file_02pi_phi)
(x_02pi_betha, W_02pi_betha)=import_Guass_points_x_Wx(excel_file_02pi_beta)

n_theta = len(W_0pi_theta)
n_phi = len(W_02pi_phi)
n_betha = len(W_02pi_betha)

def W_02pi_02pi_0pi(W_02pi_betha, W_02pi_phi, W_0pi_theta):
    'Weights of omega and zeta3'   
    W_theta_phi_betha=np.zeros((n_betha, n_phi, n_theta))
    for b in range(0, n_betha):
        for p in range(0, n_phi):
            for t in range(0, n_theta):
                                
                W_theta_phi_betha[b,p,t]= W_02pi_betha[b] * W_02pi_phi[p] * W_0pi_theta[t]
                
    return W_theta_phi_betha


W_theta_phi_betha = W_02pi_02pi_0pi(W_02pi_betha, W_02pi_phi, W_0pi_theta)





def P_total_ntc(x_02pi_betha, x_02pi_phi, x_0pi_theta):
    P=np.zeros((3,3,n_betha, n_phi, n_theta))
    for b in range(0, n_betha):
        for p in range(0, n_phi):
            for t in range(0, n_theta):
                P_btp=P_theta_phi_betha_ntc(x_02pi_betha[b], x_02pi_phi[p], x_0pi_theta[t])
                for i in range(0,3):
                    for j in range(0,3):
                        
                        P[i][j][b][p][t] = P_btp[i][j]
    return P
    

P=P_total_ntc(x_02pi_betha, x_02pi_phi, x_0pi_theta)


def sin_theta_ntc(x_0pi_theta):
    sin_theta=np.zeros((n_theta,1))
    for t in range(0, n_theta):
        sin_theta[t]=sin(x_0pi_theta[t])
    return sin_theta

sin_theta = sin_theta_ntc(x_0pi_theta)



print('\n')

def C_ntc_ave(C4_ntc, P, W_theta_phi_betha, sin_theta):
#    C=np.zeros((3,3,3,3))
    C=np.zeros((6,6))
    
    for b in range(0, n_betha):
        for ph in range(0, n_phi):
            for t in range(0, n_theta):    
    
                for i in range(0,3):
                    for j in range(0,3):
                        for k in range(0,3):
                            for l in range(0,3):
#                                for m in range(0,3):
#                                    for n in range(0,3):
#                                        for o in range(0,3):
#                                            for p in range(0,3):
#                                                                      
#                                                C[m][n][o][p] = (1./(8.*pow(np.pi,2))) * (P[i][m][b][ph][t]) * (P[j][n][b][ph][t]) *  (P[k][o][b][ph][t]) * (P[l][p][b][ph][t]) *  C4_ntc[i][j][k][l] * sin_theta[t]  * W_theta_phi_betha[b][ph][t]  +     C[m][n][o][p]
                                #C4[m][n][o][p] ---> C2[x][y]
                                #m=n=p=o=0
                                C[0][0] = (1./(8.*pow(np.pi,2))) * (P[i][0][b][ph][t]) * (P[j][0][b][ph][t]) *  (P[k][0][b][ph][t]) * (P[l][0][b][ph][t]) *  C4_ntc[i][j][k][l] * sin_theta[t]  * W_theta_phi_betha[b][ph][t]  +     C[0][0]
                                #m=n=p=o=1
                                C[1][1] = (1./(8.*pow(np.pi,2))) * (P[i][1][b][ph][t]) * (P[j][1][b][ph][t]) *  (P[k][1][b][ph][t]) * (P[l][1][b][ph][t]) *  C4_ntc[i][j][k][l] * sin_theta[t]  * W_theta_phi_betha[b][ph][t]  +     C[1][1]
                                #m=n=p=o=2
                                C[2][2] = (1./(8.*pow(np.pi,2))) * (P[i][2][b][ph][t]) * (P[j][2][b][ph][t]) *  (P[k][2][b][ph][t]) * (P[l][2][b][ph][t]) *  C4_ntc[i][j][k][l] * sin_theta[t]  * W_theta_phi_betha[b][ph][t]  +     C[2][2]
                                
                                #m=n=0; o=p=1
                                C[0][1] = (1./(8.*pow(np.pi,2))) * (P[i][0][b][ph][t]) * (P[j][0][b][ph][t]) *  (P[k][1][b][ph][t]) * (P[l][1][b][ph][t]) *  C4_ntc[i][j][k][l] * sin_theta[t]  * W_theta_phi_betha[b][ph][t]  +     C[0][1]
                                #m=n=0; o=p=2
                                C[0][2] = (1./(8.*pow(np.pi,2))) * (P[i][0][b][ph][t]) * (P[j][0][b][ph][t]) *  (P[k][2][b][ph][t]) * (P[l][2][b][ph][t]) *  C4_ntc[i][j][k][l] * sin_theta[t]  * W_theta_phi_betha[b][ph][t]  +     C[0][2]
                                #m=n=1; o=p=2
                                C[1][2] = (1./(8.*pow(np.pi,2))) * (P[i][1][b][ph][t]) * (P[j][1][b][ph][t]) *  (P[k][2][b][ph][t]) * (P[l][2][b][ph][t]) *  C4_ntc[i][j][k][l] * sin_theta[t]  * W_theta_phi_betha[b][ph][t]  +     C[1][2]
                                
                                #m=1;n=2; o=1;p=2
                                C[3][3] = 2.*(1./(8.*pow(np.pi,2))) * (P[i][1][b][ph][t]) * (P[j][2][b][ph][t]) *  (P[k][1][b][ph][t]) * (P[l][2][b][ph][t]) *  C4_ntc[i][j][k][l] * sin_theta[t]  * W_theta_phi_betha[b][ph][t]  +     C[3][3]
                                #m=2;n=0; o=2;p=0
                                C[4][4] = 2.*(1./(8.*pow(np.pi,2))) * (P[i][2][b][ph][t]) * (P[j][0][b][ph][t]) *  (P[k][2][b][ph][t]) * (P[l][0][b][ph][t]) *  C4_ntc[i][j][k][l] * sin_theta[t]  * W_theta_phi_betha[b][ph][t]  +     C[4][4]
                                #m=0;n=1; o=0;p=1
                                C[5][5] = 2.*(1./(8.*pow(np.pi,2))) * (P[i][0][b][ph][t]) * (P[j][1][b][ph][t]) *  (P[k][0][b][ph][t]) * (P[l][1][b][ph][t]) *  C4_ntc[i][j][k][l] * sin_theta[t]  * W_theta_phi_betha[b][ph][t]  +     C[5][5]
                                
    
    for i in range(0,6):
        for j in range((i+1),6):
            C[j][i]=C[i][j]
                                                
    return C                                
            



C_ntc_ave_V=C_ntc_ave(C4_ntc, P, W_theta_phi_betha,sin_theta)
C_ntc_ave_4=tensor2_Voigt_to_tensor4_NOT_Major_Sym(C_ntc_ave_V)

print('C_ntc_ave_V=')
print(C_ntc_ave_V)

E0=2.
v0=0.3
k0=E0/(3.*(1.-2.*v0))
mu0=E0/(2.*(1.+v0))

(alpha, betha)=alpha_betha_4orten_isotropes(C_ntc_ave_4)
kf=(1./3.)*alpha
muf=(1./2.)*betha

cf=0.05


print('--------------------------------')
print('Answer for Mori_Tanaka_sphériques ')

(k_homogen, mu_homogen, C_Voigt_homogen) = Mori_Tanaka_sphériques(k0,mu0,[kf],[muf],[cf])
print('\n')

print('k_ave_sph_MT=',k_homogen) 
print('mu_ave_sph_MT=',mu_homogen) 


E_agg=(9.*k_homogen*mu_homogen) / (mu_homogen + 3.*k_homogen)
v_agg=(3.*k_homogen - 2*mu_homogen) / (2*mu_homogen + 6*k_homogen)      
print('\n')

print('E_agg=',E_agg) 
print('v_agg=',v_agg) 
print('\n')

job_date_time='optimization'+time.strftime('[%X - %x]')
print(job_date_time, 'End time - Case I')
print('\n')

print('deltat_old = 3:18  ----> deltat_optimized= 0:21')


print('==========================  2  ==============================')
print('Le cas des nano-tubes de carbone séparés et distribués aléatoirement')
print('========================================================')
job_date_time='optimization'+time.strftime('[%X - %x]')
print(job_date_time, 'Start time - Case II')

print('\n')

print('OPTIMIZED CODE.......')
print('''In this case you should put the stiffness nesor of unidirection fiber. (Do not make mistake again. It's not average tensor)''')


#matrix properties
E0=2.
v0=0.3
k0=E0/(3.*(1.-2.*v0))
mu0=E0/(2.*(1.+v0))
(I,J,K)=I_J_K_4orten_isotropes()
C0_4=3.*k0*J + 2.*mu0*K


#fiber properties
a1=1.; a2=1.; a3=500.
(alpha, betha)=alpha_betha_4orten_isotropes(C_ntc_ave_4)
kf=(1./3.)*alpha
muf=(1./2.)*betha

cf=0.05


excel_file_omega='omega_512points.xls'
excel_file_zeta3='zeta3_4points.xls'


#BUT I didn't use this modifier
def Tensor2_modifier(T):
    'if abs(T) < 1.0e-7 ----> SE=0'
    for i in range(0,3):
        for j in range(0,3):
            if abs(T[i][j]) < 1.0e-4:
                T[i][j]=0.
    return T


# PUT THE TENSOR OF FIBER (JUST FIBER, UNIDIRECTION FIBER)
Tr = Mori_Tanaka_elp_ONEPHASE_Tr_INPUT_Cf(k0,mu0,C_ntc_V,cf,excel_file_zeta3, excel_file_omega,a1,a2,a3)



ArV_ntc = Mori_Tanaka_elp_ONEPHASE_Ar_INPUT_Cf(Tr,k0,mu0,C_ntc_V,cf)
Ar4_ntc = tensor2_Voigt_to_tensor4_NOT_Major_Sym(ArV_ntc)



def C_ale_ave(C4_ntc, C0_4, P, W_theta_phi_betha, sin_theta, Ar4_ntc,cf):
    
    C0_V = tensor4_to_Voigt_tensor2_NOT_Major_Sym(C0_4)
#    C= np.zeros((3,3,3,3)) + C0_4
    C= np.zeros((6,6)) + C0_V
    CCA = np.zeros((3,3,3,3))
    CCA = tensor4_contract2_tensor4(C4_ntc - C0_4, Ar4_ntc)

     #OPTIMIZED -2: Using tensorial function-------> Reduce 2 Loops
#    for i in range(0,3):
#        for j in range(0,3):
#            for k in range(0,3):
#                for l in range(0,3):
#                    for r in range(0,3):
#                        for s in range(0,3):
#                            CCA[i][j][k][l] = ((C4_ntc[i][j][r][s] - C0_4[i][j][r][s]) * Ar4_ntc[r][s][k][l]) + CCA[i][j][k][l]
#     -----------------------------------------------                          

    #OPTIMIZED -1: Just calculate 9 elements---------> Reduce 4 Loops           
    for b in range(0, n_betha):
        for ph in range(0, n_phi):
            for t in range(0, n_theta):    
    
                for i in range(0,3):
                    for j in range(0,3):
                        for k in range(0,3):
                            for l in range(0,3):
                                
                                
#                                for m in range(0,3):
#                                    for n in range(0,3):
#                                        for o in range(0,3):
#                                            for p in range(0,3):
#                                                          
#                                                        C[m][n][o][p] = (cf/(8.*pow(np.pi,2))) * (P[i][m][b][ph][t]) * (P[j][n][b][ph][t]) *  (P[k][o][b][ph][t]) * (P[l][p][b][ph][t]) *  (CCA[i][j][k][l])  * sin_theta[t]  * W_theta_phi_betha[b][ph][t]  +     C[m][n][o][p]
                                
                                                                #C4[m][n][o][p] ---> C2[x][y]
                                 #--------------------------------------------------------------------                               
                                 #m=n=p=o=0
                                 C[0][0] = (cf/(8.*pow(np.pi,2))) * (P[i][0][b][ph][t]) * (P[j][0][b][ph][t]) *  (P[k][0][b][ph][t]) * (P[l][0][b][ph][t]) *  (CCA[i][j][k][l])  * sin_theta[t]  * W_theta_phi_betha[b][ph][t]  +     C[0][0]
                                 #m=n=p=o=1
                                 C[1][1] = (cf/(8.*pow(np.pi,2))) * (P[i][1][b][ph][t]) * (P[j][1][b][ph][t]) *  (P[k][1][b][ph][t]) * (P[l][1][b][ph][t]) *  (CCA[i][j][k][l])  * sin_theta[t]  * W_theta_phi_betha[b][ph][t]  +     C[1][1]
                                 #m=n=p=o=2
                                 C[2][2] = (cf/(8.*pow(np.pi,2))) * (P[i][2][b][ph][t]) * (P[j][2][b][ph][t]) *  (P[k][2][b][ph][t]) * (P[l][2][b][ph][t]) *  (CCA[i][j][k][l])  * sin_theta[t]  * W_theta_phi_betha[b][ph][t]  +     C[2][2]

                                 #m=n=0; o=p=1
                                 C[0][1] = (cf/(8.*pow(np.pi,2))) * (P[i][0][b][ph][t]) * (P[j][0][b][ph][t]) *  (P[k][1][b][ph][t]) * (P[l][1][b][ph][t]) *  (CCA[i][j][k][l])  * sin_theta[t]  * W_theta_phi_betha[b][ph][t]  +     C[0][1]
                                 #m=n=0; o=p=2
                                 C[0][2] = (cf/(8.*pow(np.pi,2))) * (P[i][0][b][ph][t]) * (P[j][0][b][ph][t]) *  (P[k][2][b][ph][t]) * (P[l][2][b][ph][t]) *  (CCA[i][j][k][l])  * sin_theta[t]  * W_theta_phi_betha[b][ph][t]  +     C[0][2]
                                 #m=n=1; o=p=2
                                 C[1][2] = (cf/(8.*pow(np.pi,2))) * (P[i][1][b][ph][t]) * (P[j][1][b][ph][t]) *  (P[k][2][b][ph][t]) * (P[l][2][b][ph][t]) *  (CCA[i][j][k][l])  * sin_theta[t]  * W_theta_phi_betha[b][ph][t]  +     C[1][2]
                                 
                                 #m=1;n=2; o=1;p=2
                                 C[3][3] = 2.*(cf/(8.*pow(np.pi,2))) * (P[i][1][b][ph][t]) * (P[j][2][b][ph][t]) *  (P[k][1][b][ph][t]) * (P[l][2][b][ph][t]) *  (CCA[i][j][k][l])  * sin_theta[t]  * W_theta_phi_betha[b][ph][t]  +     C[3][3]
                                 #m=2;n=0; o=2;p=0
                                 C[4][4] = 2.*(cf/(8.*pow(np.pi,2))) * (P[i][2][b][ph][t]) * (P[j][0][b][ph][t]) *  (P[k][2][b][ph][t]) * (P[l][0][b][ph][t]) *  (CCA[i][j][k][l])  * sin_theta[t]  * W_theta_phi_betha[b][ph][t]  +     C[4][4]
                                 #m=0;n=1; o=0;p=1
                                 C[5][5] = 2.*(cf/(8.*pow(np.pi,2))) * (P[i][0][b][ph][t]) * (P[j][1][b][ph][t]) *  (P[k][0][b][ph][t]) * (P[l][1][b][ph][t]) *  (CCA[i][j][k][l])  * sin_theta[t]  * W_theta_phi_betha[b][ph][t]  +     C[5][5]
                                                           
    for i in range(0,6):
        for j in range((i+1),6):
            C[j][i]=C[i][j]                                                    
                                                        
                                                        
    return C
                                                        
                                                        
                                                    

C_ale_ave_MT_V= C_ale_ave(C4_ntc, C0_4, P, W_theta_phi_betha, sin_theta, Ar4_ntc,cf)
C_ale_ave_MT=tensor2_Voigt_to_tensor4_NOT_Major_Sym(C_ale_ave_MT_V)



print('C_ale_ave_MT_V=')
print(C_ale_ave_MT_V)

print('\n')
                                                        
(alpha_ale_ave_MT, betha_ale_ave_MT)=alpha_betha_4orten_isotropes(C_ale_ave_MT)
kf_ale_ave_MT=(1./3.)*alpha_ale_ave_MT
muf_ale_ave_MT=(1./2.)*betha_ale_ave_MT

print('kf_ale_ave_MT=', kf_ale_ave_MT)
print('muf_ale_ave_MT=', muf_ale_ave_MT) 
                                       
print('\n')

print('in this case the correct value is k_ale=5.17.')
print('I talked to LY he has 5.148. so maybe error is acceptable')

E_ale=(9.*kf_ale_ave_MT*muf_ale_ave_MT) / (muf_ale_ave_MT + 3.*kf_ale_ave_MT)
v_ale=(3.*kf_ale_ave_MT - 2.*muf_ale_ave_MT) / (2.*muf_ale_ave_MT + 6.*kf_ale_ave_MT)      

print('E_ale=',E_ale) 
print('v_ale=',v_ale) 


print('\n')

job_date_time='optimization'+time.strftime('[%X - %x]')
print(job_date_time, 'End time - Case II')   
print('\n')
    
print('==========================  3  ==============================')
print('Le cas des nano-tubes de carbone parfaitement alignés')
print('========================================================')
print('\n')
job_date_time='optimization'+time.strftime('[%X - %x]')
print(job_date_time, 'Start time - Case III') 
print('\n')
print('tous les nano-tubes seraient alignés selon l’axe x3') 
print('\n')

C_ntc_V= np.array([[40.7, 39.3, 12.4, 0., 0., 0.],
                      [39.3, 40.7, 12.4, 0., 0., 0.],
                      [12.4, 12.4, 625.7, 0., 0., 0.],
                      [0., 0., 0., 2.44, 0., 0.],
                      [0., 0., 0., 0., 2.44, 0.],
                      [0., 0., 0., 0., 0., 1.36]])
  
    
#matrix properties..............    
E0=2.
v0=0.3
k0=E0/(3.*(1.-2.*v0))
mu0=E0/(2.*(1.+v0))
(I,J,K)=I_J_K_4orten_isotropes()
C0_4=3.*k0*J + 2.*mu0*K


#fiber properties
a1=1.; a2=1.; a3=500.
cf=0.05  

excel_file_omega='omega_512points.xls'
excel_file_zeta3='zeta3_4points.xls' 


Cfr = C_ntc_V
C_homogen =   Mori_Tanaka_ellipsoidales_ONEPHASE_INPUT_Cf(k0,mu0,Cfr,cf,excel_file_zeta3, excel_file_omega,a1,a2,a3)  




   
print('CV_Mori_Tanaka_ellipsoidales=')    
print(C_homogen)
print('\n')

#This is just when 3 (z) is the direction of fiber
(El_MT,Et_MT,vl_MT,vt_MT,Gl_MT,n_MT)=El_Et_vl_vt_Gl_n_isotrope_transverse(C_homogen)

print('El_MT=', El_MT)
print('Et_MT=', Et_MT)

print('\n')


job_date_time='optimization'+time.strftime('[%X - %x]')
print(job_date_time, 'End time - Case III') 

print('==========================  4  ==============================')
print('Comparaisons')
print('========================================================')

print('\n')
print('Le cas des agrégats:')
print('E_agg=',E_agg) 
print('v_agg=',v_agg) 

print('\n')
print('Le cas des nano-tubes de carbone séparés et distribués aléatoirement:')
print('E_ale=',E_ale) 
print('v_ale=',v_ale) 

print('\n')

print('Le cas des nano-tubes de carbone parfaitement alignés:')
print('El_MT=E33', El_MT)
print('Et_MT=E11', Et_MT)




print('\n')
print('*************************************************************')
print('-------------NOTE-----------------')
print('*************************************************************')
print('\n')

print('''OPTIMIZATION THE CODE:
    The idea is that instead of calculation the whole 4th-order tensor (81 elements),
    we can just calculate the significants elements (I saw this in Rooland and Ilyass code.
    I think this is kind of tricky simplification. But we know that all other elements is zero. 
    It means that we first calculate the whole 81 components and when we get sure about the
    correct values, then we can calculate just 9 components.
    Son in the following I will change the code. it will reduce 4-loops. We already knew the symmetric of tensor''')
print('----------------------------------------------------------------------')

print('\n')

print('----------------------------------------------------------------------')
print('My note about optimized code')
print('''My programm is not fully optimized. it takes 5 minutes. but I want to say something for having better code'
    I don't have time to change the all functions. but I'm sure that if we do these modification. It will be defiently better.
    1- I have very small values. for example in order of 10^-17. so these values should be zero. (absoulut zero). It needs to modify the Mori_tanaka functions and I think the voight notation. maybe
    2- np.matmul for matrix production is not very optimized operator. I think using tensoriel production is better.
    3- Ilyass used fifferent way to calculate the tnesnor C. in case 2. You can check it too. but it's so risky. to use that.''')

print('\n')

print('----------------------------------------------------------------------')
print('MY FAULTS:')

print('''
      1- I didn't write the tensoriel product correctly.
      I forget the summation. Please don't forget that every time we have tensorial
      multipication we should consider sum_temp in our summation.
      2- In the case II: I put the stiffness tensor of average direction which is comppletely wrong.
      We should put the stiffness tensor of unidirectional fiber. We will calculate the average integral in the Moritanaka. eq. 11.
      Why did you put the average tensor. It said in the equation A_ntc C_ntc.
      
      I spent the whole two nights. I thought becuase of numerical errors. I modifeid every tensor. Because my value was not that much far fron the correct one
      (I got 5.25 and the answer was 5.17 So I thought maybe it's because of some errors. Because other guys had errors too. 
      But, it was not becuase of numerical errors. It was about my misunderestandying. I didn't underestand the concepts. It was not just a numerical errors.)
      ''')



