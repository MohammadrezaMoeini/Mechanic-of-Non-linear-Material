# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 19:53:05 2018

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

print('The end time: Calculation of Eshelby tensor for Elipsoidan and sperical shapes')
job_date_time='optimization'+time.strftime('[%X - %x]')
print(job_date_time, 'Start time')

print('========================================================')  
print('Problème 07 – 01 : Calculs sur le tenseur d’Eshelby')                                                               
print('========================================================')  

# =============================================================================
# Result: Inclusions sphériques
# =============================================================================

print('Inclusions sphériques ') 
print('P = 4 et Q = 32, où: nombre de quadratures') 
print('k0 = mu0 = 1')
print('a1 = a2 = a3 = 1.')
print('...............')     

k0=mu0=1.
a1=a2=a3=1.

excel_file_omega='omega_32points.xls'
excel_file_zeta3='zeta3_4points.xls'

SE_sph_numerical=Eshelby_tensor_ellipsoid(excel_file_zeta3, excel_file_omega, k0, mu0,a1,a2,a3)
(alpha_num, betha_num)=alpha_betha_4orten_isotropes(SE_sph_numerical)
SE_sph_numerical_Voigt=tensor4_to_Voigt_tensor2_NOT_Major_Sym(SE_sph_numerical)

print('SE_sph_numerical_Voigt=')
print(SE_sph_numerical_Voigt)
print('alpha_num=',alpha_num)
print('betha_num=',betha_num)

print('\n')

SE_sph_analytical=Eshelby_tensor_Spherical(k0,mu0)
(alpha_anl, betha_anl)=alpha_betha_4orten_isotropes(SE_sph_analytical)
SE_sph_analytical_Voigt=tensor4_to_Voigt_tensor2_NOT_Major_Sym(SE_sph_numerical)
print('SE_sph_analytical_Voigt=')
print(SE_sph_analytical_Voigt)
print('alpha_anl=',alpha_anl)
print('betha_anl=',betha_anl)



#print('SE_sph_numerical------SE_sph_analytica') 
#for i in range(0,3):
#    for j in range(0,3):
#        for k in range(0,3):
#            for l in range(0,3):
#                print('[i,j,k,l]',[i,j,k,l],SE_sph_numerical[i][j][k][l],'-----',SE_sph_analytical[i][j][k][l])
#                print('-----------')
#



print('\n')

print('========================================================')          
print('Fibre longue alignée selon l’axe x3') 
print('P = 4 et Q = 512: le nombre de quadratures') 
print('k0 = mu0 = 1')
print('a1 = a2 =1,  a3 = 10000.')
print('...............')     



k0=mu0=1.
a1=a2=1.
a3=1.0e4

excel_file_omega='omega_512points.xls'
excel_file_zeta3='zeta3_4points.xls'

#SE_elip_numerical=Eshelby_tensor_ellipsoid(excel_file_zeta3, excel_file_omega, k0, mu0,a1,a2,a3)

(I0,J0,K0)=I_J_K_4orten_isotropes()
C0= 3.*k0*J0 + 2.*mu0*K0
SE_elip_numerical=Eshelby_tensor_ellipsoid_C0Input(excel_file_zeta3, excel_file_omega, C0, a1, a2, a3)



v=1./8.
SE_3333=0.
SE_1111 = SE_2222 = (5.-4.*v) / (8.*(1.-v))
SE_1212 = (3. - 4.*v)/(8.*(1.-v))
SE_3131 = SE_3232 = 1./4.
SE_1122 = SE_2211 = (4.*v - 1.) /  (8.*(1.-v))
SE_1133 = SE_2233= v/(2.*(1.-v))
SE_3311=SE_3322=0.

print('-----------------------------------')
print(SE_elip_numerical[2][2][2][2], 'S_Anl=', SE_3333)
print('-----------------------------------')
print(SE_elip_numerical[0][0][0][0], 'S_Anl=', SE_1111)
print('-----------------------------------')
print(SE_elip_numerical[1][1][1][1], 'S_Anl=', SE_2222)
print('-----------------------------------')
print(SE_elip_numerical[0][1][0][1], 'S_Anl=', SE_1212)
print('-----------------------------------')
print(SE_elip_numerical[2][0][2][0], 'S_Anl=', SE_3131)
print('-----------------------------------')
print(SE_elip_numerical[2][1][2][1], 'S_Anl=', SE_3232)
print('-----------------------------------')
print(SE_elip_numerical[0][0][1][1], 'S_Anl=', SE_1122)
print('-----------------------------------')
print(SE_elip_numerical[1][1][0][0], 'S_Anl=', SE_2211)
print('-----------------------------------')
print(SE_elip_numerical[0][0][2][2], 'S_Anl=', SE_1133)
print('-----------------------------------')
print(SE_elip_numerical[1][1][2][2], 'S_Anl=', SE_2233)
print('-----------------------------------')
print(SE_elip_numerical[2][2][0][0], 'S_Anl=', SE_3311)
print('-----------------------------------')
print(SE_elip_numerical[2][2][1][1], 'S_Anl=', SE_3322)
print('\n')
print('-----------------------------------')
print('-----------------------------------')

SE_elip_numerical_Voigt=tensor4_to_Voigt_tensor2_NOT_Major_Sym(SE_elip_numerical)




print('The end time: Calculation of Eshelby tensor for Elipsoidan and sperical shapes')
job_date_time='optimization'+time.strftime('[%X - %x]')
print(job_date_time, 'Start time')









