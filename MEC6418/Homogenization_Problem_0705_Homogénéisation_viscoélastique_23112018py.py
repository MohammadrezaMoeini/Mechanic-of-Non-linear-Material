# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 16:32:22 2018

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



from sympy.integrals import laplace_transform, inverse_laplace_transform
from sympy.abc import t, s, X
import sympy as sy




print('**********************************************')
print('Problème 07 – 05 : Homogénéisation viscoélastique')
print('**********************************************')

print('\n')


#tow=[0.003593813663804626, 0.01, 0.027825594022071243, 0.0774263682681127,
#     0.21544346900318834, 0.5994842503189409, 1.6681005372000592,
#     4.6415888336127775, 12.91549665014884, 35.93813663804626]


a=np.linspace( -2, 2, 10 )
tow=[]
for i in a:
    tow.append(pow(10,i))

print('tow=')
print(tow)
print('\n')


# =============================================================================
# Writing the laplace_carson manually . NOT GOOD IDEA
# =============================================================================
def k0_LP(p):
    landa1=1.
    landa2=1./pow(10.,1.5)
    
    k0_LP = 10. + 1.2*(p/(p + landa1)) + 0.5*(p/(p + landa2))
    return k0_LP

def mu0_LP(p):
    landa1=1.
    landa2=1./pow(10.,1.5)
    
    mu0_LP = 2. + 0.1*(p/(p + landa1)) + 0.5*(p/(p + landa2))
    return mu0_LP

def k1_LP(p):
    landa1=1./pow(10.,0.5)
    landa2=1./pow(10.,1.5)
        
    k1_LP = 100. + 15.*(p/(p + landa1)) + 8.*(p/(p + landa2))
    return k1_LP

def mu1_LP(p):
    landa1=1./pow(10.,0.9)
    landa2=1./pow(10.,1.2)
    
    mu1_LP = 34. + 3.*(p/(p + landa1)) + 12.*(p/(p + landa2))
    return mu1_LP





# =============================================================================
def laplace_carson( f, t, s ):
	l = laplace_transform( f, t, s)
	l_c = sy.simplify( s*l[0] )
	
	return l_c

#Why s*l[0] ?
    #Example:
    #f = sy.sin(-t)
    #l = laplace_transform( f, t, s)
    #l = (-1/(s**2 + 1), 0, True)
    #l[0] = -1/(s**2 + 1)
# =============================================================================
   

print('---------------------------------------------')
print('Functions with respect to time (Time Domain):')

landa_k01=1.; landa_k02=1/(pow(10,1.5))
k0_t = 10. + 1.2*(sy.exp(-landa_k01*t)) + 0.5*(sy.exp(-landa_k02*t))
print('k0_t=')
print(k0_t)
print('\n')

landa_mu01=1.; landa_mu02=1/(pow(10,1.5))
mu0_t = 2. + 0.1*(sy.exp(-landa_mu01*t)) + 0.5*(sy.exp(-landa_mu02*t))
print('mu0_t=')
print(mu0_t)
print('\n')

landa_k11=1./(pow(10,0.5)); landa_k12=1./(pow(10,1.5))
k1_t = 100. + 15.*(sy.exp(-landa_k11*t)) + 8.*(sy.exp(-landa_k12*t))
print('k1_t=')
print(k1_t)
print('\n')

landa_k11=1./(pow(10,0.9)); landa_k12=1./(pow(10,1.2))
mu1_t = 34. + 3.*(sy.exp(-landa_k11*t)) + 12.*(sy.exp(-landa_k12*t))
print('mu1_t')
print(mu1_t)
print('\n')



print('---------------------------------------------')
print('Inverting to Laplace_Carson Domain:')
k0_s = laplace_carson( k0_t, t, s )
print('k0_s=')
print(k0_s)
print('\n')

mu0_s = laplace_carson( mu0_t, t, s )
print('mu0_s=')
print(mu0_s)
print('\n')

k1_s = laplace_carson( k1_t, t, s )
print('k1_s=')
print(k1_s)
print('\n')

mu1_s = laplace_carson( mu1_t, t, s )
print('mu1_s=')
print(mu1_s)
print('\n')


print('---------------------------------------------')
print('Volume fraction:')
c1=0.3
c0=1. - c1
print('c1=',c1)
       


print('---------------------------------------------')
print('Homogenization:')

def alpha_betha_homogen_in_Laplace_Carson(k0,mu0,k1,mu1, c1):
    c0=1. - c1
    
    alpha_SE=(3.*k0)/(3.*k0 + 4.*mu0)
    betha_SE=(6.*(k0 + 2.*mu0)) / (5.*(3.*k0 + 4.*mu0))

    alpha_T  = 1./ (   1. + (alpha_SE) * (1/(3.*k0)) * (3*k1 - 3*k0)     )
    betha_T = 1./ (    1. + (betha_SE) * (1/(2.*mu0)) * (2*mu1 - 2*mu0)      )
   
    alpha_A = alpha_T*(1./(1.*c0 + c1*alpha_T))
    betha_A = betha_T*(1./(1.*c0 + c1*betha_T))
    
    alpha_Chomogen = 3.*k0 + c1*(3.*k1 - 3.*k0)*(alpha_A)
    betha_Chomogen = 2.*mu0 + c1*(2.*mu1 - 2.*mu0)*(betha_A)
    
    return alpha_Chomogen, betha_Chomogen
    

(alpha_Chomogen, betha_Chomogen) = alpha_betha_homogen_in_Laplace_Carson(k0_s, mu0_s, k1_s, mu1_s, c1)



def alpha_betha_k_mu_in_time_domain(tow, alpha_Chomogen, betha_Chomogen):
    N=len(tow)
    deltasai_alpha=np.zeros((N,1))
    deltasai_betha=np.zeros((N,1))

    for i in range(0,N):
        deltasai_alpha[i]=alpha_Chomogen.subs(s,tow[i]) - alpha_Chomogen.subs(s,0.)
    
        deltasai_betha[i]=betha_Chomogen.subs(s,tow[i]) - betha_Chomogen.subs(s,0.)
    
    
    kesi=np.zeros((N,N))
    for i in range(0,len(tow)):
        for j in range(0,len(tow)):
            kesi[i][j]= tow[i]/(tow[i]+tow[j])
                

    phi_alpha = np.matmul(inv(kesi) , deltasai_alpha)
    phi_betha = np.matmul(inv(kesi) , deltasai_betha)


    sum_tmp=0.
    for i in range(0,N):
        sum_tmp = phi_alpha[i]* (sy.exp(-t*tow[i])) + sum_tmp 
        
    alpha_Chomogen_t = alpha_Chomogen.subs(s,0.) + sum_tmp

    ft_alpha_homogen = alpha_Chomogen_t [0]

    sum_tmp=0.
    for i in range(0,N):
        sum_tmp = phi_betha[i]* (sy.exp(-t*tow[i])) + sum_tmp 
        
    betha_Chomogen_t = betha_Chomogen.subs(s,0.) + sum_tmp

    ft_betha_homogen = betha_Chomogen_t [0]
    
    
    return ft_alpha_homogen, ft_betha_homogen
    

(ft_alpha_homogen, ft_betha_homogen) = alpha_betha_k_mu_in_time_domain(tow, alpha_Chomogen, betha_Chomogen)

#USE THESE ALPHA AND BETHA IN THIS WAY

k= (1./3.) * ft_alpha_homogen.subs(t,1.5)
mu= (1./2.) * ft_betha_homogen.subs(t,2.3)
print('k @1.5 = ',k)
print('mu @2.3 = ',mu)



# =============================================================================
# PLOT THE RESULTS WITH RESPECT TO TIME
# =============================================================================
#print('---------------------------------------------')
#print('Plot the results with respect to time:')
#
#time=np.linspace( 0, 3, 100 )
#k0_t_curv=np.zeros(shape(time))
#mu0_t_curv=np.zeros(shape(time))
#k1_t_curv=np.zeros(shape(time))
#mu1_t_curv=np.zeros(shape(time))
#
#k_t_curv_homogen=np.zeros(shape(time))
#mu_t_curv_homogen=np.zeros(shape(time))
#
#
#for i in range(0, len(time)):
#    k0_t_curv[i] = k0_t.subs(t, time[i])
#    mu0_t_curv[i] = mu0_t.subs(t, time[i])
#    
#    k1_t_curv[i] = k1_t.subs(t, time[i])
#    mu1_t_curv[i] = mu1_t.subs(t, time[i])
#    
#    k_t_curv_homogen[i] = (1./3.) * ft_alpha_homogen.subs(t, time[i])
#    mu_t_curv_homogen[i] = (1./2.) * ft_betha_homogen.subs(t, time[i])
#    
#    
#figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
#
#plt.plot(time, k0_t_curv, 'b--.')
#plt.plot(time, mu0_t_curv, 'b--+')
#
#plt.plot(time, k1_t_curv, 'g--.')
#plt.plot(time, mu1_t_curv, 'g--+')
#
#plt.plot(time, k_t_curv_homogen, 'r--.')
#plt.plot(time, mu_t_curv_homogen, 'r--+')
#
#plt.xlabel('time [seconds]')
#plt.ylabel('k,mu [GPa]')
#plt.title('VISCOELASTIC MATERIALS: k0,mu0,k1,mu1 -----> k,mu homogen')
#plt.legend(('k0','mu0','k1','mu1','k_homogen','mu_homogen'),
#               shadow=True, loc=(0.4, 0.5), handlelength=1.5, fontsize=14)
#    
#plt.grid(True)
#plt.axis([0, 3, 0.0, 150.])
#plt.show()



print('----------------------------------------------------------------------')
print('---------------------------  NOTE ----------------------------------')


print('''
      I wrote my function last night. The answer was not even close
      the problem was related to eq.57. I considered t as 1/kesi[i][j]
      Why????!!!!! it's inverse matrix
      tonight I noticed. that. but before that I learned alot.
      Because I wrote all functions again. 
      LY told me you cannot use your moritanaka (that actually we used that,
      this was not my problem in prevoius function
      .The prof also tols is ok.... and it's ok.
      but any way I wrote everything in paper again. and I recalculate again.
      After that I get same wrong result. Because this was not the fucking problem.
      I checked Ilyass code. I saw that he used laplase_transform function in python.
      GOOD IDEA. So I used this laplace transform and crorresponding laplace carson function.
      but I got the same wrong answer. BECAUSE IT WAS NOT THE FUUUUCKING PROBLEM.
      However, I leaned laplace_transformation and how to write a function with
      variable I mean a real variable not with def ...
      It's very cool. you can write any function as you want you just need to 
      write an import.
      to import your name of your variables. So. after that you can use
      laplace carson function. in to your function. It's reaaly better rather
      than writing manually. 
      So I really suggest to use this way. 
      
      First I had bad feeling to see Ilyass code. But now I'm feel more comfortable
      that his code just help me to improve my code. I already got the correct resutls. 
      END of th story 23-11-2018- 23Nov 2018.
      ''')

print('I think this is the best tow, I checked with Simon 23-11-2018')

