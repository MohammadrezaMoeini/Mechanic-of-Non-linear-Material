# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 11:49:08 2018

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



from Tensors_Functions_S1_S2_S3_21082018 import *
from Viscoelastic_Linear_Function_S6_Euler_NC_Rel_Flu_14102018 import *
from Viscoelastic_Linear_Function_S5_01_09102018 import *
from Viscoelastic_Linear_Problem_LAB_Linearity_26102018 import *


fluage_recouvrance_2MPa='fluage-recouvrance-2MPa.xls'





def H(x):
    "heaviside function"
    if x<0.:
        y=0.
    elif x>=0.:
        y=1.
    return y
     

def sigma_t(sigma0,t1,t2,t3,t):
    "t scalar, x vector: [sigma0,t1,t2,t3]"
    y=( (t/t1)*H(t) + ((-t+t1)/(t1))*H(t-t1) + ((-t+t2)/(t3-t2))*H(t-t2) +  ((t-t3)/(t3-t2))*H(t-t3) ) * sigma0
    
    return y


(t_2MP, sigma_2MP, e11_exp_2MP, e22_exp_2MP)=import_LAB_Date_t_sigma_e11_e22(fluage_recouvrance_2MPa)
t_exp=t_2MP
sigma_exp=sigma_2MP
t1=6.
t2=905.
t3=910.
sigma0=max(sigma_2MP)

x=[]
for i in range(0,len(t_exp)):
    y=sigma_t(sigma0,t1,t2,t3,t_exp[i])
    x.append(y)



#
#sigma_t=np.zeros(shape(t_exp))
#for i in range(0,len(t_exp)):
#    t=t_exp[i]
#    sigma_t[i]=sigma_t(sigma0,t1,t2,t3,t)
#    
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
plt.plot(t_exp, sigma_2MP, 'r--')
plt.plot(t_exp, x, 'b-')

plt.xlabel('time [seconds]')
plt.ylabel('Poisson Ratio [-]')
plt.title('"Make a comparison')
plt.legend(('sigma_2MP_exp','sigma_2MP_model'),
               shadow=True, loc=(0.5, 0.4), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([0, 1500, 0.0, 3.])
plt.show()

 


#def f(sigma0,t1,t2,t3,t_scalar):
#    "t scalar, x vector: [sigma0,t1,t2,t3]"
#    y=( (1.)*H(t_scalar-t3)) * sigma0
#    print(y)
#    return y
#   
#x=[]
#for i in range(0,len(t_exp)):
#    y=f(sigma0,t1,t2,t3,t_exp[i])
#    x.append(y)
#    
# 
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
plt.plot(t_exp, sigma_2MP, 'r--')
plt.plot(t_exp, x, 'b-')

plt.xlabel('time [seconds]')
plt.ylabel('Poisson Ratio [-]')
plt.title('"Make a comparison')
plt.legend(('sigma_2MP_exp','sigma_2MP_model'),
               shadow=True, loc=(0.5, 0.4), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([0, 1500, 0.0, 3.])
plt.show()
#    
#    
#def E_model_1(x,landa,t_scalar,sigma0,t1,t2,t3):
#    temp_sum_1=0.
#    temp_sum_2=0.
#    temp_sum_3=0.
#    temp_sum_4=0.
#    for i in range(0,len(landa)):
#        temp_sum_1=( (x[i+1]*x[i+1]) * ( t_scalar - (1./landa[i])*(1.-exp(-landa[i]*t_scalar)) ) )  + temp_sum_1
#        
#        temp_sum_2=( (x[i+1]*x[i+1]) * ( t_scalar - t1 - (1./landa[i])*(1.-exp(-landa[i]*(t_scalar-t1))) ) )  + temp_sum_2
##        
#        temp_sum_3=( (x[i+1]*x[i+1]) * ( t_scalar - t2 - (1./landa[i])*(1.-exp(-landa[i]*(t_scalar-t2))) ) ) *(-sigma0/(t3-t2))*H(t_scalar-t2) + temp_sum_3
##        
#        temp_sum_4=( (x[i+1]*x[i+1]) * ( t_scalar - t3 - (1./landa[i])*(1.-exp(-landa[i]*(t_scalar-t3))) ) )  + temp_sum_4
##        
#        
#    E_model=( ( t_scalar*(x[0]*x[0]) + temp_sum_1 )* (sigma0/t1)*H(t_scalar)  + 
#              ( (t_scalar-t1)*(x[0]*x[0]) + temp_sum_2 )*  (-sigma0/t1)*H(t_scalar-t1) +
#              ( (t_scalar-t2)*(x[0]*x[0]) + temp_sum_3 )*  (-sigma0/(t3-t2))*H(t_scalar-t2) +
#              ( (t_scalar-t3)*(x[0]*x[0]) + temp_sum_4 )*  (sigma0/(t3-t2))*H(t_scalar-t3) )
#    return E_model 
#
#x=[1,2,3,4,5,6,7,8,9,10,11,12]
#t_scalar=2.
#e=E_model(x,landa,t_scalar,sigma0,t1,t2,t3)
#
#  
#
#
#
#
#
#
#
#
#def E_model_2(x,landa,t_scalar,sigma0,t1,t2,t3):
#    temp_sum_1=0.
#    temp_sum_2=0.
#    temp_sum_3=0.
#    temp_sum_4=0.
#    
#    part_1=0.
#    part_2=0.
#    part_3=0.
#    part_4=0.
#    
#    for i in range(0,len(landa)):
#        if t_scalar>=0.:
#            temp_sum_1= ( (x[i+1]*x[i+1]) * ( t_scalar - (1./landa[i])*(1.-exp(-landa[i]*t_scalar)) ) )  + temp_sum_1
#            part_1= ( t_scalar*(x[0]*x[0]) + temp_sum_1 )* (sigma0/t1)
#        
#        if t_scalar>=t1:
#            temp_sum_2= ( (x[i+1]*x[i+1]) * ( t_scalar - t1 - (1./landa[i])*(1.-exp(-landa[i]*(t_scalar-t1))) ) )  + temp_sum_2
#            part_2= ( (t_scalar-t1)*(x[0]*x[0]) + temp_sum_2 )*  (-sigma0/t1)
#        else:
#            part_2=0.
#        
#        if t_scalar>=t2:
#            temp_sum_3= ( (x[i+1]*x[i+1]) * ( t_scalar - t2 - (1./landa[i])*(1.-exp(-landa[i]*(t_scalar-t2))) ) ) *(-sigma0/(t3-t2))*H(t_scalar-t2) + temp_sum_3
#            part_3= ( (t_scalar-t2)*(x[0]*x[0]) + temp_sum_3 )*  (-sigma0/(t3-t2))
#        else:
#            part_3=0.
#        
#        if t_scalar>=t3:
#            temp_sum_4=( (x[i+1]*x[i+1]) * ( t_scalar - t3 - (1./landa[i])*(1.-exp(-landa[i]*(t_scalar-t3))) ) )  + temp_sum_4
#            part_4=  ( (t_scalar-t3)*(x[0]*x[0]) + temp_sum_4 )*  (sigma0/(t3-t2))
#        else:
#            part_4=0.
#            
#    E_model=part_1 + part_2 + part_3 + part_4
#    return E_model 
#
#
#
#def E_model_3(x,landa,t_scalar,sigma0,t1,t2,t3):
#    temp_sum_1=0.
#    temp_sum_2=0.
#    temp_sum_3=0.
#    temp_sum_4=0.
#    
#    part_1=0.
#    part_2=0.
#    part_3=0.
#    part_4=0.
#    
#    for i in range(0,len(landa)):
#        temp_sum_1= ( (x[i+1]*x[i+1]) * ( t_scalar - (1./landa[i])*(1.-exp(-landa[i]*t_scalar)) ) )  + temp_sum_1
#        part_1= ( t_scalar*(x[0]*x[0]) + temp_sum_1 )* (sigma0/t1)
#        
#        E_model=part_1
#
#        
#        if t_scalar>=t1:
#            temp_sum_2= ( (x[i+1]*x[i+1]) * ( t_scalar - t1 - (1./landa[i])*(1.-exp(-landa[i]*(t_scalar-t1))) ) )  + temp_sum_2
#            part_2= ( (t_scalar-t1)*(x[0]*x[0]) + temp_sum_2 )*  (-sigma0/t1)
#            
#            E_model=part_1+part_2
#
#        
#        if t_scalar>=t2:
#            temp_sum_3= ( (x[i+1]*x[i+1]) * ( t_scalar - t2 - (1./landa[i])*(1.-exp(-landa[i]*(t_scalar-t2))) ) ) *(-sigma0/(t3-t2))*H(t_scalar-t2) + temp_sum_3
#            part_3= ( (t_scalar-t2)*(x[0]*x[0]) + temp_sum_3 )*  (-sigma0/(t3-t2))
#            
#            E_model=part_1+part_2 + part_3
#
#        
#        if t_scalar>=t3:
#            temp_sum_4=( (x[i+1]*x[i+1]) * ( t_scalar - t3 - (1./landa[i])*(1.-exp(-landa[i]*(t_scalar-t3))) ) )  + temp_sum_4
#            part_4=  ( (t_scalar-t3)*(x[0]*x[0]) + temp_sum_4 )*  (sigma0/(t3-t2))
#            
#            E_model=part_1+part_2 + part_3 + part_4
#
#            
#    return E_model 
#




def E_model_1(x,landa,t_scalar,sigma0,t1,t2,t3):
    
    def f_1(t):
        temp_sum_1=0.
        if t>=0.:
            for i in range(0,len(landa)):
                temp_sum_1= ( (x[i+1]*x[i+1]) * ( t_scalar - (1./landa[i])*(1.-exp(-landa[i]*t_scalar)) ) )  + temp_sum_1
            part_1= ( t_scalar*(x[0]*x[0]) + temp_sum_1 )* (sigma0/t1)
        else:
            part_1=0.
        return part_1
    
    def f_2(t):
        temp_sum_2=0.
        if t_scalar>=t1:
            for i in range(0,len(landa)):
                temp_sum_2= ( (x[i+1]*x[i+1]) * ( t_scalar - t1 - (1./landa[i])*(1.-exp(-landa[i]*(t_scalar-t1))) ) )  + temp_sum_2
            part_2= ( (t_scalar-t1)*(x[0]*x[0]) + temp_sum_2 )*  (-sigma0/t1)
        else:
            part_2=0.
        return part_2
            
    def f_3(t):
        temp_sum_3=0.
        if t_scalar>=t2:
            for i in range(0,len(landa)):
                temp_sum_3= ( (x[i+1]*x[i+1]) * ( t_scalar - t2 - (1./landa[i])*(1.-exp(-landa[i]*(t_scalar-t2))) ) ) *(-sigma0/(t3-t2))*H(t_scalar-t2) + temp_sum_3
            part_3= ( (t_scalar-t2)*(x[0]*x[0]) + temp_sum_3 )*  (-sigma0/(t3-t2))
        else:
            part_3=0.
        return part_3            
            
    def f_4(t):
        temp_sum_4=0.        
        if t_scalar>=t3:
            for i in range(0,len(landa)):
                temp_sum_4=( (x[i+1]*x[i+1]) * ( t_scalar - t3 - (1./landa[i])*(1.-exp(-landa[i]*(t_scalar-t3))) ) )  + temp_sum_4
            part_4=  ( (t_scalar-t3)*(x[0]*x[0]) + temp_sum_4 )*  (sigma0/(t3-t2))
        else:
            part_4=0.
        return part_4  


    E_model=f_1(t_scalar)+f_2(t_scalar)+f_3(t_scalar)+f_4(t_scalar)
            
    return E_model 



#landa=[2,3,4,5,6,7,8,9,10,11,12]
#x=[1,2,3,4,5,6,7,8,9,10,11,12]
#t_scalar=2.
#e=E_model(x,landa,t_scalar,sigma0,t1,t2,t3)
#
#
#
#


def E_model(x,landa,t_scalar,sigma0,t1,t2,t3):
    temp_sum_1=0.
    temp_sum_2=0.
    temp_sum_3=0.
    temp_sum_4=0.
    for i in range(0,len(landa)):
        temp_sum_1=( (x[i+1]*x[i+1]) * ( t_scalar - (1./landa[i])*     (1.-  exp(-landa[i]*t_scalar*np.heaviside(t_scalar,0.5))      )))*np.heaviside(t_scalar,0.5)           + temp_sum_1
        
        temp_sum_2=( (x[i+1]*x[i+1]) * ( t_scalar - t1 - (1./landa[i])*(1.-exp(-landa[i]*(t_scalar-t1)*np.heaviside(t_scalar-t1,0.5)      ))))*np.heaviside(t_scalar-t1,0.5)  + temp_sum_2
        
        temp_sum_3=( (x[i+1]*x[i+1]) * ( t_scalar - t2 - (1./landa[i])*(1.-exp(-landa[i]*(t_scalar-t2)*np.heaviside(t_scalar-t2,0.5)      ))))*np.heaviside(t_scalar-t2,0.5)  + temp_sum_3
        
        temp_sum_4=( (x[i+1]*x[i+1]) * ( t_scalar - t3 - (1./landa[i])*(1.-exp(-landa[i]*(t_scalar-t3)*np.heaviside(t_scalar-t3,0.5)      ))))*np.heaviside(t_scalar-t3,0.5)  + temp_sum_4
        
        
    E_model=( ( t_scalar*(x[0]*x[0])*np.heaviside(t_scalar,0.5)         + temp_sum_1 )*     (sigma0/t1)  + 
              ( (t_scalar-t1)*(x[0]*x[0])*np.heaviside(t_scalar-t1,0.5) + temp_sum_2 )*     (-sigma0/t1) +
              ( (t_scalar-t2)*(x[0]*x[0])*np.heaviside(t_scalar-t2,0.5) + temp_sum_3 )*     (-sigma0/(t3-t2)) +
              ( (t_scalar-t3)*(x[0]*x[0])*np.heaviside(t_scalar-t3,0.5) + temp_sum_4 )*     (sigma0/(t3-t2)) )
    return E_model 




#x=[1,2,3,4,5,6,7,8,9,10,11,12]
t_scalar=2000.
e=E_model(x,landa,t_scalar,sigma0,t1,t2,t3)

    

#r=[] 
#for i in range(0,len(t_exp[i])):
#    t_scalar=t_exp[i]    
#    e=np.heaviside(t_scalar,0.5)
#    r.append(np.heaviside(t_scalar,0.5))
#    



# =============================================================================
# 
# =============================================================================

# =============================================================================
# 
# =============================================================================

