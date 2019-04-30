# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 13:42:25 2018

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


# =============================================================================
# Input data 
# =============================================================================
df = pd.read_excel('04_02_donnees.xls', sheet_name='Sheet1')
t_exp_column = df['Temps']
E11_exp_column = df['Epsilon_11']
E22_exp_column = df['Epsilon_22']

E11_exp=[]; E22_exp=[]; t_exp=[]

for i in range (0,len(t_exp_column)): #write data as list    
    E11_exp.append(E11_exp_column[i])
    E22_exp.append(E22_exp_column[i])
    t_exp.append(t_exp_column[i])

#
## =============================================================================
## Drawing Graph
## =============================================================================
#--------------------------- E11 ----------------------------------------------
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, aspect=0.7*10000)

def minor_tick(x, pos):
    if not x % 1.0:
        return ""
    return "%.2f" % x

ax.xaxis.set_major_locator(MultipleLocator(4000.000)) #change the X-Dir dimention
ax.xaxis.set_minor_locator(AutoMinorLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(0.50)) #change the Y-Dir dimention
ax.yaxis.set_minor_locator(AutoMinorLocator(1))
ax.xaxis.set_minor_formatter(FuncFormatter(minor_tick))

ax.set_xlim(0, 20000)
ax.set_ylim(0, 2)

ax.tick_params(which='major', width=1.0)
ax.tick_params(which='major', length=5)
ax.tick_params(which='minor', width=1.0, labelsize=10)
ax.tick_params(which='minor', length=5, labelsize=10, labelcolor='0.25')

ax.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)


ax.plot(t_exp, E11_exp, 'b-', label="E11_exp [experimental]")
#ax.plot(t__exp, E22_exp, 'r-', label="E22_exp [experimental]")
ax.set_title("Creep-Recovery Test", fontsize=20, verticalalignment='bottom')
ax.set_xlabel("Time [second]")
ax.set_ylabel("Axial Strain E_11 [%]")
ax.legend()

plt.show()
#--------------------------- E22 ---------------------------------------------
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, aspect=10000)

def minor_tick(x, pos):
    if not x % 1.0:
        return ""
    return "%.2f" % x

ax.xaxis.set_major_locator(MultipleLocator(4000.000)) #change the X-Dir dimention
ax.xaxis.set_minor_locator(AutoMinorLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(0.25)) #change the Y-Dir dimention
ax.yaxis.set_minor_locator(AutoMinorLocator(1))
ax.xaxis.set_minor_formatter(FuncFormatter(minor_tick))

ax.set_xlim(0, 20000)
ax.set_ylim(0, -1)

ax.tick_params(which='major', width=1.0)
ax.tick_params(which='major', length=5)
ax.tick_params(which='minor', width=1.0, labelsize=10)
ax.tick_params(which='minor', length=5, labelsize=10, labelcolor='0.25')

ax.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)


ax.plot(t_exp, E22_exp, 'r-', label="E22_exp [experimental]")

ax.set_title("Creep-Recovery Test", fontsize=20, verticalalignment='bottom')
ax.set_xlabel("Time [second]")
ax.set_ylabel("Transvere Strain E_22 [%]")
ax.legend()

plt.show()
      

# =============================================================================
# Calculate epsilon dagger and double_dagger (experiment)
# =============================================================================
        #(E_dagger = E11_exp - E22_exp)
        #(E_double_dagger = E11_exp + 2E22_exp)
        
def epsilon_dagger_doubledagger(E11_exp,E22_exp):
    '''input your experimental data for axial E11 and transverse E22 strains.
    This function gives you the E_dagger = E11_exp - E22_exp and E_double_dagger = E11_exp + 2E22_exp'''
    E_dagger_exp=[]; E_doubledagger_exp=[]
    
    for i in range(0,len(E11_exp)):        
        E_dagger_exp.append(E11_exp[i] - E22_exp[i])
        E_doubledagger_exp.append(E11_exp[i] + 2.*E22_exp[i])
    
    return E_dagger_exp, E_doubledagger_exp


(E_dagger_exp, E_doubledagger_exp)=epsilon_dagger_doubledagger(E11_exp,E22_exp)


# =============================================================================
# Theoric Epsilon (E_dagger and E_dagerdager BASED ON THE MODEL)
# =============================================================================
def E_model(x,landa,t_scalar,sigma):
    temp_sum=0
    for i in range(0,len(landa)):
        temp_sum= (x[i+1]*x[i+1])*(1. - exp(-landa[i]*t_scalar) ) + temp_sum
    E_model=( (x[0]*x[0]) + temp_sum )*sigma
    return E_model
    
# =============================================================================
# Calculate Residual (error between theory and experiment)
# =============================================================================
def res_l(x,landa,t,sigma,E_dagger_or_doubledagger):
    res=[]
    for i in range(0,len(t)):
        res.append( E_dagger_or_doubledagger[i] - E_model(x,landa,t[i],sigma) )
    return res




# =============================================================================
# =============================================================================
# # SOLUTION
# =============================================================================
# =============================================================================
# =============================================================================
    

# =============================================================================
# Calculate Landa
# =============================================================================

def landa_calculator(q,n):
    'n=number of items, q=number of pices that you want to devide the decade'
    landa=[]
    j=0    
    for i in range(0,n):
        x=1./10.**(j)
        landa.append(x)
        j=j+1/q
        
    return landa  


landa=landa_calculator(2.,11)
print('-----------landa------------') 
print('landa=',landa)

# =============================================================================
# Optimization: initial vector for alpha nad betha
# =============================================================================
def initial_vector_optimization(alpha_cont=1.0,betha_cont=1.0 ,n=len(landa)):
    alpha_inital=[]; betha_inital=[]
    for i in range(0,n+1):
        alpha_inital.append(alpha_cont)
        betha_inital.append(betha_cont)
    return alpha_inital, betha_inital
(alpha_inital, betha_inital)=(initial_vector_optimization(alpha_cont=1.,betha_cont=1.,n=len(landa)))


# =============================================================================
# time and stress
# =============================================================================
t=t_exp
sigma=20.

# =============================================================================
# acceptable thermodynamiquement
# =============================================================================
def list_power2(x):
    y=[]    
    for i in range(0,len(x)):
        y.append(x[i]*x[i])
    return y 

# =============================================================================
# Step 4-3: Optimization: finding alpha_i
# =============================================================================
E_dagger_or_doubledagger=E_doubledagger_exp
print('-----------alpha------------') 

optimization= least_squares(res_l,alpha_inital, args = (landa, t, sigma, E_dagger_or_doubledagger))
x=optimization.x

alpha=list_power2(x)
print('alpha_l=',alpha)


# =============================================================================
# Step 4-3: Optimization: finding betha_i
# =============================================================================
E_dagger_or_doubledagger=E_dagger_exp

optimization= least_squares(res_l,alpha_inital, args = (landa, t, sigma, E_dagger_or_doubledagger))
y=optimization.x

betha=list_power2(y)
print('-----------betha------------') 
print('betha=',betha)




# =============================================================================
# =============================================================================
# # AFTER OPTIMIZATION
# =============================================================================
# =============================================================================
# =============================================================================

# =============================================================================
# Epsilon_Theoric (dagger & dagger_dagger)
# =============================================================================

def epsilon_after_optimization(alpha_or_betha,landa,t,sigma):
    E_model=[]
    for j in range(0,len(t_exp)):
        temp_sum=0
        for i in range(0,len(landa)):
            temp_sum=temp_sum + alpha_or_betha[i+1]*(1. - exp(-landa[i]*t[j]) )  
            
        E_model.append((alpha_or_betha[0] + temp_sum)*sigma)         
    return E_model

E_dagger_model_after_optimiation=epsilon_after_optimization(betha,landa,t,sigma)
E_doubledagger_model_after_optimiation=epsilon_after_optimization(alpha,landa,t,sigma)


# =============================================================================
# Epsilon_11 and Epsilon_22
# =============================================================================

def epsilon_1_and_2_and_v_after_optimization(E_dagger_model_after_optimiation,E_doubledagger_model_after_optimiation):
    size_E = np.shape(E_dagger_model_after_optimiation)
    E_1_model_after_optimiation=np.zeros(size_E)
    E_2_model_after_optimiation=np.zeros(size_E)
            
    for i in range(0,len(E_dagger_model_after_optimiation)):
        E_1_model_after_optimiation[i]=(1/3.)*(2*E_dagger_model_after_optimiation[i]+E_doubledagger_model_after_optimiation[i])
        E_2_model_after_optimiation[i]=(1/3.)*(E_doubledagger_model_after_optimiation[i]-E_dagger_model_after_optimiation[i])
                                
    return E_1_model_after_optimiation, E_2_model_after_optimiation

(E_1_model_after_optimiation, E_2_model_after_optimiation)=epsilon_1_and_2_and_v_after_optimization(E_dagger_model_after_optimiation,E_doubledagger_model_after_optimiation)


# =============================================================================
# Poisson ration 
# =============================================================================
def poisson_ratio_calculator(E_2,E_1):
    'pay attention input(E22,E11)'
    size_E = np.shape(E_1)
    v=np.zeros(size_E)
    
    for i in range(0,len(E_1)):
        v[i]=-E_2[i]/E_1[i]
    return v


v_model_after_optimization=poisson_ratio_calculator(E_2_model_after_optimiation,E_1_model_after_optimiation)
v_exp=poisson_ratio_calculator(E22_exp,E11_exp)






# =============================================================================
# Calculation S (With our alpha and betha)
# =============================================================================
def Sv_creep_test(alpha,betha):
    Sv=[]
    (I,J,K)=I_J_K_4orten_isotropes()
    J_V=tensor4_to_Voigt_tensor2(J)
    K_V=tensor4_to_Voigt_tensor2(K)
    for i in range(0,len(alpha)):       
        Si= np.multiply(alpha[i],J_V) + np.multiply(betha[i],K_V)
        Sv.append(Si)
    return Sv    

Sv=Sv_creep_test(alpha,betha)
print(Sv)
print('Sv',len(Sv))





# =============================================================================
# Graphs of results
# =============================================================================
#--------------------------- E_exp and E_model ---------------------------------------------
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, aspect=0.7*10000)

def minor_tick(x, pos):
    if not x % 1.0:
        return ""
    return "%.2f" % x

ax.xaxis.set_major_locator(MultipleLocator(4000.000)) #change the X-Dir dimention
ax.xaxis.set_minor_locator(AutoMinorLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(0.50)) #change the Y-Dir dimention
ax.yaxis.set_minor_locator(AutoMinorLocator(1))
ax.xaxis.set_minor_formatter(FuncFormatter(minor_tick))

ax.set_xlim(0, 20000)
ax.set_ylim(0, 2.5)

ax.tick_params(which='major', width=1.0)
ax.tick_params(which='major', length=5)
ax.tick_params(which='minor', width=1.0, labelsize=10)
ax.tick_params(which='minor', length=5, labelsize=10, labelcolor='0.25')

ax.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)

ax.plot(t_exp, E_dagger_exp, 'bo', label="E_dagger_exp")
ax.plot(t_exp, E_doubledagger_exp, 'ro', label="E_doubledagger_exp")

ax.plot(t_exp, E_dagger_model_after_optimiation, 'b--', label="E_dagger_model_after_optimiation (with betha)")
ax.plot(t_exp, E_doubledagger_model_after_optimiation, 'r--', label="E_doubledagger_model_after_optimiation (with alpha)")

ax.set_xlabel("Time [second]")
ax.set_ylabel("Make a comparison")
ax.legend()

plt.show()


#---------------------------E_1_exp and E__model ---------------------------------------------

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, aspect=0.7*10000)

def minor_tick(x, pos):
    if not x % 1.0:
        return ""
    return "%.2f" % x

ax.xaxis.set_major_locator(MultipleLocator(4000.000)) #change the X-Dir dimention
ax.xaxis.set_minor_locator(AutoMinorLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(0.50)) #change the Y-Dir dimention
ax.yaxis.set_minor_locator(AutoMinorLocator(1))
ax.xaxis.set_minor_formatter(FuncFormatter(minor_tick))

ax.set_xlim(0, 20000)
ax.set_ylim(-0.5, 2.)

ax.tick_params(which='major', width=1.0)
ax.tick_params(which='major', length=5)
ax.tick_params(which='minor', width=1.0, labelsize=10)
ax.tick_params(which='minor', length=5, labelsize=10, labelcolor='0.25')

ax.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)


ax.plot(t_exp, E11_exp, 'bo', label="E11_exp")
ax.plot(t_exp, E22_exp, 'ro', label="E22_exp")

ax.plot(t_exp, E_2_model_after_optimiation, 'r--', label="E_2_model_after_optimiation")
ax.plot(t_exp, E_1_model_after_optimiation, 'b--', label="E_1_model_after_optimiation")




ax.set_xlabel("Time [second]")
ax.set_ylabel("Make a comparison - Poisson Ratio")
ax.legend()

plt.show()


#--------------------------- E_1_exp and E__model ---------------------------------------------
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, aspect=0.7*10000)

def minor_tick(x, pos):
    if not x % 1.0:
        return ""
    return "%.2f" % x

ax.xaxis.set_major_locator(MultipleLocator(4000.000)) #change the X-Dir dimention
ax.xaxis.set_minor_locator(AutoMinorLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(0.50)) #change the Y-Dir dimention
ax.yaxis.set_minor_locator(AutoMinorLocator(1))
ax.xaxis.set_minor_formatter(FuncFormatter(minor_tick))

ax.set_xlim(0, 20000)
ax.set_ylim(0., 1.)

ax.tick_params(which='major', width=1.0)
ax.tick_params(which='major', length=5)
ax.tick_params(which='minor', width=1.0, labelsize=10)
ax.tick_params(which='minor', length=5, labelsize=10, labelcolor='0.25')

ax.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)


ax.plot(t_exp, v_exp, 'bo', label="v_exp=-E22/E11 (experiment)")
ax.plot(t_exp, v_model_after_optimization, 'r--', label="v_model=-E_2model/E1model after optimiation")

ax.set_xlabel("Time [second]")
ax.set_ylabel("Make a comparison - Poisson Ratio")
ax.legend()

plt.show()
