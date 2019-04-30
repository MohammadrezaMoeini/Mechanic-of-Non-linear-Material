"""
Created on Tue Oct  27 2018

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
#from Viscoelastic_Linear_Function_S6_Euler_NC_Rel_Flu_14102018 import *
from Viscoelastic_Linear_Problem_LAB_Euler_NC_Rel_Flu_31102018 import * #(the difference is just for importing)
from Viscoelastic_Linear_Function_S5_01_09102018 import *
from Viscoelastic_Linear_Function_LAB_interconversion_25102018 import *
from Viscoelastic_Linear_Function_LAB_1D_interconversion_09112018 import *




print('*************************************************************************')
print('Checking Domin of Linearity')
print('*************************************************************************')


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

plt.plot(t_2MP, sigma_2MP, 'r--.')
plt.plot(t_5MP, sigma_5MP, 'r--+')
plt.plot(t_10MP, sigma_10MP, 'r--v')
plt.plot(t_13MP, sigma_13MP, 'r--^')
plt.plot(t_16MP, sigma_16MP, 'r-->')
plt.plot(t_20MP, sigma_20MP, 'r--<')
plt.plot(t_22MP, sigma_22MP, 'r--*')
plt.plot(t_25MP, sigma_25MP, 'r--8')
plt.plot(t_30MP, sigma_30MP, 'r--s')
plt.plot(t_35MP, sigma_35MP, 'r--p')


plt.xlabel('time [seconds]')
plt.ylabel('Stress [MPa]')
plt.title('Experimental axial stress for Creep test')
plt.legend(('sigma_2MP','sigma_5MP','sigma_10MP','sigma_13MP','sigma_16MP','sigma_20MP',
            'sigma_22MP','sigma_25MP','sigma_30MP','sigma_35MP'),
               shadow=True, loc=(0.4, 0.3), handlelength=1.5, fontsize=14)

plt.grid(True)
plt.axis([0, 4000, 0.0, 50])
plt.show()



figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 

plt.plot(t_2MP, e11_exp_2MP, 'b--.')
plt.plot(t_5MP, e11_exp_5MP, 'b--+')
plt.plot(t_10MP, e11_exp_10MP, 'b--v')
plt.plot(t_13MP, e11_exp_13MP, 'b--^')
plt.plot(t_16MP, e11_exp_16MP, 'b-->')
plt.plot(t_20MP, e11_exp_20MP, 'b--<')
plt.plot(t_22MP, e11_exp_22MP, 'b--*')
plt.plot(t_25MP, e11_exp_25MP, 'b--8')
plt.plot(t_30MP, e11_exp_30MP, 'b--s')
plt.plot(t_35MP, e11_exp_35MP, 'b--p')


plt.xlabel('time [seconds]')
plt.ylabel('Strain [-]')
plt.title('Experimental axial strains for Creep test')
plt.legend(('e11_exp_2MP','e11_exp_5MP','e11_exp_10MP','e11_exp_13MP','e11_exp_16MP','e11_exp_20MP',
            'e11_exp_22MP','e11_exp_25MP','e11_exp_30MP','e11_exp_35MP'),
               shadow=True, loc=(0.4, 0.3), handlelength=1.5, fontsize=14)
    
plt.grid(True)
plt.axis([0, 4000, 0.0, 0.02])
plt.show()


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


plt.plot(t_2MP, gamma_2,'-.')
plt.plot(t_2MP, gamma_5, '-+')
plt.plot(t_2MP, gamma_10, '-v')
plt.plot(t_2MP, gamma_13, '--^')
plt.plot(t_2MP, gamma_16, '-->')
plt.plot(t_2MP, gamma_20, '--<')
plt.plot(t_2MP, gamma_22, '--*')
plt.plot(t_2MP, gamma_25, '--8')
plt.plot(t_2MP, gamma_30, '--s')
plt.plot(t_2MP, gamma_35, '--p')


num_plots = 20
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
x = np.arange(10)

plt.xlabel('time [seconds]')
plt.ylabel('gamma [-]')
plt.title('Le domaine de linéarité')
plt.legend(('gamma_2','gamma_5','gamma_10','gamma_13','gamma_16'
            ,'gamma_20','gamma_22','gamma_25','gamma_30','gamma_35'), ncol=5, loc='upper center',
               shadow=True, handlelength=1.5, fontsize=14)
plt.grid(True)
plt.axis([0, 4000, 0.0, 6.])
plt.show()






# =============================================================================
print('second graph')
# =============================================================================

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


figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')


plt.plot(t_2MP[0:62], gamma_2[0:62],'-.')
plt.plot(t_2MP[0:62], gamma_5[0:62], '-+')
plt.plot(t_2MP[0:62], gamma_10[0:62], '-v')
plt.plot(t_2MP[0:62], gamma_13[0:62], '--^')
plt.plot(t_2MP[0:62], gamma_16[0:62], '-->')
plt.plot(t_2MP[0:62], gamma_20[0:62], '--<')
plt.plot(t_2MP[0:62], gamma_22[0:62], '--*')
plt.plot(t_2MP[0:62], gamma_25[0:62], '--8')
plt.plot(t_2MP[0:62], gamma_30[0:62], '--s')
plt.plot(t_2MP[0:62], gamma_35[0:62], '--p')


num_plots = 20
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
x = np.arange(10)

plt.xlabel('time [seconds]')
plt.ylabel('gamma [-]')
plt.title('Le domaine de linéarité')
plt.legend(('gamma_2','gamma_5','gamma_10','gamma_13','gamma_16'
            ,'gamma_20','gamma_22','gamma_25','gamma_30','gamma_35'), ncol=5, loc='upper center',
               shadow=True, handlelength=1.5, fontsize=11)
plt.grid(True)
plt.axis([0, 900, 0.8, 1.5])
plt.show()



print('*************************************************************************')
print('Linear Viscoelastic')
print('*************************************************************************')

# =============================================================================
# Import Data as a list
# =============================================================================
def import_data_as_list(excel_file):
    df = pd.read_excel(excel_file, sheet_name='Sheet1')    
    ct,cs,ce11,ce22=df
            
    t_exp_column = df[ct]
    sigma_exp_column = df[cs]
    E11_exp_column = df[ce11]
    E22_exp_column = df[ce22]
    
    t_exp=[]; sigma_exp=[]; E11_exp=[]; E22_exp=[]
    
    for i in range (0,len(t_exp_column)): 
        t_exp.append(t_exp_column[i])
        sigma_exp.append(sigma_exp_column[i])
        E11_exp.append(E11_exp_column[i])
        E22_exp.append(E22_exp_column[i])
    
    return t_exp, sigma_exp, E11_exp, E22_exp 
 

fluage_recouvrance_2MPa='fluage-recouvrance-2MPa.xls'
fluage_recouvrance_5MPa='fluage-recouvrance-5MPa.xls'
fluage_recouvrance_10MPa='fluage-recouvrance-10MPa.xls'


(t_exp_2, sigma_exp_2, E11_exp_2, E22_exp_2)=import_data_as_list(fluage_recouvrance_2MPa)
(t_exp_5, sigma_exp_5, E11_exp_5, E22_exp_5)=import_data_as_list(fluage_recouvrance_5MPa)
(t_exp_10, sigma_exp_10, E11_exp_10, E22_exp_10)=import_data_as_list(fluage_recouvrance_10MPa)

  
# =============================================================================
# Calculate epsilon dagger and double_dagger (experiment)
# =============================================================================
#(E_dagger = E11_exp - E22_exp)
#(E_double_dagger = E11_exp + 2*E22_exp)
        
def epsilon_dagger_doubledagger(E11_exp,E22_exp):
    '''input your experimental data for axial E11 and transverse E22 strains.
    This function gives you the E_dagger = E11_exp - E22_exp and E_double_dagger = E11_exp + 2E22_exp'''
    E_dagger_exp=[]; E_doubledagger_exp=[]
    
    for i in range(0,len(E11_exp)):        
        E_dagger_exp.append(E11_exp[i] - E22_exp[i])
        E_doubledagger_exp.append(E11_exp[i] + 2.*E22_exp[i])
    
    return E_dagger_exp, E_doubledagger_exp



(E_dagger_exp_2, E_doubledagger_exp_2)=epsilon_dagger_doubledagger(E11_exp_2,E22_exp_2)
(E_dagger_exp_5, E_doubledagger_exp_5)=epsilon_dagger_doubledagger(E11_exp_5,E22_exp_5)
(E_dagger_exp_10, E_doubledagger_exp_10)=epsilon_dagger_doubledagger(E11_exp_10,E22_exp_10)

# =============================================================================
# CASE 1-2 (output list)
# =============================================================================
def E_model_list_outpout(x,landa,t_vector,sigma0,t1,t2,t3):
    
    
    def f_1(t_scalar):
        temp_sum_1=0.
        if t_scalar>=0.:
            for i in range(0,len(landa)):
                temp_sum_1= ( (x[i+1]*x[i+1]) * ( t_scalar - (1./landa[i])*(1.-exp(-landa[i]*t_scalar)) ) )  + temp_sum_1
            part_1= ( t_scalar*(x[0]*x[0]) + temp_sum_1 )   * (sigma0/t1)
        else:
            part_1=0.
        return part_1
    
    def f_2(t_scalar):
        temp_sum_2=0.
        if t_scalar>=t1:
            for i in range(0,len(landa)):
                temp_sum_2= ( (x[i+1]*x[i+1]) * ( t_scalar - t1 - (1./landa[i])*(1.-exp(-landa[i]*(t_scalar-t1))) ) )  + temp_sum_2
            part_2= ( (t_scalar-t1)*(x[0]*x[0]) + temp_sum_2 )*  (-sigma0/t1)
        else:
            part_2=0.
        return part_2
            
    def f_3(t_scalar):
        temp_sum_3=0.
        if t_scalar>=t2:
            for i in range(0,len(landa)):
                temp_sum_3= ( (x[i+1]*x[i+1]) * ( t_scalar - t2 - (1./landa[i])*(1.-exp(-landa[i]*(t_scalar-t2))) ) )  + temp_sum_3
            part_3= ( (t_scalar-t2)*(x[0]*x[0]) + temp_sum_3 )*  (-sigma0/(t3-t2))
        else:
            part_3=0.
        return part_3            
            
    def f_4(t_scalar):
        temp_sum_4=0.        
        if t_scalar>=t3:
            for i in range(0,len(landa)):
                temp_sum_4=( (x[i+1]*x[i+1]) * ( t_scalar - t3 - (1./landa[i])*(1.-exp(-landa[i]*(t_scalar-t3))) ) )  + temp_sum_4
            part_4=  ( (t_scalar-t3)*(x[0]*x[0]) + temp_sum_4 )*  (sigma0/(t3-t2))
        else:
            part_4=0.
        return part_4
    
    E_model_list=[]
    for j in range(0,len(t_vector)):
        t_scalar=t_vector[j]
        E_model_output=f_1(t_scalar)+f_2(t_scalar)+f_3(t_scalar)+f_4(t_scalar)
        E_model_list.append(E_model_output)
                   
    return E_model_list 
print('CASE 1: my_if_cluss')


# =============================================================================
# CASE 2 - Heviside function (np.heaviside)
# =============================================================================

#def E_model(x,landa,t_scalar,sigma0,t1,t2,t3):
#    temp_sum_1=0.
#    temp_sum_2=0.
#    temp_sum_3=0.
#    temp_sum_4=0.
#    for i in range(0,len(landa)):
#        temp_sum_1=( (x[i+1]*x[i+1]) * ( t_scalar - (1./landa[i])*     (1.-  exp(-landa[i]*t_scalar*np.heaviside(t_scalar,0.5)) )           ))*np.heaviside(t_scalar,0.5)      + temp_sum_1
#        
#        temp_sum_2=( (x[i+1]*x[i+1]) * ( t_scalar - t1 - (1./landa[i])*(1.-exp(-landa[i]*(t_scalar-t1)*np.heaviside(t_scalar-t1,0.5) )      )))*np.heaviside(t_scalar-t1,0.5)  + temp_sum_2
#        
#        temp_sum_3=( (x[i+1]*x[i+1]) * ( t_scalar - t2 - (1./landa[i])*(1.-exp(-landa[i]*(t_scalar-t2)*np.heaviside(t_scalar-t2,0.5) )      )))*np.heaviside(t_scalar-t2,0.5)  + temp_sum_3
#        
#        temp_sum_4=( (x[i+1]*x[i+1]) * ( t_scalar - t3 - (1./landa[i])*(1.-exp(-landa[i]*(t_scalar-t3)*np.heaviside(t_scalar-t3,0.5) )      )))*np.heaviside(t_scalar-t3,0.5)  + temp_sum_4
#        
#        
#    E_model_output=( ( t_scalar*(x[0]*x[0])*np.heaviside(t_scalar,0.5)  + temp_sum_1 )*     (sigma0/t1)  + 
#              ( (t_scalar-t1)*(x[0]*x[0])*np.heaviside(t_scalar-t1,0.5) + temp_sum_2 )*     (-sigma0/t1) +
#              ( (t_scalar-t2)*(x[0]*x[0])*np.heaviside(t_scalar-t2,0.5) + temp_sum_3 )*     (-sigma0/(t3-t2)) +
#              ( (t_scalar-t3)*(x[0]*x[0])*np.heaviside(t_scalar-t3,0.5) + temp_sum_4 )*     (sigma0/(t3-t2)) )
#    return E_model_output 
#
#print('heviside_function')

#Note for these two cases (02-11-2018)
#   I wrote two different cases.
#   first: it's by if-clauses. which is output is list. 
#   Second: by np.heaviside function which the output is scalar (so don't use this here. the res get only list. you should change that)
#   Reason: my friend told me np.heaviside is faster. (we should also use heviside in fower of exponential in order to control "infinity x 0")
#   However, I saw that my if-clauses is more faster.

    

# =============================================================================
# residual
# =============================================================================

def res_l(x,landa,t,sigma0_2,E_dagger_or_doubledagger_2,sigma0_5,E_dagger_or_doubledagger_5,sigma0_10,E_dagger_or_doubledagger_10,t1,t2,t3):
    res=[]
    
    E_model_list_2=E_model_list_outpout(x,landa,t,sigma0_2,t1,t2,t3)
    for i in range(0,len(t)):
        res.append(E_dagger_or_doubledagger_2[i] - E_model_list_2[i] )
    
    E_model_list_5=E_model_list_outpout(x,landa,t,sigma0_5,t1,t2,t3)
    for i in range(0,len(t)):
        res.append(E_dagger_or_doubledagger_5[i] - E_model_list_5[i] )
    
    E_model_list_10=E_model_list_outpout(x,landa,t,sigma0_10,t1,t2,t3)
    for i in range(0,len(t)):
        res.append(E_dagger_or_doubledagger_10[i] - E_model_list_10[i] )
        
    return res

# =============================================================================
# x^2 ----> alpha or betha    
# =============================================================================
def list_power2(x):
    y=[]    
    for i in range(0,len(x)):
        y.append(x[i]*x[i])
    return y
# =============================================================================
#  landa     
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
landa=landa_calculator(2.,9)
print('-----------landa------------') 
print('landa=',landa) 

# =============================================================================
# initial_vector_optimization
# =============================================================================
def initial_vector_optimization(alpha_cont=0.001,betha_cont=0.001 ,n=len(landa)):
    alpha_inital=[]; betha_inital=[]
    for i in range(0,n+1):
        alpha_inital.append(alpha_cont)
        betha_inital.append(betha_cont)
    return alpha_inital, betha_inital
(alpha_inital, betha_inital)=(initial_vector_optimization(alpha_cont=0.001, betha_cont=0.001 ,n=len(landa)))
# =============================================================================
# Sigma
# =============================================================================
sigma0_2=max(sigma_exp_2)
sigma0_5=max(sigma_exp_5)
sigma0_10=max(sigma_exp_10)
# =============================================================================
# time scalar parameters (t1,t2,t3)
# =============================================================================
t1=6.
t2=905.
t3=910.
# =============================================================================
# time-vector   
# =============================================================================
t=t_exp_2 # = t_exp_5 = t_exp_10
# =============================================================================
# Optimization
# =============================================================================
job_date_time='optimization'+time.strftime('[%X - %x]')
print(job_date_time, 'Start time')
#----------------------------------------------
#----------------------------------------------
# alpha
#----------------------------------------------
E_dagger_or_doubledagger_2=E_doubledagger_exp_2
E_dagger_or_doubledagger_5=E_doubledagger_exp_5
E_dagger_or_doubledagger_10=E_doubledagger_exp_10


optimization_alpha= least_squares(res_l,alpha_inital, args = (landa,t,sigma0_2,E_dagger_or_doubledagger_2,sigma0_5,E_dagger_or_doubledagger_5,sigma0_10,E_dagger_or_doubledagger_10,t1,t2,t3))

x_alpha=optimization_alpha.x
print('-----------alpha------------') 
alpha=list_power2(x_alpha)
print('alpha_l=',alpha)
    
#----------------------------------------------
# betha
#----------------------------------------------
E_dagger_or_doubledagger_2=E_dagger_exp_2
E_dagger_or_doubledagger_5=E_dagger_exp_5
E_dagger_or_doubledagger_10=E_dagger_exp_10

optimization_betha= least_squares(res_l,betha_inital, args = (landa,t,sigma0_2,E_dagger_or_doubledagger_2,sigma0_5,E_dagger_or_doubledagger_5,sigma0_10,E_dagger_or_doubledagger_10,t1,t2,t3))
y_betha=optimization_betha.x
betha=list_power2(y_betha)
print('-----------betha------------') 
print('betha=',betha)

#----------------------------------------------
job_date_time='optimization'+time.strftime('[%X - %x]')
print(job_date_time, 'End time')


# =============================================================================
# =============================================================================
# # AFTER OPTIMISATION
# =============================================================================
# =============================================================================


# =============================================================================
# Epsilon_model (dagger & dagger_dagger)
# =============================================================================
E_dagger_model_after_optimiation_2=E_model_list_outpout(y_betha,landa,t_exp_2,sigma0_2,t1,t2,t3)
E_dagger_model_after_optimiation_5=E_model_list_outpout(y_betha,landa,t_exp_5,sigma0_5,t1,t2,t3)
E_dagger_model_after_optimiation_10=E_model_list_outpout(y_betha,landa,t_exp_10,sigma0_10,t1,t2,t3)
#------------------------------------
E_doubledagger_model_after_optimiation_2=E_model_list_outpout(x_alpha,landa,t_exp_2,sigma0_2,t1,t2,t3)
E_doubledagger_model_after_optimiation_5=E_model_list_outpout(x_alpha,landa,t_exp_5,sigma0_5,t1,t2,t3)
E_doubledagger_model_after_optimiation_10=E_model_list_outpout(x_alpha,landa,t_exp_10,sigma0_10,t1,t2,t3)

# =============================================================================
# E11 and E22 after optimization
# =============================================================================

def epsilon_11_22_after_optimization(E_dagger_model_after_optimiation,E_doubledagger_model_after_optimiation):
    size_E = np.shape(E_dagger_model_after_optimiation)
    E11_model_after_optimiation=np.zeros(size_E)
    E22_model_after_optimiation=np.zeros(size_E)
            
    for i in range(0,len(E_dagger_model_after_optimiation)):
        E11_model_after_optimiation[i]=(1/3.)*(2*E_dagger_model_after_optimiation[i]+E_doubledagger_model_after_optimiation[i])
        E22_model_after_optimiation[i]=(1/3.)*(E_doubledagger_model_after_optimiation[i]-E_dagger_model_after_optimiation[i])
                                
    return E11_model_after_optimiation, E22_model_after_optimiation

(E11_model_after_optimiation_2, E22_model_after_optimiation_2)=epsilon_11_22_after_optimization(E_dagger_model_after_optimiation_2,E_doubledagger_model_after_optimiation_2)
(E11_model_after_optimiation_5, E22_model_after_optimiation_5)=epsilon_11_22_after_optimization(E_dagger_model_after_optimiation_5,E_doubledagger_model_after_optimiation_5)
(E11_model_after_optimiation_10, E22_model_after_optimiation_10)=epsilon_11_22_after_optimization(E_dagger_model_after_optimiation_10,E_doubledagger_model_after_optimiation_10)


## =============================================================================
## Graphs of results
## =============================================================================

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
plt.plot(t_exp_2, E11_model_after_optimiation_2, 'b--')
plt.plot(t_exp_5, E11_model_after_optimiation_5, 'g--')
plt.plot(t_exp_10, E11_model_after_optimiation_10, 'r--')

plt.plot(t_exp_2, E11_exp_2, 'bv')
plt.plot(t_exp_5, E11_exp_5, 'go')
plt.plot(t_exp_10, E11_exp_10, 'rs')

plt.xlabel('time [seconds]')
plt.ylabel('Axial Strain[-]')
plt.title('Strain Responce E_11 (experiment & model)')
plt.legend(('E_11_model; 2MPa','E_11_model; 5MPa','E_11_model; 10MPa','E_11_exp; 2MPa','E_11_exp; 5MPa','E_11_exp; 10MPa'),
               shadow=True, loc=(0.5, 0.4), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([0, 2500, 0., 0.004])
plt.show()



figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
plt.plot(t_exp_2, E22_model_after_optimiation_2, 'b--')
plt.plot(t_exp_5, E22_model_after_optimiation_5, 'g--')
plt.plot(t_exp_10, E22_model_after_optimiation_10, 'r--')

plt.plot(t_exp_2, E22_exp_2, 'bv')
plt.plot(t_exp_5, E22_exp_5, 'go')
plt.plot(t_exp_10, E22_exp_10, 'rs')


plt.xlabel('time [seconds]')
plt.ylabel('Transverse Strain [-]')
plt.title('Strain Responce E_22 (experiment & model)')
plt.legend(('E_22_model; 2Mpa','E_22_model; 5MPa','E_22_model; 10MPa','E_22_exp; 2MPa','E_22_exp; 5MPa','E_22_exp; 10 MPa'),
               shadow=True, loc=(0.5, 0.2), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([0, 2500, -0.0015, 0.00025])
plt.show()


# =============================================================================
# =============================================================================
# # Validation with complex loading
# =============================================================================
# =============================================================================
# =============================================================================
# Importing data (complex loading 1)
# =============================================================================

def import_complex_loading_data_as_list(excel_file):
    df = pd.read_excel(excel_file, sheet_name='Sheet1')    
    ct,cs1,cs2,cs3,cs4,cs5,cs6,ce11,ce22=df
            
    t_exp_column = df[ct]
    sigma_exp_column_1 = df[cs1]
    sigma_exp_column_2 = df[cs2]
    sigma_exp_column_3 = df[cs3]
    sigma_exp_column_4 = df[cs4]
    sigma_exp_column_5 = df[cs5]
    sigma_exp_column_6 = df[cs6]
    E11_exp_column = df[ce11]
    E22_exp_column = df[ce22]
    
    t_exp=[];
    sigma_exp_1=[]; sigma_exp_2=[]; sigma_exp_3=[]
    sigma_exp_4=[]; sigma_exp_5=[]; sigma_exp_6=[]
    E11_exp=[]; E22_exp=[]
    
    for i in range (0,len(t_exp_column)): 
        t_exp.append(t_exp_column[i])
        sigma_exp_1.append(sigma_exp_column_1[i])
        sigma_exp_2.append(sigma_exp_column_2[i])
        sigma_exp_3.append(sigma_exp_column_3[i])
        sigma_exp_4.append(sigma_exp_column_4[i])
        sigma_exp_5.append(sigma_exp_column_5[i])
        sigma_exp_6.append(sigma_exp_column_6[i])
                        
        E11_exp.append(E11_exp_column[i])
        E22_exp.append(E22_exp_column[i])
    
    return t_exp, sigma_exp_1, sigma_exp_2,sigma_exp_3,sigma_exp_4,sigma_exp_5, sigma_exp_6, E11_exp, E22_exp 
 
excel_file='Histoire_Chargement_1.xls'
(t_exp, sigma_exp_1, sigma_exp_2,sigma_exp_3,sigma_exp_4,sigma_exp_5, sigma_exp_6, E11_exp, E22_exp)=import_complex_loading_data_as_list(excel_file)

#------------------------------------------------------------------------------
# =============================================================================
# Rsponce with Crank_Nicholson (complex loading (1))
# =============================================================================
#------------------------------------------------------------------------------
print('\n')
print('=============================================================================')
print('Validation with complex loading (1)')
print('=============================================================================')

# =============================================================================
# Calculation k^-1 and mu^-1
# =============================================================================
def k_mu_inv_real_creep_test(alpha,betha):
    k_inv=[]; mu_inv=[];
    for i in range(0,len(alpha)):
        ki=3.*alpha[i]
        mui=2.*betha[i]
        k_inv.append(ki)
        mu_inv.append(mui)
    return k_inv,mu_inv

(k_inv,mu_inv)=k_mu_inv_real_creep_test(alpha,betha)

# =============================================================================
# Responce Epsilon (E11,E22,E33,E44,E55,E66)
# =============================================================================
excel_file='Histoire_Chargement_1.xls'
D=6
(t, epsilon_Crank_Nicholson)=fluage_Crank_Nicholson(excel_file,k_inv,mu_inv,landa,D)

#Just for plot --------------------------------------------------------
df = pd.read_excel(excel_file, sheet_name='Sheet1')
c0,c1,c2,c3,c4,c5,c6,c7,c8=df
column_7 = df[c7]; column_8 = df[c8]
E11_exp=  np.zeros((len(column_7),1))
E22_exp=  np.zeros((len(column_8),1))
for i in range(0,len(column_7)):
    E11_exp[i]=column_7[i]
    E22_exp[i]=column_8[i]


# Curves -------------------------------------------------------------
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
plt.plot(t[0:len(t)], epsilon_Crank_Nicholson[0,:], 'b--')
plt.plot(t[0:len(t)], E11_exp[0:len(t)], 'r.')

nt=len(t)-1
plt.xlabel('time [seconds]')
plt.ylabel('Axial Strain [-]')
plt.title('Complex loading (1)')
plt.legend(('E11 Crank_Nicholson','E11 Experiment'),
           shadow=True, loc=(0.5, 0.5), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([0., 5550, 0., 0.003])
plt.show()


figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
plt.plot(t[0:len(t)], epsilon_Crank_Nicholson[1,:], 'b--')
plt.plot(t[0:len(t)], E22_exp[0:len(t)], 'r.')

nt=len(t)-1
plt.xlabel('time [seconds]')
plt.ylabel('Transverse Strain [-]')
plt.title('Complex loading (1)')
plt.legend(('E22 Crank_Nicholson','E22 Experiment'),
           shadow=True, loc=(0.5, 0.4), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([0., 5550, -0.0015, 0.0001])
plt.show()


#------------------------------------------------------------------------------
# =============================================================================
# Rsponce with Crank_Nicholson (complex loading (2))
# =============================================================================
#------------------------------------------------------------------------------
# =============================================================================
# Responce Epsilon (E11,E22,E33,E44,E55,E66)
# =============================================================================
print('\n')
print('=============================================================================')
print('Validation with complex loading (2)')
print('=============================================================================')


excel_file='Histoire_Chargement_2.xls'
D=6
(t, epsilon_Crank_Nicholson)=fluage_Crank_Nicholson(excel_file,k_inv,mu_inv,landa,D)

#Just for plot --------------------------------------------------------
df = pd.read_excel(excel_file, sheet_name='Sheet1')
c0,c1,c2,c3,c4,c5,c6,c7,c8=df
column_7 = df[c7]; column_8 = df[c8]
E11_exp=  np.zeros((len(column_7),1))
E22_exp=  np.zeros((len(column_8),1))
for i in range(0,len(column_7)):
    E11_exp[i]=column_7[i]
    E22_exp[i]=column_8[i]


# Curves -------------------------------------------------------------
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
plt.plot(t[0:len(t)], epsilon_Crank_Nicholson[0,:], 'b--')
plt.plot(t[0:len(t)], E11_exp[0:len(t)], 'r.')

nt=len(t)
plt.xlabel('time [seconds]')
plt.ylabel('Axial Strain [-]')
plt.title('Complex loading (2)')
plt.legend(('E11 Crank_Nicholson','E11 Experiment'),
           shadow=True, loc=(0.5, 0.5), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([0., 5000, 0., 0.003])
plt.show()


figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
plt.plot(t[0:len(t)], epsilon_Crank_Nicholson[1,:], 'b--')
plt.plot(t[0:len(t)], E22_exp[0:len(t)], 'r.')

nt=len(t)
plt.xlabel('time [seconds]')
plt.ylabel('Transverse Strain [-]')
plt.title('Complex loading (2)')
plt.legend(('E22 Crank_Nicholson','E22 Experiment'),
           shadow=True, loc=(0.5, 0.3), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([0., 5000, -0.0015, 0.0])
plt.show()


# =============================================================================
# =============================================================================
# # ANSYS
# =============================================================================
# =============================================================================
print('\n')
print('=============================================================================')
print('Verifying with ANSYS (FEM), Simple Geometry')
print('=============================================================================')
## =============================================================================
## Calculation S (With our alpha and betha)
## =============================================================================
def Sv_real_creep_test(alpha,betha):
    Sv=[]
    (I,J,K)=I_J_K_4orten_isotropes()
    J_V=tensor4_to_Voigt_tensor2(J)
    K_V=tensor4_to_Voigt_tensor2(K)
    for i in range(0,len(alpha)):       
        Si= alpha[i]*J_V + betha[i]*K_V
        Sv.append(Si)
    return Sv    

S=Sv_real_creep_test(alpha,betha)


# =============================================================================
# INTERCONVENTION
# =============================================================================
print('\n')
print('*****************')
print('INTERCONVENTION')
print('*****************')
(C0, Cn, w)=interconversion_S_to_C(S,landa)
 
# =============================================================================
# Finding alpha and betha for C
# =============================================================================
(alpha_C, betha_C)=alphai_bethai_for_S2_or_C2(Cn)
(alpha0_C, betha0_C)= alphs0_betha0_for_S02_or_C02(C0)

# =============================================================================
# Finding k and mu for alpha_C and betha_C
# =============================================================================
#C=akpha(J) + betha(K) ---> C=3k(J) + 2mu(K)  ---> k=akpha/3, mu=betha/2
# BUT IT SHOULD BE KEPTS IN INTERCONVERTION (IN MY OPINION) - IMPORTANT NOTE
# DONC  k=akpha*3, mu=betha*2
def k_mu_of_C(alpha,betha):
    k=[]; mu=[];
    for i in range(0,len(alpha)):
        ki=alpha[i]*3.
        mui=betha[i]*2.
        k.append(ki)
        mu.append(mui)
    return k,mu

(k,mu)=k_mu_of_C(alpha_C, betha_C)

# =============================================================================
# ANSYS parameters
# =============================================================================
print('*****************')
print('ANSYS parameters')
print('*****************')
def ANSYS_k(k):
    k0_ANSYS=0.
    for i in range(0,len(k)):
        k0_ANSYS= k[i] + k0_ANSYS
    
    alphak_ANSYS=[]
    for i in range(1,len(k)):
        alphak_ANSYS.append(k[i]/k0_ANSYS)
    
    alphak_inf_ANSYS= k[0] / k0_ANSYS
        
    return k0_ANSYS, alphak_ANSYS, alphak_inf_ANSYS
#
#    
def ANSYS_mu(mu):
    mu0_ANSYS=0.
    for i in range(0,len(mu)):
        mu0_ANSYS= mu[i] + mu0_ANSYS
    
    alphamu_ANSYS=[]
    for i in range(1,len(mu)):
        alphamu_ANSYS.append(mu[i]/mu0_ANSYS)
    
    alphamu_inf_ANSYS= mu[0] / mu0_ANSYS
           
    return mu0_ANSYS, alphamu_ANSYS, alphamu_inf_ANSYS


(k0_ANSYS, alphak_ANSYS, alphak_inf_ANSYS) = ANSYS_k(k) 

(mu0_ANSYS, alphamu_ANSYS, alphamu_inf_ANSYS) = ANSYS_mu(mu)
    
def ANSYS_tau(w):
    tau_ANSYS=[]
    for i in range(0,len(w)):
        tau_ANSYS.append(1./w[i])
    return tau_ANSYS

tau_ANSYS= ANSYS_tau(w)  
    
v0_ANSYS=(3.*k0_ANSYS - 2*mu0_ANSYS) / (2*mu0_ANSYS + 6*k0_ANSYS)
E0_ANSYS=(9.*k0_ANSYS*mu0_ANSYS) / (mu0_ANSYS + 3.*k0_ANSYS)

print('-----------------------------------------')
print('v0_ANSYS=',v0_ANSYS)
print('E0_ANSYS=',E0_ANSYS)

print('-----------------------------------------')

j=0      
for i in range(0,len(alphak_ANSYS)):
    if abs(alphak_ANSYS[i])<1.0e-8:
#        note='cancel'
        pass
    else: 
        note='ok'
        j=j+1
        print('TBDATA,,',alphak_ANSYS[i],',',tau_ANSYS[i],',')
        
#    print(i+1,':','TBDATA,,',alphak_ANSYS[i],',',tau_ANSYS[i],',   ',note)
#    print('TBDATA,,',alphak_ANSYS[i],',',tau_ANSYS[i],',')
print('number of ok alphak_ANSYS=',j)

#for i in range(0,len(alphak_ANSYS)):
#    print('TBDATA,,',alphak_ANSYS[i],',',tau_ANSYS[i],',')
#print('All of the ok alphak_ANSYS=',len(alphak_ANSYS))
print('-----------------------------------------')
print('#: TBDATA,, alphamu_ANSYS, tau_ANSYS  note:ok or cancel')
print('\n')

j=0      
for i in range(0,len(alphamu_ANSYS)):
    if abs(alphamu_ANSYS[i])<1.0e-8:
#        note='cancel'
        pass
    else:
        note='ok'
        j=j+1
        print('TBDATA,,',alphamu_ANSYS[i],',',tau_ANSYS[i],',')
        
#    print(i+1,':','TBDATA,,',alphamu_ANSYS[i],',',tau_ANSYS[i],',   ',note)
#    print('TBDATA,,',alphamu_ANSYS[i],',',tau_ANSYS[i],',')

print('number of ok alphamu_ANSYS=',j )

#print('All of the alphamu_ANSYS=',len(alphamu_ANSYS) )
#for i in range(0,len(alphamu_ANSYS)):
#    print('TBDATA,,',abs(alphamu_ANSYS[i]),',',tau_ANSYS[i],',')
#print('All of the alphamu_ANSYS=',len(alphamu_ANSYS) )
#

print('=============================================================================')
print('=============================================================================')
print('NOTE: I had some neglectable values which were negative')
print('for example: -1x10^-23')
print('I checked everything. It was not because of Voight_notation or even neglectable values in alpha and betha')
print('Generally, it was because if 3D interconvertion')
print('because of this I talked to Elias, he said these values are ok. but you can check just for 1D')
print('After 6hr working I wrote for 1D (it was harder than 3D)')
print('but as you can see in 1D solution there is not any negative value')
print('but I do not know why still we have a difference in simple geometry. I checked everything.')
print('I think my E0 and v0 is gigger than what they are soppuse to be')
print('=============================================================================')
print('=============================================================================')



#Just for plot --------------------------------------------------------
excel_file='fluage-recouvrance-5MPa-ANSYS-Geometrie-Simplement.xls'

df = pd.read_excel(excel_file, sheet_name='Sheet1') 
ct,cs,ce11,ce22,    c0,c1,c2,c3=df
#
column_0 = df[c0]; column_1 = df[c1]; column_2 = df[c2]; column_3 = df[c3];

time_ANSYS=  np.zeros((len(column_0),1))
E22_ANSYS=  np.zeros((len(column_1),1))
E11_ANSYS=  np.zeros((len(column_2),1))
for i in range(0,len(column_7)):
    time_ANSYS[i]=column_0[i]
    E22_ANSYS[i]=column_1[i]
    E11_ANSYS[i]=column_2[i]

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
plt.plot(t_exp_5, E11_model_after_optimiation_5, 'g--')
plt.plot(t_exp_5, E11_exp_5, 'r.')
plt.plot(time_ANSYS, E11_ANSYS, 'b-.')

plt.xlabel('time [seconds]')
plt.ylabel('[-]')
plt.title('Strain Responce E_11')
plt.legend(('E_11_model_5','E_11_exp_5','E_11_ANSYS_5'),
               shadow=True, loc=(0.7, 0.4), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([0, 4000, 0., 0.003])
plt.show()


figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
plt.plot(t_exp_5, E22_model_after_optimiation_5, 'g--')
plt.plot(t_exp_5, E22_exp_5, 'r.')
plt.plot(time_ANSYS, E22_ANSYS, 'b-.')

plt.xlabel('time [seconds]')
plt.ylabel('[-]')
plt.title('Strain Responce E_22')
plt.legend(('E_22_model_5','E_22_exp_5','E_22_ANSYS_5'),
               shadow=True, loc=(0.7, 0.4), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([0, 4000, -0.001, 0.0005])
plt.show()


print('\n')
print('=============================================================================')
print('Verifying with ANSYS (FEM), Complex Geometry')
print('=============================================================================')


excel_file='Geometrie_complexe_LAB.xls'

df = pd.read_excel(excel_file, sheet_name='Sheet1') 
c1,c2,c3,c4,    c5,c6,c7,c8,    c9,c10,c11,c12=df

column_1 = df[c1]; column_2 = df[c2]; column_3 = df[c3]; column_4 = df[c4];
column_5 = df[c5]; column_6 = df[c6]; column_7 = df[c7]; column_8 = df[c8];
column_9 = df[c9]; column_10 = df[c10]; column_11 = df[c11]; column_12 = df[c12];


time_exp=  np.zeros((len(column_1),1))
sigma_exp=  np.zeros((len(column_2),1))
E_jauge_1=  np.zeros((len(column_3),1))
E_jauge_2=  np.zeros((len(column_4),1))

for i in range(0,len(column_1)):
    time_exp[i]=column_1[i]
    sigma_exp[i]=column_2[i]
    E_jauge_1[i]=column_3[i]
    E_jauge_2[i]=column_4[i]



time_ANSYS_1=  np.zeros((len(column_5),1))
E_ANSYS_jauge_1X=  np.zeros((len(column_6),1))
E_ANSYS_jauge_1Y=  np.zeros((len(column_7),1))
E_ANSYS_jauge_1XY=  np.zeros((len(column_8),1))

time_ANSYS_2=  np.zeros((len(column_9),1))
E_ANSYS_jauge_2X=  np.zeros((len(column_10),1))
E_ANSYS_jauge_2Y=  np.zeros((len(column_11),1))
E_ANSYS_jauge_2XY=  np.zeros((len(column_12),1))


for i in range(0,34):
    time_ANSYS_1[i]= column_5[i]
    E_ANSYS_jauge_1X[i]= column_6[i]
    E_ANSYS_jauge_1Y[i]= column_7[i]
    E_ANSYS_jauge_1XY[i]= column_8[i]
    
    time_ANSYS_2[i]= column_9[i]
    E_ANSYS_jauge_2X[i]= column_10[i]
    E_ANSYS_jauge_2Y[i]= column_11[i]
    E_ANSYS_jauge_2XY[i]= column_12[i]

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
plt.plot(time_exp, E_jauge_2, 'r.')
plt.plot(time_ANSYS_2, E_ANSYS_jauge_2Y, 'b-')

plt.xlabel('time [seconds]')
plt.ylabel('[-]')
plt.title('Strain Responce Gauge 2 (E_y)')
plt.legend(('Experiment','ANSYS'),
               shadow=True, loc=(0.7, 0.4), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([0, 4500, 0., 0.00014])
plt.show()


# =============================================================================
# =============================================================================
# Changement de base d'un tenseur 
# =============================================================================
# =============================================================================

epsilon_jauge_1=np.zeros((6,34))
epsilon_jauge_1_45=np.zeros((6,34))


theta=pi*45.0/180.0
P=[[cos(theta), -sin(theta), 0.],
   [sin(theta), cos(theta), 0.],
   [0., 0., 1.]]


for i in range(0,34):
    epsilon_jauge_1[0,i]=E_ANSYS_jauge_1X[i]
    epsilon_jauge_1[1,i]=E_ANSYS_jauge_1Y[i]
    epsilon_jauge_1[2,i]=E_ANSYS_jauge_1XY[i]
    
for i in range(0,34):
    e_Y=epsilon_jauge_1[:,i]
    
    e_45=Changement_de_base_tensor1(e_Y,P)
    epsilon_jauge_1_45[:,i]=e_45


epsilon_jauge_1=np.zeros((3,3,34))
epsilon_jauge_1_45=np.zeros((3,3,34))
for i in range(0,34):
    epsilon_jauge_1[0,0,i]=E_ANSYS_jauge_1X[i]
    epsilon_jauge_1[1,1,i]=E_ANSYS_jauge_1Y[i]
    epsilon_jauge_1[0,1,i]=0.5*E_ANSYS_jauge_1XY[i]
    epsilon_jauge_1[1,0,i]=0.5*E_ANSYS_jauge_1XY[i]


for i in range(0,34):
    e_Y=epsilon_jauge_1[:,:,i]
    
    e_45=Changement_de_base_tensor2(e_Y,P)
    epsilon_jauge_1_45[:,:,i]=e_45

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
plt.plot(time_exp, E_jauge_1, 'r.')
plt.plot(time_ANSYS_1[0:34], epsilon_jauge_1_45[1,1,:], 'b-')

plt.xlabel('time [seconds]')
plt.ylabel('[-]')
plt.title('Strain Responce Gauge 2 (E_y)')
plt.legend(('Experiment','ANSYS'),
               shadow=True, loc=(0.7, 0.4), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([0, 4500, 0.0, 0.0015])
plt.show()

#
## =============================================================================
## =============================================================================
## # TRY WITH 1D
## =============================================================================
## =============================================================================
#
def Sv_real_creep_test_1D(alpha,betha):
    Sv=[]
    J_V=(1./3.)*np.array([1.])
    K_V=(1./3.)*np.array([2.])
    for i in range(0,len(alpha)):       
        Si= alpha[i]*J_V + betha[i]*K_V
        Sv.append(Si)
    return Sv 

S1D=Sv_real_creep_test_1D(alpha,betha)
(C01D, Cn1D, w1D)=interconversion_S_to_C_1D(S1D,landa)


def alphai_bethai_C_1D(C_S):
    alpha=[]; betha=[];
    for i in range(0,len(C_S)):
        alpha.append(C_S[i]/3.)
        betha.append(2.*C_S[i]/3.)
    return alpha, betha

def alpha0_betha0_C0_1D(C_S):
    alpha=[]; betha=[];
    alpha.append(C_S/3.)
    betha.append(2.*C_S/3.)
    return alpha, betha
 
(alpha1D_C, betha1D_C)=alphai_bethai_C_1D(Cn1D)
(alpha01D_C, betha01D_C)= alpha0_betha0_C0_1D(C01D)
    

(k1D,mu1D)=k_mu_of_C(alpha1D_C, betha1D_C)
(k01D_ANSYS, alphak1D_ANSYS, alphak1D_inf_ANSYS) = ANSYS_k(k1D) 
(mu01D_ANSYS, alphamu1D_ANSYS, alphamu1D_inf_ANSYS) = ANSYS_mu(mu1D)

tau1D_ANSYS= ANSYS_tau(w1D)  

print('-----------------------------------------')
print('#: TBDATA,, alphak1D_ANSYS, tau1D_ANSYS')
print('\n')
for i in range(0,len(alphak1D_ANSYS)):
    print('TBDATA,,',float(alphak1D_ANSYS[i]),',',tau1D_ANSYS[i],',')  
    
print('All of the ok alphak1D_ANSYS=',len(alphak1D_ANSYS))

print('-----------------------------------------')
print('#: TBDATA,, alphamu1D_ANSYS, tau1D_ANSYS')
print('\n')

for i in range(0,len(alphamu1D_ANSYS)):
    print('TBDATA,,',float(alphamu1D_ANSYS[i]),',',tau1D_ANSYS[i],',')
print('All of the alphamu1D_ANSYS=',len(alphamu1D_ANSYS) )


v01D_ANSYS=(3.*k01D_ANSYS - 2*mu01D_ANSYS) / (2*mu01D_ANSYS + 6*k01D_ANSYS)

E01D_ANSYS=(9.*k01D_ANSYS*mu01D_ANSYS) / (mu01D_ANSYS + 3.*k01D_ANSYS)

print('v01D_ANSYS=',v01D_ANSYS)
print('E01D_ANSYS=',E01D_ANSYS)
print('-----------------------------------------')
print('WRONG: The elastic molules for 3D and 1D should be the same')
print('In the following we use interconvention for alphai and bethai seperately')
print('-----------------------------------------')



        
def alpha_betha_np(alpha_betha):
    ab=[]
    x=np.array([1.])
    for i in range(0,len(alpha_betha)):       
        Si= alpha_betha[i]*x 
        ab.append(Si)
    return ab 


a=alpha_betha_np(alpha) #like S1D
b=alpha_betha_np(betha) #like S1D

(C01Da, Cn1Da, w1Da)=interconversion_S_to_C_1D(a,landa)
(C01Db, Cn1Db, w1Db)=interconversion_S_to_C_1D(b,landa)
# w1Da = w1Db approximately       


def k_mu_of_Ca_Cb(Ca,Cb):
    k=[]; mu=[];
    for i in range(0,len(Ca)):
        ki=Ca[i]*3.
        mui=Cb[i]*2.
        k.append(ki)
        mu.append(mui)
    return k,mu

(k,mu)=k_mu_of_Ca_Cb(Cn1Da, Cn1Db) 
   
(k01D_ANSYS, alphak1D_ANSYS, alphak1D_inf_ANSYS) = ANSYS_k(k) 
(mu01D_ANSYS, alphamu1D_ANSYS, alphamu1D_inf_ANSYS) = ANSYS_mu(mu)        

print('----------------------------')
print('k_1D_After_interconversion=')
print(k)
print('\n')
print('mu_1D_After_interconversion=')
print(mu)
print('----------------------------')


# w1Da = w1Db approximately       

tau1D_ANSYS= ANSYS_tau(w1Da)        
v01D_ANSYS=(3.*k01D_ANSYS - 2*mu01D_ANSYS) / (2*mu01D_ANSYS + 6*k01D_ANSYS)      
E01D_ANSYS=(9.*k01D_ANSYS*mu01D_ANSYS) / (mu01D_ANSYS + 3.*k01D_ANSYS)

print('-----------------------------------------')
print('v01D_ANSYS=',v01D_ANSYS)
print('E01D_ANSYS=',E01D_ANSYS)
print('-----------------------------------------')
print('#: TBDATA,, alphak1D_ANSYS, tau1D_ANSYS')
print('\n')
for i in range(0,len(alphak1D_ANSYS)):
    print('TBDATA,,',float(alphak1D_ANSYS[i]),',',tau1D_ANSYS[i],',')  
    
print('All of the ok alphak1D_ANSYS=',len(alphak1D_ANSYS))

print('-----------------------------------------')
print('#: TBDATA,, alphamu1D_ANSYS, tau1D_ANSYS')
print('\n')

for i in range(0,len(alphamu1D_ANSYS)):
    print('TBDATA,,',float(alphamu1D_ANSYS[i]),',',tau1D_ANSYS[i],',')
print('All of the alphamu1D_ANSYS=',len(alphamu1D_ANSYS) )



#Just for plot --------------------------------------------------------
print('-----------------------------------------')
print('VALIDATION FOR 1D - SIMPLE GEOMETRY')
print('-----------------------------------------')


excel_file='fluage-recouvrance-5MPa-ANSYS-Geometrie-Simplement_1D.xls'

df = pd.read_excel(excel_file, sheet_name='Sheet1') 
ct,cs,ce11,ce22,    c0,c1,c2,c3=df
#
column_0 = df[c0]; column_1 = df[c1]; column_2 = df[c2]; column_3 = df[c3];

time_ANSYS=  np.zeros((len(column_0),1))
E22_ANSYS=  np.zeros((len(column_1),1))
E11_ANSYS=  np.zeros((len(column_2),1))
for i in range(0,len(column_7)):
    time_ANSYS[i]=column_0[i]
    E22_ANSYS[i]=column_1[i]
    E11_ANSYS[i]=column_2[i]

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
plt.plot(t_exp_5, E11_model_after_optimiation_5, 'g--')
plt.plot(t_exp_5, E11_exp_5, 'r.')
plt.plot(time_ANSYS, E11_ANSYS, 'b-.')

plt.xlabel('time [seconds]')
plt.ylabel('[-]')
plt.title('Strain Responce E_11')
plt.legend(('E_11_model_5','E_11_exp_5','E_11_ANSYS_5'),
               shadow=True, loc=(0.7, 0.4), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([0, 4000, 0., 0.003])
plt.show()


figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
plt.plot(t_exp_5, E22_model_after_optimiation_5, 'g--')
plt.plot(t_exp_5, E22_exp_5, 'r.')
plt.plot(time_ANSYS, E22_ANSYS, 'b-.')

plt.xlabel('time [seconds]')
plt.ylabel('[-]')
plt.title('Strain Responce E_22')
plt.legend(('E_22_model_5','E_22_exp_5','E_22_ANSYS_5'),
               shadow=True, loc=(0.7, 0.4), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([0, 4000, -0.001, 0.0005])
plt.show()


print('-----------------------------------------')
print('VALIDATION FOR 1D - COMPLEX GEOMETRY')
print('-----------------------------------------')

print('-----------------------------------------')
print('Pay attention')
print('1/2 epsilon_XY_ANSYS = epsolon_XY in our case')
print('ANSYS gives gamma')
print('-----------------------------------------')



excel_file='Geometrie_complexe_LAB_1D.xls'

df = pd.read_excel(excel_file, sheet_name='Sheet1') 
c1,c2,c3,c4,    c5,c6,c7,c8,    c9,c10,c11,c12=df

column_1 = df[c1]; column_2 = df[c2]; column_3 = df[c3]; column_4 = df[c4];
column_5 = df[c5]; column_6 = df[c6]; column_7 = df[c7]; column_8 = df[c8];
column_9 = df[c9]; column_10 = df[c10]; column_11 = df[c11]; column_12 = df[c12];


time_exp=  np.zeros((len(column_1),1))
sigma_exp=  np.zeros((len(column_2),1))
E_jauge_1=  np.zeros((len(column_3),1))
E_jauge_2=  np.zeros((len(column_4),1))

for i in range(0,len(column_1)):
    time_exp[i]=column_1[i]
    sigma_exp[i]=column_2[i]
    E_jauge_1[i]=column_3[i]
    E_jauge_2[i]=column_4[i]



time_ANSYS_1=  np.zeros((len(column_5),1))
E_ANSYS_jauge_1X=  np.zeros((len(column_6),1))
E_ANSYS_jauge_1Y=  np.zeros((len(column_7),1))
E_ANSYS_jauge_1XY=  np.zeros((len(column_8),1))

time_ANSYS_2=  np.zeros((len(column_9),1))
E_ANSYS_jauge_2X=  np.zeros((len(column_10),1))
E_ANSYS_jauge_2Y=  np.zeros((len(column_11),1))
E_ANSYS_jauge_2XY=  np.zeros((len(column_12),1))


for i in range(0,34):
    time_ANSYS_1[i]= column_5[i]
    E_ANSYS_jauge_1X[i]= column_6[i]
    E_ANSYS_jauge_1Y[i]= column_7[i]
    E_ANSYS_jauge_1XY[i]= column_8[i]
    
    time_ANSYS_2[i]= column_9[i]
    E_ANSYS_jauge_2X[i]= column_10[i]
    E_ANSYS_jauge_2Y[i]= column_11[i]
    E_ANSYS_jauge_2XY[i]= column_12[i]

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
plt.plot(time_exp, E_jauge_2, 'r.')
plt.plot(time_ANSYS_2, E_ANSYS_jauge_2Y, 'b-')

plt.xlabel('time [seconds]')
plt.ylabel('Strain [-]')
plt.title('Strain Response of Gauge 2 (E_y)')
plt.legend(('Experiment','ANSYS'),
               shadow=True, loc=(0.7, 0.4), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([0, 4500, 0., 0.00014])
plt.show()




# =============================================================================
# =============================================================================
# Changement de base d'un tenseur 
# =============================================================================
# =============================================================================
epsilon_jauge_1=np.zeros((6,34))
epsilon_jauge_1_45=np.zeros((6,34))


theta=pi*45.0/180.0
P=[[cos(theta), -sin(theta), 0.],
   [sin(theta), cos(theta), 0.],
   [0., 0., 1.]]


for i in range(0,34):
    epsilon_jauge_1[0,i]=E_ANSYS_jauge_1X[i]
    epsilon_jauge_1[1,i]=E_ANSYS_jauge_1Y[i]
    epsilon_jauge_1[2,i]=E_ANSYS_jauge_1XY[i]
    
for i in range(0,34):
    e_Y=epsilon_jauge_1[:,i]
    
    e_45=Changement_de_base_tensor1(e_Y,P)
    epsilon_jauge_1_45[:,i]=e_45


epsilon_jauge_1=np.zeros((3,3,34))
epsilon_jauge_1_45=np.zeros((3,3,34))
for i in range(0,34):
    epsilon_jauge_1[0,0,i]=E_ANSYS_jauge_1X[i]
    epsilon_jauge_1[1,1,i]=E_ANSYS_jauge_1Y[i]
    epsilon_jauge_1[0,1,i]=0.5*E_ANSYS_jauge_1XY[i]
    epsilon_jauge_1[1,0,i]=0.5*E_ANSYS_jauge_1XY[i]


for i in range(0,34):
    e_Y=epsilon_jauge_1[:,:,i]
    
    e_45=Changement_de_base_tensor2(e_Y,P)
    epsilon_jauge_1_45[:,:,i]=e_45

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
plt.plot(time_exp, E_jauge_1, 'r.')
plt.plot(time_ANSYS_1[0:34], epsilon_jauge_1_45[1,1,:], 'b-')

plt.xlabel('time [seconds]')
plt.ylabel('Strain [-]')
plt.title('Strain Response of Gauge 1 (E_y)')
plt.legend(('Experiment','ANSYS'),
               shadow=True, loc=(0.7, 0.4), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([0, 4500, 0.0, 0.0015])
plt.show()


print('If you need come approximation functions for alpha or betha please see back_up file')

print('-----------------------------------------')
print('END OF THE LINEAR VISCOELASTIC. DONT TUCH ANYTHING. JUST WRITE YOUR REPORT')
print('9 Nov. 2018 11:45 pm - LM2 Lab')
print('-----------------------------------------')








print('**********************************************************************')
print('Non linear Viscoelastique')
print('**********************************************************************')



def Si_1D_alpha_betha(alpha,betha):
    J=(1./3.)*np.array([1.])
    K=(1./3.)*np.array([2.])
    Si=[]
    
    for i in range(0,len(alpha)):
        s=alpha[i]*J + betha[i]*K
        Si.append(s)
    
    return Si

Si_1D = Si_1D_alpha_betha(alpha,betha)




# =============================================================================
# g0_sigma0    
# =============================================================================
def g0(Si_1D, e11_exp, sigma_exp, nt):
    
    epsilon0=float(e11_exp[nt]) #'why? becuase we considered it @ t=t1 not t=0'
    sigma0=float(max(sigma_exp))
    S0=float(Si_1D[0])
    
    g0=    epsilon0 / (S0*sigma0) 
    
    return g0
    
#Linear Domain
g0_2 = g0(Si_1D, e11_exp_2MP, sigma_2MP, 11)
g0_5 = g0(Si_1D, e11_exp_5MP, sigma_5MP, 11)
g0_10 = g0(Si_1D, e11_exp_10MP, sigma_10MP, 11)

#Non_Linear Domain
g0_13 = g0(Si_1D, e11_exp_13MP, sigma_13MP, 11)
g0_16 = g0(Si_1D, e11_exp_16MP, sigma_16MP, 11)
g0_20 = g0(Si_1D, e11_exp_20MP, sigma_20MP, 11)
g0_22 = g0(Si_1D, e11_exp_22MP, sigma_22MP, 11)
g0_25 = g0(Si_1D, e11_exp_25MP, sigma_25MP, 11)
g0_30 = g0(Si_1D, e11_exp_30MP, sigma_30MP, 11)
g0_35 = g0(Si_1D, e11_exp_35MP, sigma_35MP, 11)


# =============================================================================
# g1_sigma0
# =============================================================================

t2=905.
frame_t2=61 #the maximum of epsilon for all loadings

def g1(Si_1D, e11_exp, sigma_exp, g0, t_before, t_after):
    S0=float(Si_1D[0])
    
    sigma0=float(max(sigma_exp))
        
    delta_epsilon0 = g0*S0*sigma0
    
    g1 = (e11_exp[t_before] - delta_epsilon0) / e11_exp[t_after]
    
    return g1

#Linear Domain
g1_2 = g1(Si_1D, e11_exp_2MP, sigma_2MP, g0_2, 61, 73)
g1_5 = g1(Si_1D, e11_exp_5MP, sigma_5MP, g0_5, 61, 73)
g1_10 = g1(Si_1D, e11_exp_10MP, sigma_10MP, g0_10, 61, 73)

#NON-Linear Domain
g1_13 = g1(Si_1D, e11_exp_13MP, sigma_13MP, g0_13, 61, 73)
g1_16 = g1(Si_1D, e11_exp_16MP, sigma_16MP, g0_16, 61, 73)
g1_20 = g1(Si_1D, e11_exp_20MP, sigma_20MP, g0_20, 61, 73)
g1_22 = g1(Si_1D, e11_exp_22MP, sigma_22MP, g0_22, 61, 73)
g1_25 = g1(Si_1D, e11_exp_25MP, sigma_25MP, g0_25, 61, 73)
g1_30 = g1(Si_1D, e11_exp_30MP, sigma_30MP, g0_30, 61, 73)
g1_35 = g1(Si_1D, e11_exp_35MP, sigma_35MP, g0_35, 61, 73)


# =============================================================================
# g2_sigma0 et g3_sigma0: Optimization problem 
# =============================================================================
    
# =============================================================================
# INPUT: t_vector   
# =============================================================================
def epsilon_model_nonlinear(x, t_vector ,Si_1D, landa, g0, g1, sigma_exp, t00, t0):
    'x includes g2 and g3'    
    sigma0=float(max(sigma_exp))
    S0=float(Si_1D[0])
    g0 = float(g0)
    g1 = float(g1)
    
    def delta_St(t):
        delta_S=0.
        for i in range(0,len(landa)):
            Si=float(Si_1D[i+1])
            landai=float(landa[i])
            delta_S = (Si)*(1.0 - exp(-t*landai)) + delta_S
        
        return delta_S
    
        
    def f_1(t_scalar):
        if t_scalar> t00 and t_scalar< t0:
            delta_S_1 =  delta_St(  (t_scalar)/(x[1]**2.) )
            
            part_1 = ( g0*S0*sigma0  +  g1*(x[0]**2.)*delta_S_1*sigma0 )
        else:
            part_1=0.
            
        return part_1
    
    def f_2(t_scalar):
        if t_scalar> t0:
            delta_S_2 =  delta_St( (t0/(x[1]**2.)) + t_scalar - t0 ) 
                                                  
            delta_S_3 =  delta_St(t_scalar - t0 )
            
            part_2 = ( delta_S_2 - delta_S_3 )*(x[0]**2.)*sigma0
        
        else:
            part_2=0.
    
        return part_2
    
    E_model_list=[]
    for i in range(0,len(t_vector)):
        t_scalar=t_vector[i]
        E_model_output=  f_1(t_scalar) + f_2(t_scalar) #+ f_0(t_scalar)
        E_model_list.append(E_model_output)
        
    return E_model_list
        
# =============================================================================
# INPUT: t_scalar
# =============================================================================

#def epsilon_model_nonlinear(x, t_scalar ,Si_1D, landa, g0, g1, sigma_exp, t00, t0):
#    'x includes g2 and g3'    
#    sigma0=float(max(sigma_exp))
#    S0=float(Si_1D[0])
#    g0 = float(g0)
#    g1 = float(g1)
#    
#    def delta_St(t):
#        delta_S=0.
#        for i in range(0,len(landa)):
#            Si=float(Si_1D[i+1])
#            landai=float(landa[i])
#            delta_S = (Si)*(1.0 - exp(-t*landai)) + delta_S
#        
#        return delta_S
#    
#        
#    def f_1(t_scalar):
#        if t_scalar> t00 and t_scalar< t0:
#            delta_S_1 =  delta_St(  (t_scalar)/(x[1]**2.) )
#            
#            part_1 = ( g0*S0*sigma0  +  g1*(x[0]**2.)*delta_S_1*sigma0 )
#        else:
#            part_1=0.
#            
#        return part_1
#    
#    def f_2(t_scalar):
#        if t_scalar> t0:
#            delta_S_2 =  delta_St( (t0/(x[1]**2.)) + t_scalar - t0 ) 
#                                                  
#            delta_S_3 =  delta_St(t_scalar - t0 )
#            
#            part_2 = ( delta_S_2 - delta_S_3 )*(x[0]**2.)*sigma0
#        
#        else:
#            part_2=0.
#    
#        return part_2
#    
##    E_model_list=[]
##    for i in range(0,len(t_vector)):
##        t_scalar=t_vector[i]
##        E_model_output=  f_1(t_scalar) + f_2(t_scalar) #+ f_0(t_scalar)
##        E_model_list.append(E_model_output)
#    
#    E_model_output=  f_1(t_scalar) + f_2(t_scalar) #+ f_0(t_scalar)
#    
#    return E_model_output
#        
# =============================================================================
# =============================================================================
# Residual: whole curve
# =============================================================================
#def res_nonlinear(x, t_vector ,Si_1D,landa, g0, g1, sigma_exp, t00, t0,ft00,ft0, e11_exp):
#    
#    epsilon_model= epsilon_model_nonlinear(x, t_vector ,Si_1D,landa, g0, g1, sigma_exp, t00, t0)    
#    res=[]
#    #I considered after frame 12 becuase before that is not in our model
#    for i in range(ft00,len(t_vector)):        
#        res.append( float( e11_exp[i] - epsilon_model[i] ) )
#    
#    return res


# =============================================================================
# Residual: between curve
# =============================================================================
def res_nonlinear(x, t_vector ,Si_1D,landa, g0, g1, sigma_exp, t00, t0,ft00,ft0, e11_exp):
    
    epsilon_model= epsilon_model_nonlinear(x, t_vector ,Si_1D,landa, g0, g1, sigma_exp, t00, t0)    
    res=[]
    #I considered after frame 12 becuase before that is not in our model
    for i in range(ft00,ft0):        
        res.append( float( e11_exp[i] - epsilon_model[i] ) )
    
    return res

# =============================================================================
# epsilon and time modifier                   
# =============================================================================
def epsilon_exp_modifier(e11_exp,ft00,ft0):
    eps_modified=[]
    for i in range(ft00,ft0):
        eps_modified.append(e11_exp[i])
    return eps_modified

def time_exp_modifier(time_exp,ft00,ft0):
    time_modified=[]
    for i in range(ft00,ft0):
        time_modified.append(time_exp[i])
    return time_modified

# =============================================================================
# g2 & g3 calculator
# =============================================================================
def g2_g3_from_x(x):
    g2 = x[0]*x[0]
    g3 = x[1]*x[1]
    return g2,g3


# =============================================================================
# =============================================================================
# # Optimization  
# =============================================================================
# =============================================================================
print('please always append float in your residual. Otherwise you would have error: ')
print('`fun` must return at most 1-d array_like.')   





# =============================================================================
# 2 MPa - Linear Domain
# =============================================================================
#g0
#g1

#------Optimization
#------args----
t_vector=t_2MP
g0=g0_2
g1=g1_2
sigma_exp = sigma_2MP
e11_exp = e11_exp_2MP
t00=6.; t0=905. 
ft00=12; ft0=61
x_initial = [0.0,0.0]

#g2, g3 ------Optimization
optimization_g2_g3= least_squares(res_nonlinear,x_initial, args = (t_vector ,Si_1D,landa, g0, g1, sigma_exp, t00, t0,ft00,ft0, e11_exp))
x_g2_g3_2=optimization_g2_g3.x
(g2_2,g3_2) = g2_g3_from_x(x_g2_g3_2)
print('-----------2 MPa------------LINEAR')
print('g0_2=',g0_2)
print('g1_2=',g1_2) 
print('g2_2=',g2_2)
print('g3_2=',g3_2)       

#Calculation epsilon with g0, g1, g2, g3
x=x_g2_g3_2
e11_2=epsilon_model_nonlinear(x, t_vector ,Si_1D, landa, g0, g1, sigma_exp, t00, t0)            



# =============================================================================
# 5 MPa - Linear Domain
# =============================================================================
#g0

#g1

#------Optimization
#------args----
t_vector=t_10MP
g0=g0_5
g1=g1_5
sigma_exp = sigma_5MP
e11_exp = e11_exp_5MP
t00=6.; t0=905. 
ft00=12; ft0=61
x_initial = [0.01,0.01]

#g2, g3 ------Optimization
optimization_g2_g3= least_squares(res_nonlinear,x_initial, args = (t_vector ,Si_1D,landa, g0, g1, sigma_exp, t00, t0,ft00,ft0, e11_exp))
x_g2_g3_5=optimization_g2_g3.x
(g2_5,g3_5) = g2_g3_from_x(x_g2_g3_5)
print('-----------5 MPa------------LINEAR')
print('g0_5=',g0_5)
print('g1_5=',g1_5) 
print('g2_5=',g2_5)
print('g3_5=',g3_5)       

#Calculation epsilon with g0, g1, g2, g3
x=x_g2_g3_5
e11_5=epsilon_model_nonlinear(x, t_vector ,Si_1D, landa, g0, g1, sigma_exp, t00, t0)            



# =============================================================================
# 10 MPa - Linear Domain
# =============================================================================
#g0

#g1

#------Optimization
#------args----
t_vector=t_10MP
g0=g0_10
g1=g1_10
sigma_exp = sigma_10MP
e11_exp = e11_exp_10MP
t00=6.; t0=905. 
ft00=12; ft0=61
x_initial = [0.01,0.01]

#g2, g3 ------Optimization
optimization_g2_g3= least_squares(res_nonlinear,x_initial, args = (t_vector ,Si_1D,landa, g0, g1, sigma_exp, t00, t0,ft00,ft0, e11_exp))
x_g2_g3_10=optimization_g2_g3.x
(g2_10,g3_10) = g2_g3_from_x(x_g2_g3_10)
print('-----------10 MPa------------LINEAR')
print('g0_10=',g0_10)
print('g1_10=',g1_10) 
print('g2_10=',g2_10)
print('g3_10=',g3_10)       

#Calculation epsilon with g0, g1, g2, g3
x=x_g2_g3_10
e11_10=epsilon_model_nonlinear(x, t_vector ,Si_1D, landa, g0, g1, sigma_exp, t00, t0)            




# =============================================================================
# =============================================================================
# # Non_Linear Domain
# =============================================================================
# =============================================================================

# =============================================================================
# 13 MPa - Non_Linear Domain
# =============================================================================
#g0

#g1

#------Optimization
#------args----
t_vector=t_13MP
g0=g0_13
g1=g1_13
sigma_exp = sigma_13MP
e11_exp = e11_exp_13MP
t00=6.; t0=905. 
ft00=12; ft0=61
x_initial = [0.01,0.01]

#g2, g3 ------Optimization
optimization_g2_g3= least_squares(res_nonlinear,x_initial, args = (t_vector ,Si_1D,landa, g0, g1, sigma_exp, t00, t0,ft00,ft0, e11_exp))
x_g2_g3_13=optimization_g2_g3.x
(g2_13,g3_13) = g2_g3_from_x(x_g2_g3_13)
print('-----------13 MPa------------')
print('g0_13=',g0_13)
print('g1_13=',g1_13) 
print('g2_13=',g2_13)
print('g3_13=',g3_13)       

#Calculation epsilon with g0, g1, g2, g3
x=x_g2_g3_13
e11_13=epsilon_model_nonlinear(x, t_vector ,Si_1D, landa, g0, g1, sigma_exp, t00, t0)            



# =============================================================================
# 16 MPa - Non_Linear Domain
# =============================================================================
#g0

#g1

#------Optimization
#------args----
t_vector=t_16MP
g0=g0_16
g1=g1_16
sigma_exp = sigma_16MP
e11_exp = e11_exp_16MP
t00=6.; t0=905. 
ft00=12; ft0=61
x_initial = [0.01,0.01]

#g2, g3 ------Optimization
optimization_g2_g3= least_squares(res_nonlinear,x_initial, args = (t_vector ,Si_1D,landa, g0, g1, sigma_exp, t00, t0,ft00,ft0, e11_exp))
x_g2_g3_16=optimization_g2_g3.x
(g2_16,g3_16) = g2_g3_from_x(x_g2_g3_16)
print('-----------16 MPa------------')
print('g0_16=',g0_16)
print('g1_16=',g1_16) 
print('g2_16=',g2_16)
print('g3_16=',g3_16)       

#Calculation epsilon with g0, g1, g2, g3
x=x_g2_g3_16
e11_16=epsilon_model_nonlinear(x, t_vector ,Si_1D, landa, g0, g1, sigma_exp, t00, t0)            




# =============================================================================
# 20 MPa - Non_Linear Domain
# =============================================================================
#g0

#g1

#------Optimization
#------args----
t_vector=t_20MP
g0=g0_20
g1=g1_20
sigma_exp = sigma_20MP
e11_exp = e11_exp_20MP
t00=6.; t0=905. 
ft00=12; ft0=61
x_initial = [0.01,0.01]

#g2, g3 ------Optimization
optimization_g2_g3= least_squares(res_nonlinear,x_initial, args = (t_vector ,Si_1D,landa, g0, g1, sigma_exp, t00, t0,ft00,ft0, e11_exp))
x_g2_g3_20=optimization_g2_g3.x
(g2_20,g3_20) = g2_g3_from_x(x_g2_g3_20)
print('-----------20 MPa------------') 
print('g0_20=',g0_20)
print('g1_20=',g1_20)  
print('g2_20=',g2_20)
print('g3_20=',g3_20)       

#Calculation epsilon with g0, g1, g2, g3
x=x_g2_g3_20 
e11_20=epsilon_model_nonlinear(x, t_vector ,Si_1D, landa, g0, g1, sigma_exp, t00, t0)            



# =============================================================================
# 22 MPa - Non_Linear Domain
# =============================================================================
#g0

#g1

#------Optimization
#------args----
t_vector=t_22MP
g0=g0_22
g1=g1_22
sigma_exp = sigma_22MP
e11_exp = e11_exp_22MP
t00=6.; t0=905. 
ft00=12; ft0=61
x_initial = [0.01,0.01]

#g2, g3 ------Optimization
optimization_g2_g3= least_squares(res_nonlinear,x_initial, args = (t_vector ,Si_1D,landa, g0, g1, sigma_exp, t00, t0,ft00,ft0, e11_exp))
x_g2_g3_22=optimization_g2_g3.x
(g2_22, g3_22) = g2_g3_from_x(x_g2_g3_22)
print('-----------22 MPa------------') 
print('g0_22=',g0_22)
print('g1_22=',g1_22)
print('g2_22=',g2_22)
print('g3_22=',g3_22)       

#Calculation epsilon with g0, g1, g2, g3
x=x_g2_g3_22
e11_22=epsilon_model_nonlinear(x, t_vector ,Si_1D, landa, g0, g1, sigma_exp, t00, t0)            


# =============================================================================
# 25 MPa - Non_Linear Domain
# =============================================================================
#g0

#g1

#------Optimization
#------args----
t_vector=t_25MP
g0=g0_25
g1=g1_25
sigma_exp = sigma_25MP
e11_exp = e11_exp_25MP
t00=6.; t0=905. 
ft00=12; ft0=61
x_initial = [0.01,0.01]

#g2, g3 ------Optimization
optimization_g2_g3= least_squares(res_nonlinear,x_initial, args = (t_vector ,Si_1D,landa, g0, g1, sigma_exp, t00, t0,ft00,ft0, e11_exp))
x_g2_g3_25=optimization_g2_g3.x
(g2_25, g3_25) = g2_g3_from_x(x_g2_g3_25)
print('-----------25 MPa------------') 
print('g0_25=',g0_25)
print('g1_25=',g1_25) 
print('g2_25=',g2_25)
print('g3_25=',g3_25)       

#Calculation epsilon with g0, g1, g2, g3
x=x_g2_g3_25
e11_25=epsilon_model_nonlinear(x, t_vector ,Si_1D, landa, g0, g1, sigma_exp, t00, t0)            



# =============================================================================
# 30 MPa - Non_Linear Domain
# =============================================================================
#g0

#g1

#------Optimization
#------args----
t_vector=t_30MP
g0=g0_30
g1=g1_30
sigma_exp = sigma_30MP
e11_exp = e11_exp_30MP
t00=6.; t0=905. 
ft00=12; ft0=61
x_initial = [0.01,0.01]

#g2, g3 ------Optimization
optimization_g2_g3= least_squares(res_nonlinear,x_initial, args = (t_vector ,Si_1D,landa, g0, g1, sigma_exp, t00, t0,ft00,ft0, e11_exp))
x_g2_g3_30=optimization_g2_g3.x
(g2_30, g3_30) = g2_g3_from_x(x_g2_g3_30)
print('-----------30 MPa------------') 
print('g0_30=',g0_30)
print('g1_30=',g1_30)    
print('g2_30=',g2_30)
print('g3_30=',g3_30)       

#Calculation epsilon with g0, g1, g2, g3
x=x_g2_g3_30
e11_30=epsilon_model_nonlinear(x, t_vector ,Si_1D, landa, g0, g1, sigma_exp, t00, t0)            


# =============================================================================
# 35 MPa - Non_Linear Domain
# =============================================================================
#g0

#g1

#------Optimization
#------args----
t_vector=t_35MP
g0=g0_35
g1=g1_35
sigma_exp = sigma_35MP
e11_exp = e11_exp_35MP
t00=6.; t0=905. 
ft00=12; ft0=61
x_initial = [0.01,0.01]

#g2, g3 ------Optimization
optimization_g2_g3= least_squares(res_nonlinear,x_initial, args = (t_vector ,Si_1D,landa, g0, g1, sigma_exp, t00, t0,ft00,ft0, e11_exp))
x_g2_g3_35=optimization_g2_g3.x
(g2_35, g3_35) = g2_g3_from_x(x_g2_g3_35)
print('-----------35 MPa------------') 
print('g0_35=',g0_35)
print('g1_35=',g1_35) 
print('g2_35=',g2_35)
print('g3_35=',g3_35)       

#Calculation epsilon with g0, g1, g2, g3
x=x_g2_g3_35
e11_35=epsilon_model_nonlinear(x, t_vector ,Si_1D, landa, g0, g1, sigma_exp, t00, t0)            



# =============================================================================
# Graph the results
# =============================================================================
figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k') 

plt.plot(t_13MP, e11_exp_13MP, 'r--.')
plt.plot(t_16MP, e11_exp_16MP, 'r--<')
plt.plot(t_20MP, e11_exp_20MP, 'r-->')
plt.plot(t_22MP, e11_exp_22MP, 'r--^')
plt.plot(t_25MP, e11_exp_25MP, 'r--8')
plt.plot(t_30MP, e11_exp_30MP, 'rs-')
plt.plot(t_35MP, e11_exp_35MP, 'r--p')


plt.plot(t_13MP, e11_13, 'b--')
plt.plot(t_16MP, e11_16, 'b--')
plt.plot(t_20MP, e11_20, 'b--')
plt.plot(t_22MP, e11_22, 'b--')
plt.plot(t_25MP, e11_25, 'b--')
plt.plot(t_30MP, e11_30, 'b--')
plt.plot(t_35MP, e11_35, 'b--')


plt.xlabel('time [s]')
plt.ylabel('E_11 [-]')
plt.title('non-linear viscoelastic (optimization problem, case I: 0 < t < 905)')
plt.legend(('e11_exp_13MP','e11_exp_16MP', 'e11_exp_20MP','e11_exp_22MP',
            'e11_exp_25MP' , 'e11_exp_30MP' , 'e11_exp_35MP', 
            
            'e11_model'),#, 'e11_model_20', 'e11_model_22',
#            'e11_model_25', 'e11_model_30', 'e11_model_35'),
               shadow=True, loc=(0.5, 0.3), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([0, 4000, 0., 0.016])
plt.show()



# =============================================================================
# g0 Graph
# =============================================================================
sigma_graph = [2,5,10,13,16,20,22,25,30,35]
g0 = [g0_2, g0_5, g0_10, g0_13, g0_16, g0_20, g0_22, g0_25, g0_30, g0_35]

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
plt.plot(sigma_graph, g0, 'ro-')

plt.xlabel('sigma')
plt.ylabel('g0')
plt.title('g0')
plt.legend(('g0'),
               shadow=True, loc=(0.7, 0.4), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([2, 35, 1, 1.2])
plt.show()

# =============================================================================
# g1 Graph
# =============================================================================
g1 = [g1_2, g1_5, g1_10, g1_13, g1_16, g1_20, g1_22, g1_25, g1_30, g1_35]

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
plt.plot(sigma_graph, g1, 'ro-')

plt.xlabel('sigma')
plt.ylabel('g1')
plt.title('g1')
plt.legend(('g1'),
               shadow=True, loc=(0.7, 0.4), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([2, 35, 0.9, 1.2])
plt.show()


# =============================================================================
# g2 Graph
# =============================================================================
g2 = [g2_2, g2_5, g2_10, g2_13, g2_16, g2_20, g2_22, g2_25, g2_30, g2_35]
sigma_graph = [2, 5, 10, 13, 16, 20, 22, 25, 30, 35]

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
plt.plot(sigma_graph, g2, 'ro-')

plt.xlabel('sigma')
plt.ylabel('g2')
plt.title('g2')
plt.legend(('g2'),
               shadow=True, loc=(0.4, 0.5), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([2, 35, 0., 3.])
plt.show()




# =============================================================================
# g3 Graph
# =============================================================================
g3 = [g3_2, g3_5, g3_10, g3_13, g3_16, g3_20, g3_22, g3_25, g3_30, g3_35]
sigma_graph = [2, 5, 10, 13,16,20,22,25,30,35]

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
plt.plot(sigma_graph, g3, 'ro-')

plt.xlabel('sigma')
plt.ylabel('g3')
plt.title('g3')
plt.legend(('g3'),
               shadow=True, loc=(0.4, 0.5), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([2, 35, 28, 42.])
plt.show()



print('''Discussion:
    1- As you can see the plots is good the the practical domain. It means that our model is just valid in this part. 
        So, I jus considered this part of my result in optimization. 
        The total results is good but as I mentioned I didnt considered after t0. So you can see that,
        after the t0 results is not very good. We can explain this based on the:
    2- the optimization doens't work on 2, 5, 20 MPa. because on that experiments we are in the linear domain
    But still we need to have valid results. I don't know why?
    An and don't want to talk about this in my report.. 

3- but the good point is that i use the same values for all the loadings.            
    ''')


print('\n')
    
# =============================================================================
# =============================================================================
print('========================================================================')
print('===================Considering the whole graph ==========================')
print('===================res_nonlinear_whole ==========================')
print('========================================================================')
   
# =============================================================================
# =============================================================================

def res_nonlinear_whole(x, t_vector ,Si_1D,landa, g0, g1, sigma_exp, t00, t0,ft00,ft0,ft3, ft4, e11_exp):
    
    epsilon_model= epsilon_model_nonlinear(x, t_vector ,Si_1D,landa, g0, g1, sigma_exp, t00, t0)    
    res=[]
    #I considered after frame 12 becuase before that is not in our model
    for i in range(ft00,ft0):        
        res.append( float( e11_exp[i] - epsilon_model[i] ) )
    
#ft3=75 # the second part after t3    
    for i in range(ft3,ft4):        
        res.append( float( e11_exp[i] - epsilon_model[i] ) )    
    
    return res

# =============================================================================
ft3=66 #START OPTIMIZATION
ft4=len(t_13MP)#115 #END OPTIMIZATION
# =============================================================================

# =============================================================================
# 2 MPa - Linear Domain
# =============================================================================
#g0
#g1

#------Optimization
#------args----
t_vector=t_2MP
g0=g0_2
g1=g1_2
sigma_exp = sigma_2MP
e11_exp = e11_exp_2MP
t00=6.; t0=905. 
ft00=12; ft0=61
x_initial = [0.01,0.01]

#g2, g3 ------Optimization
optimization_g2_g3= least_squares(res_nonlinear_whole,x_initial, args = (t_vector ,Si_1D,landa, g0, g1, sigma_exp, t00, t0,ft00,ft0,ft3,ft4, e11_exp))
x_g2_g3_2=optimization_g2_g3.x
(g2_2,g3_2) = g2_g3_from_x(x_g2_g3_2)
print('-----------2 MPa------------LINEAR')
print('g0_2=',g0_2)
print('g1_2=',g1_2) 
print('g2_2=',g2_2)
print('g3_2=',g3_2)       

#Calculation epsilon with g0, g1, g2, g3
x=x_g2_g3_2
e11_2=epsilon_model_nonlinear(x, t_vector ,Si_1D, landa, g0, g1, sigma_exp, t00, t0)            



# =============================================================================
# 5 MPa - Linear Domain
# =============================================================================
#g0

#g1

#------Optimization
#------args----
t_vector=t_10MP
g0=g0_5
g1=g1_5
sigma_exp = sigma_5MP
e11_exp = e11_exp_5MP
t00=6.; t0=905. 
ft00=12; ft0=61
x_initial = [0.01,0.01]

#g2, g3 ------Optimization
optimization_g2_g3= least_squares(res_nonlinear_whole,x_initial, args = (t_vector ,Si_1D,landa, g0, g1, sigma_exp, t00, t0,ft00,ft0,ft3,ft4, e11_exp))
x_g2_g3_5=optimization_g2_g3.x
(g2_5,g3_5) = g2_g3_from_x(x_g2_g3_5)
print('-----------5 MPa------------LINEAR')
print('g0_5=',g0_5)
print('g1_5=',g1_5) 
print('g2_5=',g2_5)
print('g3_5=',g3_5)       

#Calculation epsilon with g0, g1, g2, g3
x=x_g2_g3_5
e11_5=epsilon_model_nonlinear(x, t_vector ,Si_1D, landa, g0, g1, sigma_exp, t00, t0)            



# =============================================================================
# 10 MPa - Linear Domain
# =============================================================================
#g0

#g1

#------Optimization
#------args----
t_vector=t_10MP
g0=g0_10
g1=g1_10
sigma_exp = sigma_10MP
e11_exp = e11_exp_10MP
t00=6.; t0=905. 
ft00=12; ft0=61
x_initial = [0.01,0.01]

#g2, g3 ------Optimization
optimization_g2_g3= least_squares(res_nonlinear_whole,x_initial, args = (t_vector ,Si_1D,landa, g0, g1, sigma_exp, t00, t0,ft00,ft0,ft3,ft4, e11_exp))
x_g2_g3_10=optimization_g2_g3.x
(g2_10,g3_10) = g2_g3_from_x(x_g2_g3_10)
print('-----------10 MPa------------LINEAR')
print('g0_10=',g0_10)
print('g1_10=',g1_10) 
print('g2_10=',g2_10)
print('g3_10=',g3_10)       

#Calculation epsilon with g0, g1, g2, g3
x=x_g2_g3_10
e11_10=epsilon_model_nonlinear(x, t_vector ,Si_1D, landa, g0, g1, sigma_exp, t00, t0)            




# =============================================================================
# =============================================================================
# # Non_Linear Domain
# =============================================================================
# =============================================================================

# =============================================================================
# 13 MPa - Non_Linear Domain
# =============================================================================

#------Optimization
#------args----
t_vector=t_13MP
g0=g0_13
g1=g1_13
sigma_exp = sigma_13MP
e11_exp = e11_exp_13MP
t00=6.; t0=905. 
ft00=12; ft0=61; 
x_initial = [0.01,0.01]

#g2, g3 ------Optimization
optimization_g2_g3= least_squares(res_nonlinear_whole,x_initial, args = (t_vector ,Si_1D,landa, g0, g1, sigma_exp, t00, t0,ft00,ft0,ft3,ft4, e11_exp))
x_g2_g3_13=optimization_g2_g3.x
(g2_13,g3_13) = g2_g3_from_x(x_g2_g3_13)
print('-----------13 MPa------------')
print('g0_13=',g0_13)
print('g1_13=',g1_13) 
print('g2_13=',g2_13)
print('g3_13=',g3_13)       

#Calculation epsilon with g0, g1, g2, g3
x=x_g2_g3_13
e11_13=epsilon_model_nonlinear(x, t_vector ,Si_1D, landa, g0, g1, sigma_exp, t00, t0)            





# =============================================================================
# 16 MPa - Non_Linear Domain
# =============================================================================
#g0

#g1

#------Optimization
#------args----
t_vector=t_16MP
g0=g0_16
g1=g1_16
sigma_exp = sigma_16MP
e11_exp = e11_exp_16MP
t00=6.; t0=905. 
ft00=12; ft0=61
x_initial = [0.01,0.01]

#g2, g3 ------Optimization
optimization_g2_g3= least_squares(res_nonlinear_whole,x_initial, args = (t_vector ,Si_1D,landa, g0, g1, sigma_exp, t00, t0,ft00,ft0,ft3,ft4, e11_exp))
x_g2_g3_16=optimization_g2_g3.x
(g2_16,g3_16) = g2_g3_from_x(x_g2_g3_16)
print('-----------16 MPa------------')
print('g0_16=',g0_16)
print('g1_16=',g1_16) 
print('g2_16=',g2_16)
print('g3_16=',g3_16)       

#Calculation epsilon with g0, g1, g2, g3
x=x_g2_g3_16
e11_16=epsilon_model_nonlinear(x, t_vector ,Si_1D, landa, g0, g1, sigma_exp, t00, t0)            





# =============================================================================
# 20 MPa - Non_Linear Domain
# =============================================================================
#g0

#g1

#------Optimization
#------args----
t_vector=t_20MP
g0=g0_20
g1=g1_20
sigma_exp = sigma_20MP
e11_exp = e11_exp_20MP
t00=6.; t0=905. 
ft00=12; ft0=61; 
x_initial = [0.01,0.01]

#g2, g3 ------Optimization
optimization_g2_g3= least_squares(res_nonlinear_whole,x_initial, args = (t_vector ,Si_1D,landa, g0, g1, sigma_exp, t00, t0,ft00,ft0,ft3,ft4, e11_exp))
x_g2_g3_20=optimization_g2_g3.x
(g2_20,g3_20) = g2_g3_from_x(x_g2_g3_20)
print('-----------20 MPa------------') 
print('g0_20=',g0_20)
print('g1_20=',g1_20)  
print('g2_20=',g2_20)
print('g3_20=',g3_20)       

#Calculation epsilon with g0, g1, g2, g3
x=x_g2_g3_20 
e11_20=epsilon_model_nonlinear(x, t_vector ,Si_1D, landa, g0, g1, sigma_exp, t00, t0)            



# =============================================================================
# 22 MPa - Non_Linear Domain
# =============================================================================
#g0

#g1

#------Optimization
#------args----
t_vector=t_22MP
g0=g0_22
g1=g1_22
sigma_exp = sigma_22MP
e11_exp = e11_exp_22MP
t00=6.; t0=905. 
ft00=12; ft0=61; 
x_initial = [0.01,0.01]

#g2, g3 ------Optimization
optimization_g2_g3= least_squares(res_nonlinear_whole,x_initial, args = (t_vector ,Si_1D,landa, g0, g1, sigma_exp, t00, t0,ft00,ft0,ft3,ft4, e11_exp))
x_g2_g3_22=optimization_g2_g3.x
(g2_22, g3_22) = g2_g3_from_x(x_g2_g3_22)
print('-----------22 MPa------------') 
print('g0_22=',g0_22)
print('g1_22=',g1_22)
print('g2_22=',g2_22)
print('g3_22=',g3_22)       

#Calculation epsilon with g0, g1, g2, g3
x=x_g2_g3_22
e11_22=epsilon_model_nonlinear(x, t_vector ,Si_1D, landa, g0, g1, sigma_exp, t00, t0)            


# =============================================================================
# 25 MPa - Non_Linear Domain
# =============================================================================
#g0

#g1

#------Optimization
#------args----
t_vector=t_25MP
g0=g0_25
g1=g1_25
sigma_exp = sigma_25MP
e11_exp = e11_exp_25MP
t00=6.; t0=905. 
ft00=12; ft0=61; 
x_initial = [0.01,0.01]

#g2, g3 ------Optimization
optimization_g2_g3= least_squares(res_nonlinear_whole,x_initial, args = (t_vector ,Si_1D,landa, g0, g1, sigma_exp, t00, t0,ft00,ft0,ft3,ft4, e11_exp))
x_g2_g3_25=optimization_g2_g3.x
(g2_25, g3_25) = g2_g3_from_x(x_g2_g3_25)
print('-----------25 MPa------------') 
print('g0_25=',g0_25)
print('g1_25=',g1_25) 
print('g2_25=',g2_25)
print('g3_25=',g3_25)       

#Calculation epsilon with g0, g1, g2, g3
x=x_g2_g3_25
e11_25=epsilon_model_nonlinear(x, t_vector ,Si_1D, landa, g0, g1, sigma_exp, t00, t0)            



# =============================================================================
# 30 MPa - Non_Linear Domain
# =============================================================================
#g0

#g1

#------Optimization
#------args----
t_vector=t_30MP
g0=g0_30
g1=g1_30
sigma_exp = sigma_30MP
e11_exp = e11_exp_30MP
t00=6.; t0=905. 
ft00=12; ft0=61; 
x_initial = [0.01,0.01]

#g2, g3 ------Optimization
optimization_g2_g3= least_squares(res_nonlinear_whole,x_initial, args = (t_vector ,Si_1D,landa, g0, g1, sigma_exp, t00, t0,ft00,ft0,ft3,ft4, e11_exp))
x_g2_g3_30=optimization_g2_g3.x
(g2_30, g3_30) = g2_g3_from_x(x_g2_g3_30)
print('-----------30 MPa------------') 
print('g0_30=',g0_30)
print('g1_30=',g1_30)    
print('g2_30=',g2_30)
print('g3_30=',g3_30)       

#Calculation epsilon with g0, g1, g2, g3
x=x_g2_g3_30
e11_30=epsilon_model_nonlinear(x, t_vector ,Si_1D, landa, g0, g1, sigma_exp, t00, t0)            


# =============================================================================
# 35 MPa - Non_Linear Domain
# =============================================================================
#g0

#g1

#------Optimization
#------args----
t_vector=t_35MP
g0=g0_35
g1=g1_35
sigma_exp = sigma_35MP
e11_exp = e11_exp_35MP
t00=6.; t0=905. 
ft00=12; ft0=61; 
x_initial = [0.01,0.01]

#g2, g3 ------Optimization
optimization_g2_g3= least_squares(res_nonlinear_whole,x_initial, args = (t_vector ,Si_1D,landa, g0, g1, sigma_exp, t00, t0,ft00,ft0,ft3,ft4, e11_exp))
x_g2_g3_35=optimization_g2_g3.x
(g2_35, g3_35) = g2_g3_from_x(x_g2_g3_35)
print('-----------35 MPa------------') 
print('g0_35=',g0_35)
print('g1_35=',g1_35) 
print('g2_35=',g2_35)
print('g3_35=',g3_35)       

#Calculation epsilon with g0, g1, g2, g3
x=x_g2_g3_35
e11_35=epsilon_model_nonlinear(x, t_vector ,Si_1D, landa, g0, g1, sigma_exp, t00, t0)            



# =============================================================================
# Graph the results
# =============================================================================
figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k') 

plt.plot(t_13MP, e11_exp_13MP, 'r--.')
plt.plot(t_16MP, e11_exp_16MP, 'r--<')
plt.plot(t_20MP, e11_exp_20MP, 'r-->')
plt.plot(t_22MP, e11_exp_22MP, 'r--^')
plt.plot(t_25MP, e11_exp_25MP, 'r--8')
plt.plot(t_30MP, e11_exp_30MP, 'r--s')
plt.plot(t_35MP, e11_exp_35MP, 'r--p')


plt.plot(t_13MP, e11_13, 'b--')
plt.plot(t_16MP, e11_16, 'b--')
plt.plot(t_20MP, e11_20, 'b--')
plt.plot(t_22MP, e11_22, 'b--')
plt.plot(t_25MP, e11_25, 'b--')
plt.plot(t_30MP, e11_30, 'b--')
plt.plot(t_35MP, e11_35, 'b--')


plt.xlabel('time [s]')
plt.ylabel('E_11 [-]')
plt.title('non-linear viscoelastic (optimization problem, case II: 0 < t < 4000)')
plt.legend(('e11_exp_13MP', 'e11_exp_16MP', 'e11_exp_20MP','e11_exp_22MP',
            'e11_exp_25MP' , 'e11_exp_30MP' , 'e11_exp_35MP',
            
            'e11_model'),# 'e11_model_20', 'e11_model_22', 'e11_model_25', 'e11_model_30', 'e11_model_35'),
#            'e11_13', 'e11_20' , 'e11_22', 'e11_25', 'e11_30', 'e11_35'),
               shadow=True, loc=(0.5, 0.3), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([0, 4000, 0., 0.016])
plt.show()

print('ft3=',ft3)
print('ft4=',ft4 )



# =============================================================================
# g2 Graph
# =============================================================================
g2 = [g2_2, g2_5, g2_10, g2_13, g2_16, g2_20, g2_22, g2_25, g2_30, g2_35]
sigma_graph = [2, 5, 10, 13, 16, 20, 22, 25, 30, 35]

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
plt.plot(sigma_graph, g2, 'ro-')

plt.xlabel('sigma')
plt.ylabel('g2')
plt.title('g2')
plt.legend(('g2'),
               shadow=True, loc=(0.7, 0.4), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([2, 35, 0.5, 1.5])
plt.show()




# =============================================================================
# g3 Graph
# =============================================================================
g3 = [g3_2, g3_5, g3_10, g3_13, g3_16, g3_20, g3_22, g3_25, g3_30, g3_35]
sigma_graph = [2, 5, 10, 13,16,20,22,25,30,35]

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k') 
plt.plot(sigma_graph, g3, 'ro-')

plt.xlabel('sigma')
plt.ylabel('g3')
plt.title('g3')
plt.legend(('g3'),
               shadow=True, loc=(0.7, 0.4), handlelength=1.5, fontsize=12)
plt.grid(True)
plt.axis([2, 35, 0, 1.2])
plt.show()

