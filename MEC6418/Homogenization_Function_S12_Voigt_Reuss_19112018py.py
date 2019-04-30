# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 12:26:11 2018

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
from numpy.linalg import inv

from Tensors_Functions_S1_S2_S3_21082018 import *
from Viscoelastic_Linear_Problem_LAB_Euler_NC_Rel_Flu_31102018 import * #(the difference is just for importing)
from Viscoelastic_Linear_Function_S5_01_09102018 import *
from Viscoelastic_Linear_Function_LAB_interconversion_25102018 import *
from Viscoelastic_Linear_Function_LAB_1D_interconversion_09112018 import *

from Homogenization_Function_S11_Eshelby_16112018py import *



def Voigt(k0,mu0,k1,mu1,c1):
    c0=1.-c1
    (I,J,K)=I_J_K_4orten_isotropes()
    J_V=tensor4_to_Voigt_tensor2(J)
    K_V=tensor4_to_Voigt_tensor2(K) 
    
    C_Voigt_V= 3.* (k1 + c0*(k0-k1)) *J_V + 2.*(mu1+c0*(mu0-mu1))*K_V
    return C_Voigt_V



def Reuss(k0,mu0,k1,mu1,c1):
    c0=1.-c1
    (I,J,K)=I_J_K_4orten_isotropes()
    J_V=tensor4_to_Voigt_tensor2(J)
    K_V=tensor4_to_Voigt_tensor2(K) 
    
    C_Reuss_V=( (3.*k0 *k1) / (k0 + c0*(k1-k0)) ) *J_V + ( (2.*mu0*mu1) / (mu0+c0*(mu1-mu0)) ) *K_V
    return C_Reuss_V





