# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 10:34:57 2018

@author: momoe
"""

from Tensors_Functions_S1_S2_S3_21082018 import *
import numpy as np
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter


df = pd.read_excel('04_01_donnees5MPa.xls', sheet_name='Sheet1')
E_5MPa = df['Déformation']
t_5MPa = df['Temps']

df = pd.read_excel('04_01_donnees10MPa.xls', sheet_name='Sheet1')
E_10MPa = df['Déformation']
t_10MPa = df['Temps']

df = pd.read_excel('04_01_donnees15MPa.xls', sheet_name='Sheet1')
E_15MPa = df['Déformation']
t_15MPa = df['Temps']

df = pd.read_excel('04_01_donnees20MPa.xls', sheet_name='Sheet1')
E_20MPa = df['Déformation']
t_20MPa = df['Temps']


time_frames_5=len(t_5MPa)
time_frames_10=len(t_10MPa)
time_frames_15=len(t_15MPa)
time_frames_20=len(t_20MPa)


# =============================================================================
# Figure - Strains For Stress (ideally Heaviside function)
# =============================================================================

fig = plt.figure(figsize=(20, 5))
ax = fig.add_subplot(1, 1, 1, aspect=6.5)


def minor_tick(x, pos):
    if not x % 1.0:
        return ""
    return "%.2f" % x

ax.xaxis.set_major_locator(MultipleLocator(10.000))
ax.xaxis.set_minor_locator(AutoMinorLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1.000))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.xaxis.set_minor_formatter(FuncFormatter(minor_tick))

ax.set_xlim(0, 200)
ax.set_ylim(0, 15)

ax.tick_params(which='major', width=1.0)
ax.tick_params(which='major', length=5)
ax.tick_params(which='minor', width=1.0, labelsize=10)
ax.tick_params(which='minor', length=5, labelsize=10, labelcolor='0.25')

ax.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)


ax.plot(t_5MPa, E_5MPa, 'r.-', label="E_5MPa [experimental]")
ax.plot(t_10MPa, E_10MPa, 'b.-', label="E_10MPa [experimental]")



ax.set_title("Creep-Recovery Test (5 and 10 MPa)", fontsize=20, verticalalignment='bottom')
ax.set_xlabel("Time [second]")
ax.set_ylabel("Axial Strain E [%]")

ax.legend()



color = 'red'
ax.annotate('5 MPa', xy=(60.0, 6.5), xycoords='data',
            xytext=(50, 8), textcoords='data',
            weight='bold', color=color,
            arrowprops=dict(arrowstyle='->',
                            connectionstyle="arc3",
                            color=color))


color = 'blue'
ax.annotate('10 MPa', xy=(100.0, 9), xycoords='data',
            xytext=(80, 9.5), textcoords='data',
            weight='bold', color=color,
            arrowprops=dict(arrowstyle='->',
                            connectionstyle="arc3",
                            color=color))


plt.show()



#------------------------------------------------------------------------------


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, aspect=0.4)


def minor_tick(x, pos):
    if not x % 1.0:
        return ""
    return "%.2f" % x

ax.xaxis.set_major_locator(MultipleLocator(10.000)) #change the X-Dir dimention
ax.xaxis.set_minor_locator(AutoMinorLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(20.000)) #change the Y-Dir dimention
ax.yaxis.set_minor_locator(AutoMinorLocator(1))
ax.xaxis.set_minor_formatter(FuncFormatter(minor_tick))

ax.set_xlim(0, 200)
ax.set_ylim(0, 250)

ax.tick_params(which='major', width=1.0)
ax.tick_params(which='major', length=5)
ax.tick_params(which='minor', width=1.0, labelsize=10)
ax.tick_params(which='minor', length=5, labelsize=10, labelcolor='0.25')

ax.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)


ax.plot(t_5MPa, E_15MPa, 'r.-', label="E_15MPa [experimental]")
ax.plot(t_10MPa, E_20MPa, 'b.-', label="E_20MPa [experimental]")



ax.set_title("Creep-Recovery Test (15 and 20 MPa)", fontsize=20, verticalalignment='bottom')
ax.set_xlabel("Time [second]")
ax.set_ylabel("Axial Strain E [%]")

ax.legend()


color = 'red'
ax.annotate('10 MPa', xy=(100.0, 6.5+20.), xycoords='data',
            xytext=(80., 8+20), textcoords='data',
            weight='bold', color=color,
            arrowprops=dict(arrowstyle='->',
                            connectionstyle="arc3",
                            color=color))

color = 'blue'
ax.annotate('20 MPa', xy=(100.0, 9.+50.), xycoords='data',
            xytext=(80., 9.5+50), textcoords='data',
            weight='bold', color=color,
            arrowprops=dict(arrowstyle='->',
                            connectionstyle="arc3",
                            color=color))

plt.show()

# =============================================================================
# 
# =============================================================================



fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, aspect=0.4)


def minor_tick(x, pos):
    if not x % 1.0:
        return ""
    return "%.2f" % x

ax.xaxis.set_major_locator(MultipleLocator(10.000)) #change the X-Dir dimention
ax.xaxis.set_minor_locator(AutoMinorLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(20.000)) #change the Y-Dir dimention
ax.yaxis.set_minor_locator(AutoMinorLocator(1))
ax.xaxis.set_minor_formatter(FuncFormatter(minor_tick))

ax.set_xlim(0, 200)
ax.set_ylim(0, 250)

ax.tick_params(which='major', width=1.0)
ax.tick_params(which='major', length=5)
ax.tick_params(which='minor', width=1.0, labelsize=10)
ax.tick_params(which='minor', length=5, labelsize=10, labelcolor='0.25')

ax.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)


ax.plot(t_5MPa, E_5MPa, 'g-', label="E_5MPa [experimental]")
ax.plot(t_10MPa, E_10MPa, 'k-', label="E_10MPa [experimental]")
ax.plot(t_5MPa, E_15MPa, 'r-', label="E_15MPa [experimental]")
ax.plot(t_10MPa, E_20MPa, 'b-', label="E_20MPa [experimental]")



ax.plot(t_5MPa, E_5MPa+E_5MPa, 'k--', label="E_5MPa+E_5MPa")





ax.set_title("Creep-Recovery Test", fontsize=20, verticalalignment='bottom')
ax.set_xlabel("Time [second]")
ax.set_ylabel("Axial Strain E [%]")

ax.legend()



color = 'black'
ax.annotate('E_5MPa+E_5MPa', xy=(90.0, 15), xycoords='data',
            xytext=(60., 8+20), textcoords='data',
            weight='bold', color=color,
            arrowprops=dict(arrowstyle='->',
                            connectionstyle="arc3",
                            color=color))




color = 'black'
ax.annotate('''It's linear''', xy=(130, 80), xycoords='data',
            xytext=(150, 100), textcoords='data',
            weight='bold', color=color,
            arrowprops=dict(arrowstyle='->',
                            connectionstyle="arc3",
                            color=color))


plt.show()



# =============================================================================
# 
# =============================================================================


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, aspect=0.4)


def minor_tick(x, pos):
    if not x % 1.0:
        return ""
    return "%.2f" % x

ax.xaxis.set_major_locator(MultipleLocator(10.000)) #change the X-Dir dimention
ax.xaxis.set_minor_locator(AutoMinorLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(20.000)) #change the Y-Dir dimention
ax.yaxis.set_minor_locator(AutoMinorLocator(1))
ax.xaxis.set_minor_formatter(FuncFormatter(minor_tick))

ax.set_xlim(0, 200)
ax.set_ylim(0, 250)

ax.tick_params(which='major', width=1.0)
ax.tick_params(which='major', length=5)
ax.tick_params(which='minor', width=1.0, labelsize=10)
ax.tick_params(which='minor', length=5, labelsize=10, labelcolor='0.25')

ax.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)


ax.plot(t_5MPa, E_5MPa, 'g-', label="E_5MPa [experimental]")
ax.plot(t_10MPa, E_10MPa, 'k-', label="E_10MPa [experimental]")
ax.plot(t_5MPa, E_15MPa, 'r-', label="E_15MPa [experimental]")
ax.plot(t_10MPa, E_20MPa, 'b-', label="E_20MPa [experimental]")



ax.plot(t_5MPa, E_5MPa+E_5MPa, 'k--', label="E_5MPa+E_5MPa")

ax.plot(t_5MPa, E_5MPa+E_10MPa, 'r--', label="E_5MPa+E_10MPa")

ax.plot(t_5MPa, E_10MPa+E_10MPa, 'b--', label="E_10MPa+E_10MPa")


ax.set_title("Creep-Recovery Test", fontsize=20, verticalalignment='bottom')
ax.set_xlabel("Time [second]")
ax.set_ylabel("Axial Strain E [%]")
ax.legend()


color = 'black'
ax.annotate('''After sigma=10 MPa it's not linear''', xy=(130, 80), xycoords='data',
            xytext=(120, 100), textcoords='data',
            weight='bold', color=color)

color = 'red'
ax.annotate('''E_15 != E_10 + E_5''', xy=(130, 80), xycoords='data',
            xytext=(120, 80), textcoords='data',
            weight='bold', color=color)

color = 'blue'
ax.annotate('''E_20 != E_10 + E_10''', xy=(130, 80), xycoords='data',
            xytext=(120, 60), textcoords='data',
            weight='bold', color=color)


plt.show()

