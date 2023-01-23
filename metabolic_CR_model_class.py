#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 16:54:29 2021

@author: ashish

defines the class
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sys import exit
import pandas as pd
import glob
import pickle
from textwrap import wrap
import time
import itertools
from scipy import integrate
from copy import deepcopy


machine_precision_cutoff=1e-10
class Species_with_Reactions:
    def __init__(self, init_state, dynamics_functions, params):
        """
        Initialize all parameters in a species pool, initial conditions, set of reactions,
        functions to generate dynamics
        
        reactions_def=[n_reactions, Substrates, Products, Sub_to_res, Pro_to_res]  where Substrates, Products 
            define the substrates and products of each reaction. Sub_to_res and Pro_to_res are 2d arrays to map
            substrates and products to resources respectively
        
        init_state = [N0,R0] where N0 and R0 specify initial consumer and resource concentrations,
            and specify size of pool
        
        dynamics_functions = [dNdt,dRdt] where dNdt(N,R,params) and dRdt(N,R,params) are 
            vectorized functions of the consumer and resource dynamics

            
        params is a Python dictionary containing the parameters that required by these functions, and is passed to 
            the new plate instance in the next argument. 

        """

        N, R = init_state
        self.n_res = len(R)  # number of resources
        self.n_sp = len(N)  # number of species                
        self.N = N.copy()
        self.R = R.copy()
#        if np.any(self.R<machine_precision_cutoff): print('R cannot be zero exactly, but Rsupply can')
        self.R=np.clip(self.R,machine_precision_cutoff,None) #to prevent division by zero
        
        self.n_reactions=params['n_reactions']
#        n_reactions, Substrates, Products, Sub_to_res, Pro_to_res=reactions_def     
#        self.Substrates=Substrates.copy()
#        self.Products=Products.copy()
#        self.Sub_to_res=Sub_to_res.copy()
#        self.Pro_to_res=Pro_to_res.copy()

        self.dNdt, self.dRdt =  dynamics_functions
        
        self.params = params.copy()
        
#        assert self.params['tau']==1./self.params['m'],  print('tau and 1/m not equal',self.params['tau'], 1./self.params['m'] )
       
        
        
    def dydt(self,y,t,params_comp,S_comp):
            """
            Combine N and R into a single vector with a single dynamical equation
            
            y = [N1,N2,N3...NS,R1,R2,R3...RM]
            
            t = time
            
            params = params to pass to dNdt,dRdt
            
            S_comp = number of species in compressed consumer vector
                (with extinct species removed)
            """
            return np.hstack([self.dNdt(y[:S_comp],y[S_comp:],params_comp),
                              self.dRdt(y[:S_comp],y[S_comp:],params_comp)])     

    def update_state(self,N_new,R_new):
        self.N=N_new
        self.R=R_new
        self.R=np.clip(self.R,machine_precision_cutoff,None) #to prevent division by zero

    

# =============================================================================
#     def add_species(self, N_inv, params_inv, n_added=1):
#         '''
#         isn't updated to include recently introduced parameters
#         '''
#         
#         
#         params_new = deepcopy(self.params)
#         N_new = np.append(self.N, N_inv)
#         n_sp_new = self.n_sp+n_added
#         if 'nu_max' in params_new.keys():
#             if n_added == 1 and params_new['nu_max'].ndim == 1:
#                 params_new['nu_max'] = np.append(
#                     self.params['nu_max'], params_inv['nu_max'][:, np.newaxis], axis=1)
#             else:
#                 params_new['nu_max'] = np.append(
#                     self.params['nu_max'], params_inv['nu_max'], axis=1)
#         if 'Km' in params_new.keys():
#             if n_added == 1 and params_new['Km'].ndim == 1:
#                 params_new['Km'] = np.append(
#                     self.params['Km'], params_inv['Km'][:, np.newaxis], axis=1)
#             else:
#                 params_new['Km'] = np.append(
#                     self.params['Km'], params_inv['Km'], axis=1)
#         if 'Keq' in params_new.keys():
#             if n_added == 1 and params_new['Keq'].ndim == 1:
#                 params_new['Keq'] = np.append(
#                     self.params['Keq'], params_inv['Keq'][:, np.newaxis], axis=1)
#             else:
#                 params_new['Keq'] = np.append(
#                     self.params['Keq'], params_inv['Keq'], axis=1)
#         if 'rho' in params_new.keys():
#             if type(params_new['rho']) == np.ndarray:
#                 if n_added == 1 and params_new['rho'].ndim == 1:
#                     params_new['rho'] = np.append(
#                         self.params['rho'], params_inv['rho'][:, np.newaxis], axis=1)
#                 else:
#                     params_new['rho'] = np.append(
#                         self.params['rho'], params_inv['rho'], axis=1)
#             else:
#                 assert params_new['rho'] == params_inv['rho']
#         for name in ['m', 'g']:
#             if name in params_new.keys():
#                 if type(params_new[name]) == np.ndarray:
#                     params_new[name] = np.append(self.params[name], params_inv[name])
#                 else:
#                     # all species have same m, g
#                     assert params_new[name] == params_inv[name]
#                     
#         self.N=N_new
#         self.params=params_new
#         self.n_sp =n_sp_new   
#  
# =============================================================================
