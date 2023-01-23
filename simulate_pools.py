#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 13:57:29 2021

@author: ashish
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
from scipy import stats
from copy import deepcopy
import sys
import matplotlib as mpl
import argparse
if os.getcwd().startswith("/Users/ashish"):
    sys.path.append('/Users/ashish/Documents/GitHub/metabolic-CRmodel/')
else:   
    sys.path.append('/home/a-m/ashishge/thermodynamic_model/code/')

   
from metabolic_CR_model_functions import *
from metabolic_CR_model_class import *
from thermo_analysis_functions import *

fontSize=12
fontSizeSmall=10
labelSize=8

machine_precision=1e-6

parser = argparse.ArgumentParser()
parser.add_argument("-f", help="file name", default=None) ## only a single folder handled for now!
parser.add_argument("-d", help="data folder", default='/Users/ashish/Downloads/temp_folder/')
#parser.add_argument("-a", help="analysis_options", default=None)
#    parser.add_argument("-v", help="param value", default=None)
args = parser.parse_args()
file_name='temp.dat'
#analysis_options=args.a
if args.f is not None:
    file_name=args.f
if args.d is not None:
    data_folder=args.d

assert (data_folder), 'define data folder'
if not os.path.isdir(data_folder):os.mkdir(data_folder)

    
#number_of_pools = 20
number_of_pools = 10
#number_of_pools = 4

# Create and initialize pool
#n_sp = 3200
n_sp = 600

##for Fig 5-d7 only!
#n_sp = 200
N_pool = np.ones(n_sp)

R_exp=np.array([80.,0.,0., 0., 0. ,0.]) ### could be dummy value and changed later.
params_pool = {'m': .01,'reaction_network_type': 'all'}
n_res=len(R_exp)

# =============================================================================
# ##for varying b as exp_conditions 
# b_strength_list=np.array([1.25e-3, 2.5e-3, 5.e-3, 1.e-2, 2.e-2, 4.e-2, 8.e-2, 1.6e-1, 3.2e-1])
# b_expts_list=np.outer(b_strength_list, np.random.lognormal(0,.01, n_res))
# =============================================================================

# =============================================================================
# ##for varying m as exp_conditions
# #m_expts_list=np.array([5.e-3, 1.e-2, 2.e-2, 5.e-2, 1.e-1, 2.e-1, 5.e-1, 1. ]) ## for dilution 1
# m_expts_list=np.array([0.625e-2, 1.25e-2, 2.5e-2, 5.e-2, 1.e-1, 2.e-1, 4.e-1, 8.e-1, 1.6]) ## for dilution 2,3
# exp_conditions_list=m_expts_list
# =============================================================================



##for varying m, R0 as exp_conditions while RO*m is held fixed
#m_expts_list=np.array([0.3125e-2,0.625e-2,1.25e-2, 2.5e-2, 5.e-2, 1.e-1, 2.e-1, 4.e-1])  ## 
#m_expts_list=np.array([0.625e-2,1.25e-2, 2.5e-2, 5.e-2, 1.e-1, 2.e-1, 4.e-1, 8.e-1])## for expstyle 1,2,3
#m_expts_list=np.array([0.25e-2, 0.5e-2, 1e-2, 2.e-2, 4.e-2, 8.e-2, 1.6e-1])## for expstyle 4,5

m_expts_list=np.array([0.25e-2, 0.5e-2, 1e-2, 2.e-2, 4.e-2, 8.e-2, 1.6e-1])## for expstyle 7
#R0_x_m=10 ## for exptyles 1-6 i think?
R0_x_m=.8 ## for exptyle style 7

exp_conditions_list=m_expts_list




# =============================================================================
# ######### for the buggy runs
# m_expts_list=np.array([0.125e-2, 0.25e-2, 0.5e-2,1.0e-2, 2e-2, 4e-2, 1.0e-1, 2.0e-1, 4.0e-1, 8.0e-1]) ## for ,6,7
# #m_expts_list=np.array([0.1e-2, 0.4e-2,1.6e-2, 6.4e-2, 2.5e-1, 1.0, 2., 4.0, 8.]) ## for 8 and S6_dilution_Flux_small_Constant5, 6
# #m_expts_list=np.array([1.56e-2,3.12e-2, 6.25e-2, 2.5e-1, 0.5, 1.0, 2.]) ## for 9,10
# #m_expts_list=np.array([3.75e-3, 7.5e-3, 1.5e-2, 3.12e-2, 6.25e-2,1.25e-1, 2.5e-1, 0.5, 1.0, 2.]) ## for 10-2
# #R0_x_m=2  ## for S6_dilution_Flux_small_Constant1-5
# R0_x_m=100  ## for S6_dilution_Flux_Constant
# =============================================================================




# =============================================================================
# ##for varying Rexp as exp_conditions
# # R_expts_list=[np.array([40.,0.,0 ,0.]),np.array([20.,0.,0 ,0.]),
# #              np.array([10.,0.,0 ,0.]),np.array([5.,0.,0 ,0.])]
# # R_expts_list=[np.array([80.,0.,0., 0., 0. ,0.]),np.array([40.,0.,0., 0., 0. ,0.]),
# #              np.array([20.,0.,0., 0., 0. ,0.]),np.array([10.,0.,0., 0., 0. ,0.])]
# 
# R_expts_list=[np.array([80.,0.,0., 0., 0. ,0.])]
# R_exp=R_expts_list[0]
# exp_conditions_list=R_expts_list
# =============================================================================






mean_KP=1. ## the mean is only approximate, it will be modified by choice of sigma.
mean_KS=1.
mean_kcat=1. 

# =============================================================================
# mean_KP=np.e ## For dilution_4  the mean is only approximate, it will be modified by choice of sigma.
# mean_KS=np.e 
# mean_kcat=np.e 
# =============================================================================


#sigma_ks=.15 ### usually was .01 till 2/4/2022, also used for  expStyle2,4, 5
#sigma_ks=.01 ### for bstrength simulations
sigma_ks=.01 ## for expStyle1 and expstyle 7
#sigma_ks=.4 ## for expStyle3

# =============================================================================
# #### for the runs that had the Rsupply bug which showed weird results.
# sigma_ks=.2 ####S6_dilution_Flux_Constant10, S6_dilution_Flux_Constant10-3
# sigma_ks=.4 ####S6_dilution_Flux_Constant3insp, S6_dilution_Flux_Constant10-2
# sigma_ks=.4 ### for S6_dilution_Flux_small_Constant1
# sigma_ks=.1 ### for S6_dilution_Flux_small_Constant2
# sigma_ks=1. ### for S6_dilution_Flux_small_Constant3
# sigma_ks=.01 ### for S6_dilution_Flux_small_Constant7,S6_dilution_Flux_Constant8,9
# sigma_ks=.45 ### for Figure 5 dilution3,5
# sigma_ks=1.5 ### for Figure 5 dlution 4
# sigma_ks=.7 ### for Figure 5
# =============================================================================




# params_pool.update({'Energies':np.array([15,12,9,6,3,0.])})
params_pool.update({'Energies':np.array([5,4,3,2,1,0.])}) #Expstyle1-4, and expt_style7
#params_pool.update({'Energies':np.array([5,4.2,3.7,2.4,1.4,0.])}) ### Expstle5
#params_pool.update({'Energies':np.array([7,4.2,3.7,3.2,1.4,0.])  }) ### Expstle6

# params_pool.update({'RT':2.5})
n_res=6
n_reactions=int(n_res*(n_res-1)/2)
#fATP_list=np.random.uniform(.15, .85,n_reactions) ## between .15 and.85 to be reasonable

# =============================================================================
# ### used for Fig.2 sims
# fATP_list=np.array([0.24879874, 0.57264421, 0.74799647, 0.67996101, 0.53403094,
#        0.60366031, 0.34189523, 0.80536414, 0.69045453, 0.75887878,
#        0.27632003, 0.27251061, 0.43838223, 0.15982021, 0.75559034]) ### random uniform between .15 and .85
# =============================================================================
    
# =============================================================================
# ## used for dilution2 
# fATP_list=np.array([0.68730415, 0.17320986, 0.41040171, 0.19615637, 0.33648042,
#        0.24646699, 0.16331297, 0.3298413 , 0.61473464, 0.31088368,
#        0.46497682, 0.80308221, 0.25018526, 0.49567051, 0.49742547])### random uniform between .15 and .85
#     
# =============================================================================
 ### used for varying m, keep R_x_m fixed dilution 3, like Fig5
fATP_list= np.array([0.55924781, 0.67994016, 0.48802272, 0.54706976, 0.34345378,
       0.45869557, 0.55469341, 0.42688083, 0.59636616, 0.48048182,
       0.4499956 , 0.18805555, 0.66911148, 0.47426608, 0.45927053])### random uniform between .15 and .85
    
    
assumptions_init={'nu_constraint':'uniform_lognormal',
                        'growth_type': 'thermodynamic_inhibition'}
assumptions_init.update({'ATP_allocation':'fATP_list',
                        'fATP_list':fATP_list})




df_master = {'number_of_pools': number_of_pools, 'number_of_experiments': len(
    exp_conditions_list), 'exp_conditions_list': exp_conditions_list}
print(exp_conditions_list)
start_time = time.time()
for pool_id in range(number_of_pools):
     
    propagation_time_step = 10
    pool_df = {}
    
    params_pool_i=deepcopy(params_pool)
    params_pool_i.update({'K_P': np.random.lognormal(mean=np.log(mean_KP), 
                                                     sigma=sigma_ks, size=n_reactions),
                         'K_S': np.random.lognormal(mean=np.log(mean_KS), 
                                                     sigma=sigma_ks, size=n_reactions),
                         'Kcat': np.random.lognormal(mean=np.log(mean_kcat), 
                                                     sigma=sigma_ks, size=n_reactions)})
    N_init, R_init, params_pool_filled, assumptions_pool = initialize_parameters(
                [N_pool, R_exp], params_pool_i, assumptions_init)
       
    for expt_id, exp_condition in enumerate(exp_conditions_list):
#         print (assumptions_pool)
        params_init=deepcopy(params_pool_filled)
        
         
# =============================================================================
#         ### when Rexp varied between expts
#         R_exp=exp_condition 
#         params_init.update({'R_supply':R_exp}) #### HAD A BUG. should have been R_supply! 6/22
#         N_init, R_init=[N_pool,R_exp]
# =============================================================================
        
# =============================================================================
#         ### when m varied between expts, tau also needs to be updated!
#         m=exp_condition
#         params_init.update({'m':m})
#         params_init.update({'tau':1./m})
# =============================================================================
        
        
        ### when m varied, R0*m is held fixed, tau also needs to be updated!
        m=exp_condition
        R0val=R0_x_m/m
        Rsupply=np.zeros(n_res)
        Rsupply[0]=R0val
        params_init.update({'m':m})
        params_init.update({'tau':1./m})
        params_init.update({'R_supply':Rsupply})
        N_init, R_init=[N_pool,R_exp]
        
# =============================================================================
#         print ('pool ',pool_id, 'expt', expt_id,'  val ', exp_condition )
#         print('m=', m, 'Rsupply=', params_init['R_supply'])
# =============================================================================
        
        

        def dNdt(N, R, params):
            return MakeConsumerDynamics(assumptions_init)(N, R, params)
        def dRdt(N, R, params):
            return MakeResourceDynamics(assumptions_init)(N, R, params)
        world_i=Species_with_Reactions([N_init, R_init], [dNdt, dRdt], params_init)

        if expt_id == 0:
#             pool_df.update({'params': params_pool_filled})
            pool_df.update({'N_init': N_init})
#         df_exp_condition = {'R_supply': R_exp,'assumptions':assumptions_pool}
#        df_exp_condition = {'R_supply': exp_condition,
#                            'assumptions':assumptions_pool,
#                           'params':params_init}
#        df_exp_condition = {'exp_variable':'m',
#                            'exp_condition': exp_condition,
#                            'assumptions':assumptions_pool,
#                           'params':params_init}
        

        df_exp_condition = {'exp_variable':'m_R0m_fixed',
                    'exp_condition': exp_condition,
                    'assumptions':assumptions_pool,
                   'params':world_i.params}
        
        

        # simulate expt till steady state.
        reached_SteadyState = False
        currentTime = 0
        Ncurrent = N_init
        Rcurrent = R_init
        ctr = 0

        while (reached_SteadyState == False):
            Ncurrent, Rcurrent = IntegrateDynamics(world_i,world_i.params, T0=currentTime,
                                                   T=currentTime + propagation_time_step,
                                                   ns=30, return_all=False)
            currentTime = currentTime + propagation_time_step
            if Ncurrent is None:
                print ('breaking')
                sys.exit(1)
                break
            Ncurrent[Ncurrent < world_i.params['popn_cutoff']] = 0.0
            check_SS = np.max(np.abs(compute_dlogNdt(Ncurrent, Rcurrent, world_i.params, dNdt)))
            world_i.update_state(Ncurrent,Rcurrent)
            
            if check_SS < 1e-5:## was 1e-3 till 3-4-21  ## was 1e-5 till 1-11-22
                reached_SteadyState = True
                if len(np.nonzero(Ncurrent)[0])>len(Rcurrent):
                    print ('competitive exclusion issue', Ncurrent[np.nonzero(Ncurrent)],Rcurrent)
                    print (compute_dlogNdt(Ncurrent, Rcurrent, world_i.params, dNdt)[np.nonzero(Ncurrent)])
                    reached_SteadyState = False ## continue running to see if sim will end.
                    
#                    sys.exit(1)
            elif ctr > 1000:
                print('too long.',exp_condition)
                print(Ncurrent)
                print(check_SS)
                sys.exit(1.)
            ctr += 1
        SS_df = {'SS_values': [Ncurrent, Rcurrent, currentTime]}
        df_exp_condition.update({'SS_values': [Ncurrent, Rcurrent, currentTime]})
        pool_df.update({'expt'+str(expt_id): df_exp_condition})
        
# =============================================================================
#         print('after sim')
#         print('pool ',pool_id, 'expt',expt_id )
#         print('simmed value=')
#         print(pool_df['expt'+str(expt_id)]['params']['R_supply'])
# =============================================================================

        

    df_master.update({'pool'+str(pool_id): pool_df})
    
    
        
with open(data_folder+file_name, 'wb') as f:
    pickle.dump(df_master, f, protocol=2)
    
create_analysis_df(data_folder,file_name)

end_time = time.time()
print('time for block', end_time-start_time,'secs') 
