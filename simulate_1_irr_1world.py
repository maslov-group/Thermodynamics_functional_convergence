#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 18:46:56 2022

@author: ashish

Simulating just one world
We are simulating the effect of one irreversible reaction in the network
in these simulations, the energetics in each world is different. 
So for each world the maximum dissipation is goign to be differetn

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
parser.add_argument("-n", help="number of worlds", default=1)
#parser.add_argument("-a", help="analysis_options", default=None)
#    parser.add_argument("-v", help="param value", default=None)
args = parser.parse_args()
file_name='test.dat'
#analysis_options=args.a
if args.f is not None:
    file_name=args.f
if args.d is not None:
    data_folder=args.d

number_of_worlds=args.n

assert (data_folder), 'define data folder'
if not os.path.isdir(data_folder):os.mkdir(data_folder)

    

#number_of_pools = 10
#number_of_pools = 4 ## for S6_1irr_1
number_of_pools = 5 ## for S6_1irr_2

# Create and initialize pool
#n_sp = 3200
n_sp = 600


N_pool = np.ones(n_sp)

R_exp=np.array([80.,0.,0., 0., 0. ,0.]) ### could be dummy value and changed later.
#R_exp=np.array([80.,0.,0., 0.]) ### could be dummy value and changed later.

n_res=len(R_exp)





# =============================================================================
# ##for varying m as exp_conditions
# #m_expts_list=np.array([5.e-3, 1.e-2, 2.e-2, 5.e-2, 1.e-1, 2.e-1, 5.e-1, 1. ]) ## for dilution 1
# m_expts_list=np.array([0.625e-2, 1.25e-2, 2.5e-2, 5.e-2, 1.e-1, 2.e-1, 4.e-1, 8.e-1, 1.6]) ## for dilution 2,3
# exp_conditions_list=m_expts_list
# =============================================================================

# =============================================================================
# ##for varying m, R0 as exp_conditions while RO*m is held fixed
# #m_expts_list=np.array([0.3125e-2,0.625e-2,1.25e-2, 2.5e-2, 5.e-2, 1.e-1, 2.e-1, 4.e-1])  ## 
# #m_expts_list=np.array([0.625e-2,1.25e-2, 2.5e-2, 5.e-2, 1.e-1, 2.e-1, 4.e-1, 8.e-1])## for expstyle 1,2,3
# m_expts_list=np.array([0.25e-2, 0.5e-2, 1e-2, 2.e-2, 4.e-2, 8.e-2, 1.6e-1])## for expstyle 4,5
# R0_x_m=10
# exp_conditions_list=m_expts_list
# 
# =============================================================================



mean_KP=1. ## the mean is only approximate, it will be modified by choice of sigma.
mean_KS=1.
mean_kcat=1. 
#sigma_ks=.15 for S6_1irr_1
sigma_ks=.01 #for S6_1irr_2





params_pool = {'m': .01,'reaction_network_type': 'all'}

# =============================================================================
# n_res=4
# ##params_pool.update({'Energies':np.array([36,24,12,0]),'E_ATP':1 }) ### S4_1irr_1
# #irr_dG=10
# params_pool.update({'Energies':np.array([51,34,17,0]),'E_ATP':1 }) ### S4_1irr_2
# #irr_dG=15
# =============================================================================


n_res=6

# =============================================================================
# ### S6_1irr_1 and 2
# #params_pool.update({'Energies':np.array([60,48,36,24,12,0]),'E_ATP':1 }) 
# #irr_dG=10  ### S6_1irr_1 and 2
# =============================================================================
# =============================================================================
#  ### S6_1irr_3
# params_pool.update({'Energies':np.array([80,64,48,32,16,0]),'E_ATP':1 }) ### S6_1irr_3
# irr_dG=12 
# =============================================================================

## S6_1irr_4
params_pool.update({'Energies':np.array([100,80,60,40,20,0]),'E_ATP':1 }) ### S6_1irr_4
irr_dG=18 


n_reactions=int(n_res*(n_res-1)/2)


irr_reaction_list=(np.arange(n_reactions+1)-1).astype(int)
exp_conditions_list=irr_reaction_list


assumptions_init_world={'nu_constraint':'uniform_lognormal',
                        'growth_type': 'thermodynamic_inhibition'}
assumptions_init_world.update({'ATP_allocation':'quant_residue_rand'})





df_master = {'number_of_worlds':number_of_worlds,
        'number_of_pools': number_of_pools, 'number_of_experiments': len(
    exp_conditions_list), 'exp_conditions_list': exp_conditions_list}
print(exp_conditions_list)
start_time = time.time()


params_world_i=deepcopy(params_pool)

### using initialize parameters to create the reaction energetics to use in the world
_, _, params_with_energies, _ = initialize_parameters(
                [N_pool, R_exp], params_world_i, assumptions_init_world)

nATP_list=params_with_energies['N_ATP']
assumptions_init={'nu_constraint':'uniform_lognormal',
                  'growth_type': 'thermodynamic_inhibition',
                  'ATP_allocation':'nATP_list',
                  'nATP_list':nATP_list
                    }
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
        
        
        
        ### changing the nATP in the reaction of interest.
        assumptions_exp=deepcopy(assumptions_init)
        if exp_condition>=0:
            rxn_id=exp_condition
            DG_pool= params_pool_filled['DG']
            DE_pool= params_pool_filled['DEnergy']
            nATP_exp=assumptions_exp['nATP_list']
            assert params_pool_filled['E_ATP']==1,'otherwise need to scale'
            
            nATP_exp[exp_condition]=-DE_pool[exp_condition]- irr_dG
            assert -DE_pool[exp_condition] >irr_dG+1,'energy gap needs to be large enought to allow for reversibility' 
            
            assumptions_exp.update({'nATP_list':nATP_exp})
            params_init.update({'assumptions':assumptions_exp})
            params_init=create_reaction_network(n_res, params_init)
                        
    
# =============================================================================
#             ### when m varied between expts, tau also needs to be updated!
#             m=exp_condition
#             params_init.update({'m':m})
#             params_init.update({'tau':1./m})
#             
# =============================================================================
                    
        print ('pool ',pool_id, 'expt', expt_id,'  val ', exp_condition )
        print(params_init['assumptions'])
        print (params_init['DG'])
        

        def dNdt(N, R, params):
            return MakeConsumerDynamics(assumptions_exp)(N, R, params)
        def dRdt(N, R, params):
            return MakeResourceDynamics(assumptions_exp)(N, R, params)
        world_i=Species_with_Reactions([N_init, R_init], [dNdt, dRdt], params_init)

# =============================================================================
#             if expt_id == 0:
#     #             pool_df.update({'params': params_pool_filled})
#                 pool_df.update({'N_init': N_init})
#     #         df_exp_condition = {'R_supply': R_exp,'assumptions':assumptions_pool}
#     #        df_exp_condition = {'R_supply': exp_condition,
#     #                            'assumptions':assumptions_pool,
#     #                           'params':params_init}
# =============================================================================
        pool_df.update({'N_init': N_init})
        df_exp_condition = {'exp_variable':'m',
                            'exp_condition': exp_condition,
                            'assumptions':assumptions_exp,
                           'params':params_init}
        
# =============================================================================
# 
#         df_exp_condition = {'exp_variable':'m_R0m_fixed',
#                     'exp_condition': exp_condition,
#                     'assumptions':assumptions_pool,
#                    'params':world_i.params}
# =============================================================================
        
        

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
        

        
#        df_master.update({'world'+str(world_id)+'_pool'+str(pool_id): pool_df})
    df_master.update({'pool'+str(pool_id): pool_df})


#### output_file_name=file_name.replace('.dat', '-w'+str(world_id)+'.dat')  
output_file_name=file_name
with open(data_folder+output_file_name, 'wb') as f:
     pickle.dump(df_master, f, protocol=2)
create_analysis_df(data_folder,output_file_name)

end_time = time.time()
print('time for block', end_time-start_time,'secs') 
