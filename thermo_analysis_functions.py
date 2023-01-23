#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 17:54:15 2021

@author: ashish
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sys import exit
import sys
import pandas as pd
import glob
import pickle
from textwrap import wrap
import time
import itertools
from scipy import integrate
from scipy.spatial import distance
from copy import deepcopy
from scipy.special import comb
import warnings
if os.getcwd().startswith("/Users/ashish"):
    sys.path.append('/Users/ashish/Documents/GitHub/metabolic-CRmodel/')
else:   
    sys.path.append('/home/a-m/ashishge/thermodynamic_model/code/')

   
from metabolic_CR_model_functions import *
from metabolic_CR_model_class import *


def create_analysis_df(data_folder,file_name):
    analysis_file_name = 'analysis_'+file_name
    with open(data_folder+file_name, 'rb') as f:
        data_df = pickle.load(f)

    exp_conditions_list = data_df['exp_conditions_list']
    analysis_df = {'exp_conditions_list': exp_conditions_list}
    number_of_pools = data_df['number_of_pools']
    for expt_id, exp_condition in enumerate(exp_conditions_list):
        n_survivors_list = []
        reactions_in_final_community_list = []  # list of reactions in final community
        # list of reactions split by survivors as well
        reactions_in_final_survivors_list = []
        abundance_of_survivors_list = []
        idx_of_survivors_list = []
        enzyme_and_abundance_weighted_reactions_in_survivors_list = []
        enzyme_budget_allocated_to_reactions_in_survivors_list=[]
        fluxes_by_survivors_list=[]
        flux_in_community_list=[]
        Rss_list=[]
        params_list=[]
        for pool_id in range(number_of_pools):
            #         print(exp_condition,pool_id)
            current_df = data_df['pool'+str(pool_id)]['expt'+str(expt_id)]
            params_df=current_df['params']  
            
            params_list.append(params_df)
            Nf, Rf, Tf = current_df['SS_values']
            Nf = np.ravel(Nf)
            
            
# =============================================================================
#             print ('expt_id, pool id', expt_id, pool_id)
#             print (params_df['R_supply'])
#             print ('Rss sum=', np.sum(Rf))
# =============================================================================
            
            idx_survivors = np.ravel(np.nonzero(Nf))
            n_survivors_list.append(len(idx_survivors))
            Rss_list.append(Rf)
            n_reactions = params_df['n_reactions']
            reactions_in_final_community = np.zeros(n_reactions).astype(int)
            reactions_in_final_survivors = []
            abundance_of_survivors = []
            enzyme_and_abundance_weighted_reactions_in_survivors = []
            enzyme_budget_allocated_to_reactions_in_survivors = []
            fluxes_by_survivors=[]
            if len(idx_survivors) > 0:
                abundance_of_survivors = Nf[idx_survivors]
                if 'thermodynamic' in params_df['assumptions']['growth_type']:
                    nu_matrix = params_df['Enzyme_alloc']
                elif 'Michaelis-Menten' in params_df['assumptions']['growth_type']:
                    nu_matrix = params_df['Enzyme_alloc']
                elif 'product' in params_df['assumptions']['growth_type']:
                    nu_matrix = params_df['nu_max']
                for survivor in idx_survivors:
                    reactions_in_final_community[np.nonzero(
                        nu_matrix[survivor])] += 1
                    temp = np.zeros(n_reactions)
                    temp[np.nonzero(nu_matrix[survivor])] = 1
                    reactions_in_final_survivors.append(list(temp))
                    enzyme_and_abundance_weighted_reactions_in_survivors.append(
                        nu_matrix[survivor]*Nf[survivor])
                    enzyme_budget_allocated_to_reactions_in_survivors.append(
                        nu_matrix[survivor]/np.sum(nu_matrix[survivor]))
                def calc_fluxes(N,R,params):
                    return Make_flux_calculator(params_df['assumptions'])(N,R,params)
                fluxes_SS=calc_fluxes(Nf, Rf, params_df)
    #             tot_flux=np.sum(fluxes_SS,axis=0)
                for survivor in idx_survivors:          
                    fluxes_by_survivors.append(fluxes_SS[survivor]) 
                flux_community=np.sum(np.array(fluxes_by_survivors),axis=0)
            else:
                print ('no survivors')
                sys.exit(1)

            fluxes_by_survivors_list.append(fluxes_by_survivors)
            flux_in_community_list.append(flux_community)
    #         print('community',reactions_in_final_community)
            idx_of_survivors_list.append(idx_of_survivors_list)
            abundance_of_survivors_list.append(abundance_of_survivors)
            enzyme_and_abundance_weighted_reactions_in_survivors_list.append(
                enzyme_and_abundance_weighted_reactions_in_survivors)
            enzyme_budget_allocated_to_reactions_in_survivors_list.append(enzyme_budget_allocated_to_reactions_in_survivors)
            # list of reactions in final community
            reactions_in_final_community_list.append(reactions_in_final_community)
            reactions_in_final_survivors_list.append(reactions_in_final_survivors)

    #         print (exp_condition, pool_id)
    #         print ('idx survivors', idx_survivors)
    #         print ('survivor abundance',len(abundance_of_survivors), abundance_of_survivors)
    #         print ('survivor fluxes',np.shape(fluxes_by_survivors), fluxes_by_survivors)
    #         print ('reactions in survivors', np.shape(reactions_in_final_survivors), reactions_in_final_survivors)

        reaction_substrates=params_df['substrates']
        reaction_products=params_df['products']
        outcome_stats=statistics_of_outcomes(flux_in_community_list,reactions_in_final_community_list,
                              number_of_pools)  
        temp_df={'reactions_in_final_community_list': reactions_in_final_community_list,
                   'reactions_in_final_survivors_list': reactions_in_final_survivors_list,
                   'exp_condition': exp_condition,
                   'n_survivors_list': n_survivors_list,
                   'Rss_list':Rss_list,
                   'reaction_substrates': reaction_substrates, 'reaction_products': reaction_products,
                   'idx_of_survivors': idx_of_survivors_list, 'abundance_of_survivors': abundance_of_survivors_list,
                   'enzyme_and_abundance_weighted_reactions_in_survivors': enzyme_and_abundance_weighted_reactions_in_survivors_list,
                'enzyme_budget_allocated_to_reactions_in_survivors_list': enzyme_budget_allocated_to_reactions_in_survivors_list,
                 'fluxes_by_survivors_list':fluxes_by_survivors_list,'flux_in_community_list':flux_in_community_list,
                 'sim_params':params_list
                }
        temp_df.update(outcome_stats)
        n_resources=len(Rf)
        n_reactions=len(reaction_substrates)
        analysis_df.update({'expt'+str(expt_id): temp_df})

    analysis_df.update({'reaction_substrates': reaction_substrates,
                        'reaction_products': reaction_products,
                        'n_resources': n_resources, 'n_reactions': n_reactions,
                       'number_of_pools':number_of_pools})
    with open(data_folder+analysis_file_name, 'wb') as f:
        pickle.dump(analysis_df, f, protocol=2)
        
        
def statistics_of_outcomes(flux_in_community_list,reactions_in_final_community_list,
                              number_of_pools):
    flx_cutoff=1e-3
    flux_in_community_arr=np.array(flux_in_community_list)
    flux_in_community_arr[flux_in_community_arr<flx_cutoff]=0.       
    functional_distance_arr=distance.pdist(flux_in_community_arr,metric='jensenshannon')   
    nonzero_flux_counter=np.sum(flux_in_community_arr>flx_cutoff,axis=0)    
    outcome_stats={'functional_distance_arr':functional_distance_arr,
                  'nonzero_flux_counter':nonzero_flux_counter}
    return outcome_stats