#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 14:40:23 2021

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
import pickle

if os.getcwd().startswith("/Users/ashish"):
    sys.path.append('/Users/ashish/Documents/GitHub/metabolic-CRmodel/')
else:   
    sys.path.append('/home/a-m/ashishge/thermodynamic_model/code/')
   
from metabolic_CR_model_functions import *
from metabolic_CR_model_class import *







parser = argparse.ArgumentParser()
parser.add_argument("-f", help="file name", default=None) ## only a single folder handled for now!
parser.add_argument("-d", help="data folderf", default=None)
#parser.add_argument("-a", help="analysis_options", default=None)
#    parser.add_argument("-v", help="param value", default=None)
args = parser.parse_args()

#analysis_options=args.a
if args.f is not None:
    file_name=args.f
if args.d is not None:
    data_folder=args.d



#file_name = 'sim_rxn_network4_bigE_uniLogNorm_ATP_fp7.dat'
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
    for pool_id in range(number_of_pools):
        #         print(exp_condition,pool_id)
        current_df = data_df['pool'+str(pool_id)]['expt'+str(expt_id)]
        Nf, Rf, Tf = current_df['SS_values']
        Nf = np.ravel(Nf)
        idx_survivors = np.ravel(np.nonzero(Nf))
        n_survivors_list.append(len(idx_survivors))
        n_reactions = data_df['pool'+str(pool_id)]['params']['n_reactions']
        reactions_in_final_community = np.zeros(n_reactions).astype(int)
        reactions_in_final_survivors = []
        abundance_of_survivors = []
        enzyme_and_abundance_weighted_reactions_in_survivors = []
        enzyme_budget_allocated_to_reactions_in_survivors = []
        if len(idx_survivors) > 0:
            abundance_of_survivors = Nf[idx_survivors]
            if 'thermodynamic' in data_df['pool'+str(pool_id)]['params']['assumptions']['growth_type']:
                nu_matrix = data_df['pool' +
                                    str(pool_id)]['params']['Enzyme_alloc']
            elif 'Michaelis-Menten' in data_df['pool'+str(pool_id)]['params']['assumptions']['growth_type']:
                nu_matrix = data_df['pool' +
                                    str(pool_id)]['params']['Enzyme_alloc']
            elif 'product' in data_df['pool'+str(pool_id)]['params']['assumptions']['growth_type']:
                nu_matrix = data_df['pool'+str(pool_id)]['params']['nu_max']
            for survivor in idx_survivors:
                reactions_in_final_community[np.nonzero(
                    nu_matrix[survivor])] += 1
                temp = np.zeros(n_reactions)
                temp[np.nonzero(nu_matrix[survivor])] = 1
                reactions_in_final_survivors.append(temp)
                enzyme_and_abundance_weighted_reactions_in_survivors.append(
                    nu_matrix[survivor]*Nf[survivor])
                enzyme_budget_allocated_to_reactions_in_survivors.append(
                    nu_matrix[survivor]/np.sum(nu_matrix[survivor]))
#         print('community',reactions_in_final_community)
        idx_of_survivors_list.append(idx_of_survivors_list)
        abundance_of_survivors_list.append(abundance_of_survivors)
        enzyme_and_abundance_weighted_reactions_in_survivors_list.append(
            enzyme_and_abundance_weighted_reactions_in_survivors)
        enzyme_budget_allocated_to_reactions_in_survivors_list.append(enzyme_budget_allocated_to_reactions_in_survivors)
        # list of reactions in final community
        reactions_in_final_community_list.append(reactions_in_final_community)
        reactions_in_final_survivors_list.append(reactions_in_final_survivors)
    reaction_substrates=data_df['pool'+str(pool_id)]['params']['substrates']
    reaction_products=data_df['pool'+str(pool_id)]['params']['products']
    temp_df={'reactions_in_final_community_list': reactions_in_final_community_list,
               'reactions_in_final_survivors_list': reactions_in_final_survivors_list,
               'R_supply': exp_condition[0], 'growth_type': exp_condition[1],
               'n_survivors_list': n_survivors_list,
               'reaction_substrates': reaction_substrates, 'reaction_products': reaction_products,
               'idx_of_survivors': idx_of_survivors_list, 'abundance_of_survivors': abundance_of_survivors_list,
               'enzyme_and_abundance_weighted_reactions_in_survivors': enzyme_and_abundance_weighted_reactions_in_survivors_list,
            'enzyme_budget_allocated_to_reactions_in_survivors_list': enzyme_budget_allocated_to_reactions_in_survivors_list
            }
    n_resources=len(Rf)
    n_reactions=len(reaction_substrates)
    analysis_df.update({'expt'+str(expt_id): temp_df})

analysis_df.update({'reaction_substrates': reaction_substrates,
                    'reaction_products': reaction_products,
                    'n_resources': n_resources, 'n_reactions': n_reactions,
                    'number_of_pools':number_of_pools})
with open(data_folder+analysis_file_name, 'wb') as f:
    pickle.dump(analysis_df, f, protocol=2)