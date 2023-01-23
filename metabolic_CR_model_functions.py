# -*- coding: utf-8 -*-
"""
This file contains code that defines the metabolic CR model and its time dynamics
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
from copy import deepcopy
from scipy.special import comb
import warnings

def rms(y):
    return np.sqrt(np.mean(y**2))

def check_steady_state(N,R,params,dNdt, threshold=1e-4):
    
    SS_deviation=np.max(np.abs(compute_dlogNdt(N, R, params, dNdt)))
    
    max_invader_growth=-1.
    at_SS=False
    if SS_deviation< threshold:   
        at_SS=True
        N_extinct=np.ones(len(N))
        N_extinct[N>params['popn_cutoff']]=0.    
        max_invader_growth=np.max(compute_dlogNdt(N_extinct, R, params, dNdt))
    
    
    if max_invader_growth >0:
        print ('extinct species could have invaded', max_invader_growth)
        invader_growth=compute_dlogNdt(N_extinct, R, params, dNdt)
        print ('n invaders, growth rates', np.sum(invader_growth>0),invader_growth )
    
    return at_SS,SS_deviation,  max_invader_growth
    
    
def compute_dlogNdt(N,R,params,dNdt):
    '''
    computes dlogNdt, which is used to check steady state.
    '''
    N_survived=np.zeros(len(N))
    N_survived[N>params['popn_cutoff']]=1.
    dlogNdt_arr=dNdt(N_survived,R,params)
    return dlogNdt_arr



    
    
    
    
    


def BinaryMatrix_min1resource(a, b, p):
    """
    Construct binary random matrix with atleast one nonzero element per row
    p = probability that element equals 1 (otherwise 0)
    """
    assert p >= 1./b
    condition=False
    ctr=0
    while condition == False:
        r=np.random.rand(a, b)
        m=np.zeros((a, b))
        m[r < p]=1.0
        ctr += 1
        if np.all(np.sum(m, axis=1) > 0.5):
            condition=True
        if ctr >= 10:
            sys.exit(1)
    return m

def create_nu_matrix(nsp,nreact, nu_constraint='mean_lognormal', params=None, mu_tot=1., sd=.1):
    """
    Construct a matrix of numax where total nu of a species is  constrained in various ways.
    
    if nu_constraint='mean_lognormal' 
        then soft constraint on total nu.
        each species has 1 to nreactions from a uniform distribution
        nu_max for each reaction is distributed lognormally with mean=mu_tot/nreactions, sd=1    
    if  nu_constraint='sum_lognormal'
        then hard constraint on total nu.
        each species has 1 to nreactions from a uniform distribution nu_max for each reaction is chosen from a lognormal with mean=mu_tot, sd=1 
        and then renormalized to sum to 1.
    if  nu_constraint='mean_sd_lognormal'
        theb both mean and sd of the lognromal distribution (for nu_max) is scaled by the number of nonzero reactions 
        
    if  nu_constraint='uniform_lognormal'
        then total enzyme budget is lognormally distributed. Each nonzero reaction is assigned from the budget via a uniform distribution
    if  nu_constraint='linear_enzyme_cost'
        then same as uniform_lognormal except that the amount of enzyme in that reaction is = budget assigned/cost per enzyme.
    """
#    print('mean is  e^mu',np.exp(mu_tot))
    
    
    if 'nu-mu_tot' in params:
        mu_tot=params['nu-mu_tot']
    if 'nu-sd'in params:
        sd=params['nu-sd']
    
    
    M=np.zeros((nsp, nreact))
    reactions=np.arange(nreact,dtype=int)   
    if nu_constraint=='mean_lognormal':
        for i in range(nsp):
            sparsity=np.random.choice(reactions)+1
            reactions_chosen=np.random.choice(reactions, size=sparsity, replace=False)
            M[i,reactions_chosen]=np.random.lognormal(mu_tot/sparsity, sd, sparsity)
    elif nu_constraint=='sum_lognormal':     
        print('ned wingreen-style degeneracy!')
        for i in range(nsp):
            degenerate_row=True
            while degenerate_row==True:
                M[i]=np.zeros(nreact)
                sparsity=np.random.choice(reactions)+1
                reactions_chosen=np.random.choice(reactions, size=sparsity, replace=False)           
                temp=np.random.lognormal(mu_tot, sd, sparsity)
                M[i,reactions_chosen]=temp*10./np.sum(temp)
                degenerate_row=False
                for j in range (i):
                    if np.allclose(M[i],M[j], rtol=1e-2): ## we don't want them to be superclose to each other
                        degenerate_row=True
                        break
    elif nu_constraint=='mean_sd_lognormal':       
        for i in range(nsp):
            sparsity=np.random.choice(reactions)+1
            reactions_chosen=np.random.choice(reactions, size=sparsity, replace=False)
            M[i,reactions_chosen]=np.random.lognormal(mu_tot/sparsity, sd/sparsity, sparsity)
    
    elif nu_constraint=='uniform_lognormal' or nu_constraint=='linear_enzyme_cost':       

        for i in range(nsp):            
            sum_c=np.random.lognormal(mu_tot, sd)
            sparsity=np.random.choice(reactions)+1
            reactions_chosen=np.random.choice(reactions, size=sparsity, replace=False)    
            if nu_constraint=='uniform_lognormal':
                M[i,reactions_chosen]=np.random.dirichlet(np.ones(sparsity))*sum_c ## alpha=1, so partitions are from uniform distribution                
            elif nu_constraint=='linear_enzyme_cost':
                budgets_assigned=np.random.dirichlet(np.ones(sparsity))*sum_c
                enz_costs=np.asarray(params['Enzyme_costs'])[reactions_chosen]
                M[i,reactions_chosen]=budgets_assigned/enz_costs
                
    return M



    
def create_reaction_network(n_res, params_in):
    '''
    creates the reaction_network of type specified by reaction_network_type and assigns delta energies to each reaction
    
    reaction_network_type:
        'linear' means resources are arranged in a linear chain. R0->R1->R2->R3.. no alternate routes, such as R0->R2 are allowed
        'all' means resources are fully connected. '
        'user-defined' means its provided by the user.
    The energy hierarchy, without loss of generality is still R0>R1>R2...
    '''
    
#    params=params_in.copy()
    params=deepcopy(params_in)
    
    if  params['reaction_network_type'] == 'user_defined':
#        print('reaction network is user defined')
        return params
    
    if 'E_ATP' not in params: params['E_ATP']=1 ### some value , could be made in KJ/mol....
    if 'Enzyme_cost_factor' not in params: params['Enzyme_cost_factor']=0.1 ### deprecated parameter...
    def assign_energies_to_resources(n_res, params):
        if 'Energies' in params: ## energies are user defined
            if 'energy_dist' not in params: params['energy_dist']='user_defined'
            return params
        else:
            if 'energy_dist' not in params: params['energy_dist']='uniform_gaps'
            
            if params['energy_dist']=='uniform_gaps':
                energy_gaps=np.random.uniform(low=params['E_ATP'], high= 10.*params['E_ATP'], size=n_res)
                params['Energies']=np.cumsum(energy_gaps)[::-1]# descending  order depending since lower energies is favorable direction           
            return params
    
    def allocate_ATP(Denergy,params,rxn_idx):# allocate n_ATPs produced in each reaction
        
        max_ATP=-Denergy/params['E_ATP']
        
        if isinstance(params['assumptions']['ATP_allocation'],str):
        
            if params['assumptions']['ATP_allocation'] =='random':
                N_ATP= np.random.uniform(0.,max_ATP)  ## atelast one ATP is produced        
            elif params['assumptions']['ATP_allocation'] =='maximum': ## deprecated
                N_ATP=1.*int(-Denergy/params['E_ATP'])
                print ('deprecated')
            elif 'f_' in params['assumptions']['ATP_allocation']:
                frac=float(params['assumptions']['ATP_allocation'].replace('f_',''))
                N_ATP=frac*max_ATP
                
            elif 'c_' in params['assumptions']['ATP_allocation']:
                const=float(params['assumptions']['ATP_allocation'].replace('c_',''))
                N_ATP=const
                assert N_ATP <max_ATP ,'reaction needs to be feasible'
                
            elif 'fATP_list' in params['assumptions']['ATP_allocation']:
                fATP_list=params['assumptions']['fATP_list']
                N_ATP=fATP_list[rxn_idx]*max_ATP
                
            elif 'nATP_list' in params['assumptions']['ATP_allocation']: ## number of ATP in reaction is provided
                nATP_list=params['assumptions']['nATP_list']
                N_ATP=nATP_list[rxn_idx]
            
            elif 'dEATP_list' in params['assumptions']['ATP_allocation']: ## number of ATP in reaction is provided
                dE_ATP_list=params['assumptions']['dEATP_list']
                N_ATP=dE_ATP_list[rxn_idx]/params['E_ATP']
                
            elif 'quant_residue_rand' in params['assumptions']['ATP_allocation']:
                ### picks a random number between 0 and 5RT as the residual energy dissipated (since max quantas are squeezed out)
                assert -Denergy>6,'this is assuming energy gaps between reactions are larger'
                E_Diss=np.random.random()*5
                N_ATP= (-Denergy-E_Diss)/params['E_ATP']
            
                
            else:   
                print ('invalid ATP allocation')
                sys.exit(1)
                
        else:## constant, deprecated
            N_ATP=float(params['assumptions']['ATP_allocation'])
    
        return N_ATP
    
    params=assign_energies_to_resources(n_res, params) ## assigns energies and energy distribution
    Energies=params['Energies']
    assert len(Energies)==n_res
    
    
    if  params['reaction_network_type'] == 'linear':    
        assert params['stoichiometry_provided']==False,'stoich not implemented in delta G calc yet'
        
        n_reactions = n_res-1 # number of reactions if only single step reactions are indexed
        # assign S and P to each reaction
        Substrates = np.arange(n_reactions).astype(int)
        Products = np.arange(n_reactions).astype(int)+1
        Sub_to_res = np.zeros((n_reactions, n_res)).astype(int)
        Pro_to_res = np.zeros((n_reactions, n_res)).astype(int)     
        DEnergies = []
        N_ATPs=[]
        DGs=[]
        Enzyme_costs = []
        for i in range(n_reactions):
            Sub_to_res[i, Substrates[i]] = 1
            Pro_to_res[i, Products[i]] = 1
            
            ### calculate energetics of the reaction
            Denergyi=Energies[Products[i]]-Energies[Substrates[i]]
            assert Denergyi<0,'energy needs to be less than 0'
            DEnergies.append(Denergyi)   
            Enzyme_costs.append(-params['Enzyme_cost_factor']*Denergyi)
            
            N_ATPi=allocate_ATP(Denergyi, params, i)            
            N_ATPs.append(N_ATPi)
            DG_i=Denergyi+N_ATPi*params['E_ATP']## driving force left
            assert DG_i<0, 'too much atp produced?'
            DGs.append(DG_i)
            
        
        params.update({'DEnergy': np.array(DEnergies), 'DG':np.array(DGs),'N_ATP':np.array(N_ATPs), 
                       'Energy_yield': np.array(N_ATPs)*params['E_ATP'],
                       'Enzyme_costs':Enzyme_costs})
        params.update({'n_reactions':n_reactions,
                       'substrates': Substrates, 'products': Products, 
                       'Sub_to_res': Sub_to_res, 'Pro_to_res': Pro_to_res})
        return params 
            
    elif  params['reaction_network_type'] == 'all':
        assert params['stoichiometry_provided']==False,'stoich not implemented in delta G calc yet'
        n_reactions = int(comb(n_res,2))
        Substrates=[]
        Products=[]
        Sub_to_res = np.zeros((n_reactions, n_res)).astype(int)
        Pro_to_res = np.zeros((n_reactions, n_res)).astype(int)   
        DEnergies=[]
        N_ATPs=[]
        DGs=[]
        Enzyme_costs = []
        idx=0
        for reaction in itertools.permutations(np.arange(n_res).astype(int),2):
            if reaction[0]<reaction[1]:
                Substrates.append(int(reaction[0]))
                Products.append(int(reaction[1]))
                Sub_to_res[idx, Substrates[idx]] = 1
                Pro_to_res[idx, Products[idx]] = 1               
                ### calculate energetics of the reaction
                Denergyi=Energies[Products[idx]]-Energies[Substrates[idx]]
                assert Denergyi<0,'energy needs to be less than 0'
                DEnergies.append(Denergyi) 
                Enzyme_costs.append(-params['Enzyme_cost_factor']*Denergyi)
                N_ATPi=allocate_ATP(Denergyi, params, idx)    
                N_ATPs.append(N_ATPi)
                DG_i=Denergyi+N_ATPi*params['E_ATP'] ## driving force left
                assert DG_i<0, 'too much atp produced?'
                DGs.append(DG_i)
                idx+=1
                
        Substrates=np.array(Substrates)     
        Products=np.array(Products)
        assert len(Substrates)==n_reactions,'number of reactions is incorrect'
        params.update({'DEnergy': np.array(DEnergies), 'DG':np.array(DGs),'N_ATP':np.array(N_ATPs), 
                       'Energy_yield': np.array(N_ATPs)*params['E_ATP'],
                       'Enzyme_costs':Enzyme_costs })
        params.update({'n_reactions':n_reactions,
                       'substrates': Substrates, 'products': Products, 
                       'Sub_to_res': Sub_to_res, 'Pro_to_res': Pro_to_res})
        
        return params 
    
    else:   
        print ('invalid reaction network type',params['reaction_network_type'])
        return None
        

def initialize_parameters(init_state, params_input, assumptions={}):
        """
        Initialize a new inoculant in a specified environment
        Initialize set of reactions in the model
        init_state = [N0,R0] where N0 and R0 are 1-D arrays specifying the consumer and resource concentrations.
        """
        params=params_input.copy()
        N0, R0 = init_state
        N = N0.copy()
        R = R0.copy()
        R_supply = R0.copy()
        
        
        n_res = len(R0)  # number of resources
        n_sp = len(N0)  # number of species
# =============================================================================
#         n_reactions = n_res-1 # number of reactions if only single step reactions are indexed
#         # assign S and P to each reaction
#         Substrates = np.arange(n_reactions).astype(int)
#         Products = np.arange(n_reactions).astype(int)+1
#         '''
#         if multistep reactions are distinct:
#         # for reaction in itertools.permutations(np.arange(n_res),2):
#         #     if reaction[0]<reaction[1]:
#         #         Substrates.append(reaction[0])
#         #         Products.append(reaction[1])
#         '''
#         Sub_to_res = np.zeros((n_reactions, n_res)).astype(int)
#         Pro_to_res = np.zeros((n_reactions, n_res)).astype(int)
#         for i in range(n_reactions):
#             Sub_to_res[i, Substrates[i]] = 1
#             Pro_to_res[i, Products[i]] = 1
# =============================================================================
        
        
        if 'growth_type' not in assumptions: assumptions.update({'growth_type': 'thermodynamic_inhibition'})       
        if 'supply' not in assumptions: assumptions.update({'supply': 'external'})
        if 'ATP_allocation' not in assumptions: assumptions.update({'ATP_allocation': 'random'})
        if 'nu_constraint' not in assumptions: assumptions.update({'nu_constraint': 'mean_lognormal'})
        params.update({'assumptions':assumptions})
        
        if 'reaction_network_type' not in params: params.update({'reaction_network_type': 'linear'})
        if 'stoichiometry_provided' not in params: params.update({'stoichiometry_provided':False})
        params=create_reaction_network(n_res, params)
        n_reactions=params['n_reactions']
        ### assign default values.
        if 'g' not in params: params.update({'g': 1.})
        ###9-13-21 set m and tau to be same.
        if 'm' not in params: params.update({'m': 1.}) ## was 0.1 before 9-13-21
        if 'tau' not in params: params.update({'tau': 1./params['m']}) ## BUG!! we have tau=1./m, otherwise tau may not be updated
        ## if para
        if 'rho' not in params: params.update({'rho': 1.})
        if 'yield' not in params: params.update({'yield': 1.})
        if 'popn_cutoff' not in params: params.update({'popn_cutoff':1e-3})
       
        params.update({'R_supply': R_supply})
        
        assert 'Rsupply' not in params,' its called R_supply not Rsupply'
        
        if 'anabolism' not in assumptions['growth_type']:
            params.update({'tau': 1./params['m']}) ### tau is always 1./m in this model.

        

        if 'thermodynamic' in assumptions['growth_type'] or 'anabolism' in assumptions['growth_type']:
            
            for pname in ['rho_rev', 'K_P', 'K_S', 'Kcat']:
                if pname in params:
                    if isinstance(params[pname], (int, float)): ### they should actually be an array
                        params[pname]=np.ones(n_reactions)*params[pname]
                
            ###   'rho_rev' and 'K_P' are really the same parameter. both here for legacy compatibility reasons
            
            if 'RT' not in params:
                params.update({'RT': 1.}) 
            if 'Enzyme_alloc' not in params: ##species differ in their enzyme allocation
                Ealloc =create_nu_matrix(n_sp, n_reactions, nu_constraint=assumptions['nu_constraint'], params=params)
                params.update({'Enzyme_alloc': Ealloc})
            
            ### KS, KP, Kcat are all species indepent, reaction specific    
            if 'K_S' not in params:
                KS = np.ones(n_reactions).reshape(n_reactions)          
                params.update({'K_S': KS})  
                
            if 'Kcat' not in params:
                Kcat = np.ones(n_reactions).reshape(n_reactions) 
                params.update({'Kcat': Kcat})
                
            if 'thermodynamic_inhibition_dynamicYield'in assumptions['growth_type'] and 'Diss_per_Biomass' not in params:#
                params.update({'Diss_per_Biomass': 1.})
                
                
            
            with warnings.catch_warnings(): ## we are okay with dividing by zero in this section.
#                warnings.filterwarnings("ignore", message="float division by zero")
#                warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
                warnings.filterwarnings("ignore")
                    # could also have used this error catcher: with np.errstate(divide='ignore', invalid='ignore'): 
                if 'K_P' in params and 'rho_rev' in params:
                    print ('only rho_rev used, Kp is not used ') ## refer to [Noor etal FEBS lett 2013]          
                    params.update({'K_P': 1./params['rho_rev']})                
                elif 'K_P' in params:
                    params.update({'rho_rev': 1./ params['K_P']})
                elif 'rho_rev' in params:
                    params.update({'K_P': 1./params['rho_rev']}) 
                
                else: # KP and rho_rev are both not provided.
                    params.update({'rho_rev': np.ones(n_reactions).reshape(n_reactions)}) 
                    params.update({'K_P': 1./params['rho_rev']}) 

            
            
        elif 'Michaelis-Menten' in assumptions['growth_type'] :
             if 'Kcat' not in params:
                Kcat = np.ones(n_reactions).reshape(n_reactions) 
                params.update({'Kcat': Kcat})
             if 'K_S' not in params:
                KS = np.ones(n_reactions).reshape(n_reactions)          
                params.update({'K_S': KS})
             if 'Enzyme_alloc' not in params: ##species differ in their enzyme allocation
                Ealloc =create_nu_matrix(n_sp, n_reactions, nu_constraint=assumptions['nu_constraint'], params=params)
                params.update({'Enzyme_alloc': Ealloc})
        
        elif 'product' in assumptions['growth_type'] :     
            if 'Keq' not in params:
        #         K_eqs = np.random.lognormal(mean=1., sigma=1., size=n_reactions) 
                K_eqs = 2*np.ones(n_sp*n_reactions).reshape(n_sp, n_reactions)  
                ## if species independent and reshaped, need to edit compress params code.                                                                   
                params.update({'Keq': K_eqs})
            if 'Km' not in params:
                Km = 0.5*np.ones(n_sp*n_reactions).reshape(n_sp, n_reactions) 
                params.update({'Km': Km})
    
            if 'nu_max' not in params:
                nu_max =np.random.lognormal(
                    mean=10., sigma=1., size=n_sp*n_reactions).reshape(n_sp, n_reactions)
                params.update({'nu_max': nu_max})
        
   
                
                
        elif 'linear' in assumptions['growth_type'] :
            if 'nu_max' not in params:
                nu_max =np.random.lognormal(
                    mean=10., sigma=1., size=n_sp*n_reactions).reshape(n_sp, n_reactions)
                params.update({'nu_max': nu_max})
        
        
        else:   
            print ('badly specified growth type', assumptions['growth_type'])

        
        if 'anabolism' in assumptions['growth_type']:
            assumptions.update({'supply': 'external_noRdilution'})
            if 'b_i' not in params:         
                params.update({'b_i': np.ones(n_res)})
        
    #     dNdt=MakeConsumerDynamics(assumptions)(N,R,params)
    #     dRdt=MakeResourceDynamics(assumptions)(N,R,params)    
        return N, R, params, assumptions




    
def MakeConsumerDynamics(assumptions):
    """
    Returns a function of N, R, and the model parameters, which itself returns the
        vector of resource rates of change dN/dt
    """
    
    def dynamic_yield(R,params): ### simplified implementation of Heijnen method based on constant dissipation approximation from von stockar,..liu et all 2008
        dG_cat= params['DG']/params['RT'] +  np.log(R[params['products']]/R[params['substrates']])   ## does not account for ATP production, like in Heijnen method papers. 
        dynamic_yield_val= -dG_cat/params['Diss_per_Biomass']
        
#        print (dG_cat)
#        print (dynamic_yield_val)
#        sys.exit(1)
        return np.clip(dynamic_yield_val,0, None)

    
    
    
    def species_independent_part_of_JEnergy_thermo(R,params):
        #(params['Energy_yield']/params['yield']) converts metabolic flux into biomass yield       
        if params['stoichiometry_provided']: ## stoichiometry of reactions is not 1     
            return  (params['Energy_yield']/params['yield']
                     )* params['Kcat']*np.power(R[params['substrates']]/params['K_S'],params['stoich_subs']) *(
                             1.-np.exp(params['DG']/params['RT']) *np.power(R[params['products']],params['stoich_prods'])/np.power(R[params['substrates']],params['stoich_subs'])
                         )/(1.+np.power(R[params['substrates']]/params['K_S'],params['stoich_subs'])+ np.power(R[params['products']]*params['rho_rev'],params['stoich_prods']))
        else:
            return  (params['Energy_yield']/params['yield']
                     )* params['Kcat']*(R[params['substrates']]/params['K_S'])*(1.-np.exp(params['DG']/params['RT']) *R[params['products']]/R[params['substrates']]
                         )/(1.+R[params['substrates']]/params['K_S']+ params['rho_rev']*R[params['products']])
    
    
    def species_independent_part_of_JEnergy_thermo_dynamicYield(R,params):
        #(params['Energy_yield']/params['yield']) converts metabolic flux into biomass yield       
        if params['stoichiometry_provided']: ## stoichiometry of reactions is not 1     
            print('error not implemented this for dynamic yield')
        else:
            return  (dynamic_yield(R,params)
                     )* params['Kcat']*(R[params['substrates']]/params['K_S'])*(1.-np.exp(params['DG']/params['RT']) *R[params['products']]/R[params['substrates']]
                         )/(1.+R[params['substrates']]/params['K_S']+ params['rho_rev']*R[params['products']])
    
    
    
    def species_independent_part_of_MM(R, params):
        if params['stoichiometry_provided']: ## stoichiometry of reactions is not 1  
            return (params['Energy_yield']/params['yield']
                     )*params['Kcat']*(np.power(R[params['substrates']],params['stoich_subs'])/params['K_S'])/(
                            1.+np.power(R[params['substrates']],params['stoich_subs'])/params['K_S'])
        else:
            return (params['Energy_yield']/params['yield']
                     )*params['Kcat']*(R[params['substrates']]/params['K_S'])/(
                            1.+R[params['substrates']]/params['K_S'])      
         
    sigma = {'linear': lambda R, params: params['nu_max']*R[params['substrates']],
                         
             
                     
            'thermodynamic_inhibition': lambda R, params: np.clip( 
                    params['Enzyme_alloc']* species_independent_part_of_JEnergy_thermo(R,params),0, None), 
            
            'thermodynamic_inhibition_dynamicYield': lambda R, params: np.clip( 
                    params['Enzyme_alloc']* species_independent_part_of_JEnergy_thermo_dynamicYield(R,params),0, None), 
                     
                            
            'thermodynamic_inhibition_unclipped': lambda R, params:
                    params['Enzyme_alloc']* species_independent_part_of_JEnergy_thermo(R,params), 
#             'thermodynamic_inhibition_unclipped': lambda R, params: params['Kcat']*params['Enzyme_alloc']*(R[params['substrates']]/params['K_S'])*(1.-np.exp(params['DG_net']/params['RT'])
#                     )/(1.+R[params['substrates']]/params['K_S']+R[params['products']]/params['K_P']), 
             
            'anabolism_noRdilution':lambda R, params: np.clip( 
                    params['Enzyme_alloc']* species_independent_part_of_JEnergy_thermo(R,params),0, None), 
                    ##identical to thermo model for consumer dynamics 
                
            'Michaelis-Menten': lambda R, params: 
                params['Enzyme_alloc']*species_independent_part_of_MM(R, params),
            
             'product_inhibition': lambda R, params: np.clip( 
                     params['nu_max']*(R[params['substrates']]-R[params['products']]/params['Keq'])/(params['Km'] 
                +R[params['substrates']]+params['rho']*R[params['products']]/params['Keq']),0, None),             
             'product_inhibition_unclipped': lambda R, params: params['nu_max']*(R[params['substrates']]-R[params['products']]/params['Keq'])/(params['Km'] 
                +R[params['substrates']]+params['rho']*R[params['products']]/params['Keq'])
                        
            }
    def J_in(R, params): return sigma[assumptions['growth_type']](R, params)
    #J_in = lambda R,params: sigma[assumptions['growth_type']](R, params)
    

    return lambda N, R, params: params['g']*N*(np.sum(J_in(R, params),axis=1)-params['m'])


def MakeResourceDynamics(assumptions,return_fluxes=False):
    """
    Returns a function of N, R, and the model parameters, which itself returns the
        vector of resource rates of change dR/dt
    """
    
    def species_independent_part_of_J_thermo(R,params):        
        if params['stoichiometry_provided']: ## stoichiometry of reactions is not 1  
            return  params['Kcat']*np.power(R[params['substrates']]/params['K_S'],params['stoich_subs']) *(
                    1.-np.exp(params['DG']/params['RT']) *np.power(R[params['products']],params['stoich_prods'])/np.power(R[params['substrates']],params['stoich_subs'])
                         )/(1.+np.power(R[params['substrates']]/params['K_S'],params['stoich_subs'])+np.power(R[params['products']]*params['rho_rev'],params['stoich_prods']))
            
        else:
            return params['Kcat']*(R[params['substrates']]/params['K_S'])*(1.-np.exp(params['DG']/params['RT']) *R[params['products']]/R[params['substrates']]
                         )/(1.+R[params['substrates']]/params['K_S']+params['rho_rev']*R[params['products']])
    
      
    def species_independent_part_of_MM(R, params):
        if params['stoichiometry_provided']: ## stoichiometry of reactions is not 1  
            return params['Kcat']*(np.power(R[params['substrates']]/params['K_S'],params['stoich_subs']))/(
                            1.+np.power(R[params['substrates']]/params['K_S'],params['stoich_subs']))
        else:
            return params['Kcat']*(R[params['substrates']]/params['K_S'])/(
                            1.+R[params['substrates']]/params['K_S'])
    
    sigma = {'linear': lambda R, params: params['nu_max']*R[params['substrates']],
             
             'thermodynamic_inhibition': lambda R, params: np.clip( 
                    params['Enzyme_alloc']* species_independent_part_of_J_thermo(R,params),0, None), 
                     
            'thermodynamic_inhibition_dynamicYield': lambda R, params: np.clip(         ### same as thermodynamic inhibition because dynamic yield doesnt affect instantaneous resource consumption
                    params['Enzyme_alloc']* species_independent_part_of_J_thermo(R,params),0, None), 
                     
            'thermodynamic_inhibition_unclipped': lambda R, params:
                    params['Enzyme_alloc']* species_independent_part_of_J_thermo(R,params),
            
            'anabolism_noRdilution':lambda R, params: np.clip( 
                    params['Enzyme_alloc']* species_independent_part_of_J_thermo(R,params),0, None) , 
                    ##identical to thermo model for consumer dynamics 
                    
            'Michaelis-Menten': lambda R, params:
                    params['Enzyme_alloc']* species_independent_part_of_MM(R, params),
                     
             'product_inhibition': lambda R, params:  np.clip( params['nu_max']*(
                 R[params['substrates']]-R[params['products']]/params['Keq'])/(params['Km'] 
                +R[params['substrates']]+params['rho']*R[params['products']]/params['Keq']),0,None),
             'product_inhibition_unclipped': lambda R, params: params['nu_max']*(
                 R[params['substrates']]-R[params['products']]/params['Keq'])/(params['Km'] 
                +R[params['substrates']]+params['rho']*R[params['products']]/params['Keq'])            
            }
        
    h = {'off': lambda R, params: 0.,
         'external': lambda R, params: (params['R_supply']-R)/params['tau'],
         'self-renewing': lambda R, params: params['r']*R*(params['R_supply']-R)/params['tau'],
         'external_noRdilution':lambda R, params: params['R_supply']/params['tau'],
         }
    ## matrix multiplication to map substrates/products of each reaction into resources
    
    def J_in_res(R, params): 
        if params['stoichiometry_provided']:
            return np.matmul(params['stoich_subs']*sigma[assumptions['growth_type']](R, params), params['Sub_to_res'])
        else:
            return np.matmul(sigma[assumptions['growth_type']](R, params), params['Sub_to_res'])

    def J_out_res(R, params): 
        if params['stoichiometry_provided']:
            return np.matmul(params['stoich_prods']*sigma[assumptions['growth_type']](R, params), params['Pro_to_res'])
        else:
            return np.matmul(sigma[assumptions['growth_type']](R, params), params['Pro_to_res'])
    
#    J_in_res = lambda R,params: np.matmul(sigma[assumptions['growth_type']](R, params), params['Sub_to_res']) 
#    J_out_res = lambda R,params: np.matmul(sigma[assumptions['growth_type']](R, params), params['Pro_to_res'])
    
    
    if return_fluxes== True: ## the user wants the fluxes through each reaction, not the change in resource abundances
        return lambda N, R, params: N[:,None]*sigma[assumptions['growth_type']](R, params)
        
     
    if assumptions['growth_type']== 'anabolism_noRdilution':
        return lambda N, R, params: (h[assumptions['supply']](R, params)
                                     - J_in_res(R, params).T.dot(N)
                                     + J_out_res(R, params).T.dot(N) - params['b_i']*R*np.sum(N)   )
    
    return lambda N, R, params: (h[assumptions['supply']](R, params)
                                     - J_in_res(R, params).T.dot(N)
                                     + J_out_res(R, params).T.dot(N))
    

def Make_flux_calculator(assumptions):
    return MakeResourceDynamics(assumptions,return_fluxes=True)
    
def compress_params(params, not_extinct, n_sp):
    params_comp=deepcopy(params)
    ##compresses only species specific parameters.    
    if 'Enzyme_alloc' in params_comp.keys():
        params_comp['Enzyme_alloc'] = params['Enzyme_alloc'][not_extinct[:n_sp], :]
        
    
    
    
    ######## these are species specific parameters in product inhibition growth type
    if 'nu_max' in params_comp.keys():       
        params_comp['nu_max'] = params['nu_max'][not_extinct[:n_sp], :]
       
    if 'Km' in params_comp.keys():
        params_comp['Km'] = params['Km'][not_extinct[:n_sp], :]
       
    if 'Keq' in params_comp.keys():
        params_comp['Keq'] = params['Keq'][not_extinct[:n_sp], :]
    if 'rho' in params_comp.keys():
        if type(params_comp['rho' ]) == np.ndarray:
            params_comp['rho' ] = params['rho' ][not_extinct[:n_sp], :]
    ##############################################################################        
            
   ######### other potentially species-specific parameters.        
    for name in ['m', 'g']:
        if name in params_comp.keys():
            if type(params_comp[name]) == np.ndarray:
                assert len(params_comp[name]
                           ) == n_sp, 'Invalid length for ' + name
                params_comp[name] = params[name][not_extinct[:n_sp]]
    return params_comp
    
 
    
def IntegrateDynamics(CommInstance,params,T0=0,T=1,ns=2,
                      compress_species=True, return_all=False, log_time=False):
    """
    N,R,dNdt,dRdt,n_sp,n_res
    Integrator for dynamics
    """
    y0=np.hstack([CommInstance.N,CommInstance.R]) 
    not_extinct = np.full(len(y0), True, dtype=bool)
    
    if compress_species: ## remove extinct species from ODEint to speed up
        not_extinct[:CommInstance.n_sp] = y0[:CommInstance.n_sp]>CommInstance.params['popn_cutoff']
    ##resources are not compressed since they are supplied externally, 
    ##(and reactions would need to managed carefully as well.  )
    
    
    ## make into 1D array to simulate dynamics
    S_comp = np.sum(not_extinct[:CommInstance.n_sp]) #record the new point dividing species from resources  
    not_extinct_idx = np.where(not_extinct)[0]
    y0_comp = y0[not_extinct]
    params_comp=compress_params(CommInstance.params, not_extinct, CommInstance.n_sp)

    if log_time:
        t = 10**(np.linspace(np.log10(T0),np.log10(T0+T),ns))
    else:
        t = np.linspace(T0,T0+T,ns)
        
    if return_all:
        out = integrate.odeint(CommInstance.dydt,y0_comp,t,args=(params_comp,S_comp),mxstep=10000,atol=1e-4)
        traj = np.zeros((np.shape(out)[0],CommInstance.n_sp+CommInstance.n_res))    
        traj[:,not_extinct_idx] = out    
        Ntraj=traj[:,:CommInstance.n_sp]
        Rtraj=traj[:,CommInstance.n_sp:]
        return t, Ntraj, Rtraj
    else:
        try:
            out = integrate.odeint(CommInstance.dydt,y0_comp,t,args=(params_comp,S_comp),mxstep=10000,atol=1e-4)[-1]
        except ValueError as err:
            print ("value error was thrown: ",err)
            print (not_extinct_idx, y0_comp)
            print (params_comp,S_comp)
            
            
            return None, [y0_comp, params_comp,S_comp]
           
        yf = np.zeros(len(y0))
        yf[not_extinct_idx] = out
        Nf=yf[:CommInstance.n_sp]
        Rf=yf[CommInstance.n_sp:]
        return Nf, Rf   
        


