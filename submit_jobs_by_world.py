#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 18:42:01 2022

@author: ashish
"""


import numpy as np
import subprocess
import os
import sys
import stat


if os.getcwd().startswith("/Users/ashish"):
    sys.path.append('/Users/ashish/Documents/GitHub/metabolic-CRmodel/')
else:   
    sys.path.append('/home/a-m/ashishge/thermodynamic_model/code/')

   
from metabolic_CR_model_functions import *
from metabolic_CR_model_class import *

home="/home/a-m/ashishge/thermodynamic_model/code/"
code_name="simulate_1_irr_1world.py"
this_file="submit_jobs_by_world.py"
analysis_code_name="analyze_sim.py"
#destfold="/home/a-m/ashishge/thermodynamic_model/data/S4_1irr_2/"
destfold="/home/a-m/ashishge/thermodynamic_model/data/S6_1irr_4/"
if not os.path.isdir(destfold):os.mkdir(destfold)

number_of_worlds=50 #10 for S6_1irr_1

    
#################     copying these files to the runfolder ####################
copy_command="cp -rf "+this_file+" "+destfold+this_file
exit_status = subprocess.call(copy_command, shell=True)
if exit_status == 1:   print ("\n Job {0} failed to submit".format(copy_command) ) # Check to make sure the job submitted
copy_command="cp -rf "+code_name+" "+destfold+code_name
exit_status = subprocess.call(copy_command, shell=True)
if exit_status == 1: print ("\n Job {0} failed to submit".format(copy_command)) # Check to make sure the job submitted
copy_command="cp -rf "+analysis_code_name+" "+destfold+analysis_code_name
exit_status = subprocess.call(copy_command, shell=True)
if exit_status == 1: print ("\n Job {0} failed to submit".format(copy_command)) # Check to make sure the job submitted
         
for world_id in range(number_of_worlds):
    sim_file_name = 'sim_1irr'+'-w'+str(world_id)+'.dat'    
    
    os.chdir(destfold) 
    st = os.stat(code_name)#### changes the file permissions just in case
    os.chmod(code_name, st.st_mode | stat.S_IEXEC) 
    
    
    sim_id=world_id
    jobname='sim_world_'+str(sim_id)
    output_file='out_w'+str(sim_id)+'.txt'
    subscript_str = f"""#!/bin/bash -l
    module load Python/3.7.2-IGB-gcc-8.2.0
    #SBATCH -J {jobname}
    #SBATCH -o {output_file}
    python {code_name} -f {sim_file_name} -d {destfold}"""
    ### for input file containing parameters use: -i {input_file_name}"""
    
    subscript_name='sub_job'+str(sim_id)+'.sh'
    with open(subscript_name, "w") as script:
            print(subscript_str, file = script)
            script.close()
    qsub_command = "sbatch "+subscript_name
    print(qsub_command)
    
    
    Popen_return=subprocess.Popen(qsub_command,shell=True, stdout=subprocess.PIPE)
    
    response=Popen_return.communicate()[0].decode("utf-8") 
    
    job_id=response.replace('Submitted batch job ','')
    job_id=job_id.replace('\n','').replace(' ','')

#print (response)
#print (job_id)


#exit_status = subprocess.call(qsub_command, shell=True)
#if exit_status == 1:  # Check to make sure the job submitted        
#    print("\n Job {0} failed to submit".format(qsub_command))  




# =============================================================================
# analysis_job_name="analysis"+str(sim_id)   
# output_file=analysis_job_name+".txt"
# subscript_str = f"""#!/bin/bash -l
# module load Python/3.7.2-IGB-gcc-8.2.0
# #SBATCH -J {analysis_job_name}
# #SBATCH -o {output_file}
# python {analysis_code_name} -f {sim_file_name} -d {destfold}"""
# 
# subscript_name='sub_analysis'+str(sim_id)+'.sh'
# with open(subscript_name, "w") as script:
#         print(subscript_str, file = script)
#         script.close()
# qsub_command = "sbatch --depend=afterany:"+job_id+' '+subscript_name
# print(qsub_command)
# 
# exit_status = subprocess.call(qsub_command, shell=True)
# if exit_status == 1:  # Check to make sure the job submitted        
#     print("\n Job {0} failed to submit".format(qsub_command))
# =============================================================================

























