#!/usr/bin/env python
# coding: utf-8
import numpy as np
import random
import gurobipy as gp
from gurobipy import Model,quicksum,GRB
import matplotlib.pyplot as plt
import time
import scipy.stats as stats
from Algorithm import *

#run the GAP Algorithm with T=500,1000, k=2,5,10,20 and gamma from 0 to 1, where T is the arrival sequence length and k is the capacity of all the bins. The demand distribution is chosen to be bernoulli distribution.

def run_gamma(dis='ber'):
    testnum=20
    testiter=5

    Gamma=np.arange(0,1.1,0.1)
    len_G=len(Gamma)

    #used to save the rewards of GAP algorithm in each iteration
    GAP_1=np.zeros(testnum*testiter)
    GAP_2=np.zeros(testnum*testiter)
    GAP_3=np.zeros(testnum*testiter)
    GAP_4=np.zeros(testnum*testiter)
    GAP_5=np.zeros(testnum*testiter)
    GAP_6=np.zeros(testnum*testiter)
    GAP_7=np.zeros(testnum*testiter)
    GAP_8=np.zeros(testnum*testiter)
    
    #used to save the rewards of LP algorithm in each iteration
    LP_1=np.zeros(testnum*testiter)
    LP_2=np.zeros(testnum*testiter)
    LP_3=np.zeros(testnum*testiter)
    LP_4=np.zeros(testnum*testiter)
    LP_5=np.zeros(testnum*testiter)
    LP_6=np.zeros(testnum*testiter)
    LP_7=np.zeros(testnum*testiter)
    LP_8=np.zeros(testnum*testiter)
    
    #used to save the LP ratio in each iteration
    Ratio_GAP_1=np.zeros(testnum*testiter)
    Ratio_GAP_2=np.zeros(testnum*testiter)
    Ratio_GAP_3=np.zeros(testnum*testiter)
    Ratio_GAP_4=np.zeros(testnum*testiter)
    Ratio_GAP_5=np.zeros(testnum*testiter)
    Ratio_GAP_6=np.zeros(testnum*testiter)
    Ratio_GAP_7=np.zeros(testnum*testiter)
    Ratio_GAP_8=np.zeros(testnum*testiter)
    
    #used to save the average lp ratio
    Ra_GAP_1=np.zeros(len_G)
    Ra_GAP_2=np.zeros(len_G)
    Ra_GAP_3=np.zeros(len_G)
    Ra_GAP_4=np.zeros(len_G)
    Ra_GAP_5=np.zeros(len_G)
    Ra_GAP_6=np.zeros(len_G)
    Ra_GAP_7=np.zeros(len_G)
    Ra_GAP_8=np.zeros(len_G)
    
#used to save the average rewards
    R_GAP_1=np.zeros(len_G)
    R_GAP_2=np.zeros(len_G)
    R_GAP_3=np.zeros(len_G)
    R_GAP_4=np.zeros(len_G)
    R_GAP_5=np.zeros(len_G)
    R_GAP_6=np.zeros(len_G)
    R_GAP_7=np.zeros(len_G)
    R_GAP_8=np.zeros(len_G)
    
    S_GAP_1=np.zeros(len_G)
    S_GAP_2=np.zeros(len_G)
    S_GAP_3=np.zeros(len_G)
    S_GAP_4=np.zeros(len_G)
    S_GAP_5=np.zeros(len_G)
    S_GAP_6=np.zeros(len_G)
    S_GAP_7=np.zeros(len_G)
    S_GAP_8=np.zeros(len_G)

    c_2=np.array([5,5,5,5,5,5,5,5,5,5],dtype=np.float64)
    c_1=np.array([10,10,10,10,10,10,10,10,10,10],dtype=np.float64)
    c_3=np.array([2,2,2,2,2,2,2,2,2,2],dtype=np.float64)
    c_4=np.array([20,20,20,20,20,20,20,20,20,20],dtype=np.float64)


    for gamma in Gamma:

        for x in range(testnum):
            
            p_1=np.loadtxt('Instances/Instance10/Bin10/P/p'+str(x))
            v_1=np.loadtxt('Instances/Instance10/Bin10/V/v'+str(x))
            
            p_2=np.loadtxt('Instances/Instance5/Bin10/P/p'+str(x))
            v_2=np.loadtxt('Instances/Instance5/Bin10/V/v'+str(x))
            
            p_3=np.loadtxt('Instances/Instance2/Bin10/P/p'+str(x))
            v_3=np.loadtxt('Instances/Instance2/Bin10/V/v'+str(x))
            
            p_4=np.loadtxt('Instances/Instance20/Bin10/P/p'+str(x))
            v_4=np.loadtxt('Instances/Instance20/Bin10/V/v'+str(x))
            #p=generate_p(10)
            #v=generate_v(10,10)
            
            if dis=='truncnorm':
                #parameter=generate_truncnorm(10)
                parameter_1=np.loadtxt('Instances/Instance10/Bin10/Tr/tr'+str(x))
                parameter_2=np.loadtxt('Instances/Instance5/Bin10/Tr/tr'+str(x))
                parameter_3=np.loadtxt('Instances/Instance2/Bin10/Tr/tr'+str(x))
                parameter_4=np.loadtxt('Instances/Instance20/Bin10/Tr/tr'+str(x))
            
            elif dis=='uni':
                #parameter=generate_uni(10,1)
                parameter_1=np.loadtxt('Instances/Instance10/Bin10/Uni/uni'+str(x))
                parameter_2=np.loadtxt('Instances/Instance5/Bin10/Uni/uni'+str(x))
                parameter_3=np.loadtxt('Instances/Instance2/Bin10/Uni/uni'+str(x))
                parameter_4=np.loadtxt('Instances/Instance20/Bin10/Uni/uni'+str(x))
                
            else:
                #parameter=generate_ber(10)
                parameter_1=np.loadtxt('Instances/Instance10/Bin10/Ber/ber'+str(x))
                parameter_2=np.loadtxt('Instances/Instance5/Bin10/Ber/ber'+str(x))
                parameter_3=np.loadtxt('Instances/Instance2/Bin10/Ber/ber'+str(x))
                parameter_4=np.loadtxt('Instances/Instance20/Bin10/Ber/ber'+str(x))
                
            
            sequence=np.zeros((testiter,500,2))
            np.random.seed(1)
            for test in range(testiter):
                
                if dis=='truncnorm':
                    sequence[test]=simulation_truncnorm(p_1,parameter_1,10,500)
                    s=sequence[test][:,1]
                    
                elif dis=='uni':
                    sequence[test]=simulation_uni(p_1,parameter_1,10,500)
                    s=sequence[test][:,1]
                    
                else:
                    sequence[test]=simulation_ber(p_1,parameter_1,10,500)
                    s=sequence[test][:,1]
                
                LP_1[test+x*testiter]=solve_OPT(c_1,v_1,p_1,s,sequence[test])[0]
                    
            np.random.seed(1)
            for test in range(testiter):
                
                GAP_1[test+x*testiter]=GAP_Algorithm(c_1,v_1,p_1,sequence[test],100,gamma,10,0)
                
                

                
                Ratio_GAP_1[test+x*testiter]=GAP_1[test+x*testiter]/LP_1[test+x*testiter]
                
                
            sequence=np.zeros((testiter,500,2))
            np.random.seed(1)
            for test in range(testiter):
                
                if dis=='truncnorm':
                    sequence[test]=simulation_truncnorm(p_2,parameter_2,10,500)
                    s=sequence[test][:,1]
                    
                elif dis=='uni':
                    sequence[test]=simulation_uni(p_2,parameter_2,10,500)
                    s=sequence[test][:,1]
                    
                else:
                    sequence[test]=simulation_ber(p_2,parameter_2,10,500)
                    s=sequence[test][:,1]
                    
                LP_3[test+x*testiter]=solve_OPT(c_2,v_2,p_2,s,sequence[test])[0]
                    
            np.random.seed(1)
            for test in range(testiter):
                
                GAP_3[test+x*testiter]=GAP_Algorithm(c_2,v_2,p_2,sequence[test],100,gamma,10,0)
                Ratio_GAP_3[test+x*testiter]=GAP_3[test+x*testiter]/LP_3[test+x*testiter]
                
                
            sequence=np.zeros((testiter,1000,2))
            np.random.seed(1)
            for test in range(testiter):
                
                if dis=='truncnorm':
                    sequence[test]=simulation_truncnorm(p_2,parameter_2,10,1000)
                    s=sequence[test][:,1]
                    
                elif dis=='uni':
                    sequence[test]=simulation_uni(p_2,parameter_2,10,1000)
                    s=sequence[test][:,1]
                    
                else:
                    sequence[test]=simulation_ber(p_2,parameter_2,10,1000)
                    s=sequence[test][:,1]
                
                LP_4[test+x*testiter]=solve_OPT(c_2,v_2,p_2,s,sequence[test])[0]
            
            np.random.seed(1)
            for test in range(testiter):
                    
                GAP_4[test+x*testiter]=GAP_Algorithm(c_2,v_2,p_2,sequence[test],100,gamma,10,0)

                
                Ratio_GAP_4[test+x*testiter]=GAP_4[test+x*testiter]/LP_4[test+x*testiter]
            
            sequence=np.zeros((testiter,1000,2))
            np.random.seed(1)
            for test in range(testiter):
                
                if dis=='truncnorm':
                    sequence[test]=simulation_truncnorm(p_1,parameter_1,10,1000)
                    s=sequence[test][:,1]
                    
                elif dis=='uni':
                    sequence[test]=simulation_uni(p_1,parameter_1,10,1000)
                    s=sequence[test][:,1]
                    
                else:
                    sequence[test]=simulation_ber(p_1,parameter_1,10,1000)
                    s=sequence[test][:,1]
                    
                LP_2[test+x*testiter]=solve_OPT(c_1,v_1,p_1,s,sequence[test])[0]


            np.random.seed(1)
            for test in range(testiter):

                GAP_2[test+x*testiter]=GAP_Algorithm(c_1,v_1,p_1,sequence[test],100,gamma,10,0)
                Ratio_GAP_2[test+x*testiter]=GAP_2[test+x*testiter]/LP_2[test+x*testiter]
                
                
            sequence=np.zeros((testiter,1000,2))
            np.random.seed(1)
            for test in range(testiter):
                
                if dis=='truncnorm':
                    sequence[test]=simulation_truncnorm(p_3,parameter_3,10,1000)
                    s=sequence[test][:,1]
                    
                elif dis=='uni':
                    sequence[test]=simulation_uni(p_3,parameter_3,10,1000)
                    s=sequence[test][:,1]
                    
                else:
                    sequence[test]=simulation_ber(p_3,parameter_3,10,1000)
                    s=sequence[test][:,1]
                LP_5[test+x*testiter]=solve_OPT(c_3,v_3,p_3,s,sequence[test])[0]
                    



            np.random.seed(1)
            for test in range(testiter):

                GAP_5[test+x*testiter]=GAP_Algorithm(c_3,v_3,p_3,sequence[test],100,gamma,10,0)
                Ratio_GAP_5[test+x*testiter]=GAP_5[test+x*testiter]/LP_5[test+x*testiter]
                
            
            sequence=np.zeros((testiter,500,2))    
            np.random.seed(1)
            for test in range(testiter):
                
                if dis=='truncnorm':
                    sequence[test]=simulation_truncnorm(p_3,parameter_3,10,500)
                    s=sequence[test][:,1]
                    
                elif dis=='uni':
                    sequence[test]=simulation_uni(p_3,parameter_3,10,500)
                    s=sequence[test][:,1]
                    
                else:
                    sequence[test]=simulation_ber(p_3,parameter_3,10,500)
                    s=sequence[test][:,1]
                    
                LP_6[test+x*testiter]=solve_OPT(c_3,v_3,p_3,s,sequence[test])[0]
                    



            np.random.seed(1)
            for test in range(testiter):

                GAP_6[test+x*testiter]=GAP_Algorithm(c_3,v_3,p_3,sequence[test],100,gamma,10,0)
                Ratio_GAP_6[test+x*testiter]=GAP_6[test+x*testiter]/LP_6[test+x*testiter]
            
            
            sequence=np.zeros((testiter,1000,2)) 
            np.random.seed(1)
            for test in range(testiter):
                
                if dis=='truncnorm':
                    sequence[test]=simulation_truncnorm(p_4,parameter_4,10,1000)
                    s=sequence[test][:,1]
                    
                elif dis=='uni':
                    sequence[test]=simulation_uni(p_4,parameter_4,10,1000)
                    s=sequence[test][:,1]
                    
                else:
                    sequence[test]=simulation_ber(p_4,parameter_4,10,1000)
                    s=sequence[test][:,1]
                
                LP_7[test+x*testiter]=solve_OPT(c_4,v_4,p_4,s,sequence[test])[0]
                    



            np.random.seed(1)
            for test in range(testiter):

                GAP_7[test+x*testiter]=GAP_Algorithm(c_4,v_4,p_4,sequence[test],100,gamma,10,0)
                Ratio_GAP_7[test+x*testiter]=GAP_7[test+x*testiter]/LP_7[test+x*testiter]
                
            
            sequence=np.zeros((testiter,500,2))
            np.random.seed(1)
            for test in range(testiter):
                
                if dis=='truncnorm':
                    sequence[test]=simulation_truncnorm(p_4,parameter_4,10,500)
                    s=sequence[test][:,1]
                    
                elif dis=='uni':
                    sequence[test]=simulation_uni(p_4,parameter_4,10,500)
                    s=sequence[test][:,1]
                    
                else:
                    sequence[test]=simulation_ber(p_4,parameter_4,10,500)
                    s=sequence[test][:,1]
                LP_8[test+x*testiter]=solve_OPT(c_4,v_4,p_4,s,sequence[test])[0]
                    



            np.random.seed(1)
            for test in range(testiter):

                GAP_8[test+x*testiter]=GAP_Algorithm(c_4,v_4,p_4,sequence[test],100,gamma,10,0)
                Ratio_GAP_8[test+x*testiter]=GAP_8[test+x*testiter]/LP_8[test+x*testiter]

                






        R_GAP_1[int(gamma/0.1)]=np.mean(GAP_1)
        R_GAP_2[int(gamma/0.1)]=np.mean(GAP_2)
        R_GAP_3[int(gamma/0.1)]=np.mean(GAP_3)
        R_GAP_4[int(gamma/0.1)]=np.mean(GAP_4)
        R_GAP_5[int(gamma/0.1)]=np.mean(GAP_5)
        R_GAP_6[int(gamma/0.1)]=np.mean(GAP_6)
        R_GAP_7[int(gamma/0.1)]=np.mean(GAP_7)
        R_GAP_8[int(gamma/0.1)]=np.mean(GAP_8)
        
        Ra_GAP_1[int((gamma/0.1))]=np.mean(Ratio_GAP_1)
        Ra_GAP_2[int((gamma/0.1))]=np.mean(Ratio_GAP_2)
        Ra_GAP_3[int((gamma/0.1))]=np.mean(Ratio_GAP_3)
        Ra_GAP_4[int((gamma/0.1))]=np.mean(Ratio_GAP_4)
        Ra_GAP_5[int((gamma/0.1))]=np.mean(Ratio_GAP_5)
        Ra_GAP_6[int((gamma/0.1))]=np.mean(Ratio_GAP_6)
        Ra_GAP_7[int((gamma/0.1))]=np.mean(Ratio_GAP_7)
        Ra_GAP_8[int((gamma/0.1))]=np.mean(Ratio_GAP_8)
        
        S_GAP_1[int((gamma/0.1))]=np.std(Ratio_GAP_1)
        S_GAP_2[int((gamma/0.1))]=np.std(Ratio_GAP_2)
        S_GAP_3[int((gamma/0.1))]=np.std(Ratio_GAP_3)
        S_GAP_4[int((gamma/0.1))]=np.std(Ratio_GAP_4)
        S_GAP_5[int((gamma/0.1))]=np.std(Ratio_GAP_5)
        S_GAP_6[int((gamma/0.1))]=np.std(Ratio_GAP_6)
        S_GAP_7[int((gamma/0.1))]=np.std(Ratio_GAP_7)
        S_GAP_8[int((gamma/0.1))]=np.std(Ratio_GAP_8)
        
    f1=open(dis+'-gamma.txt','w+')
    f1.write('Gamma K2T500 K2T1000 K5T500 K5T1000 K10T500 K10T1000 K20T500 K20T1000'+'\n')
    for i in range(len_G):
        f1.write(str(0.1*i)+' ')
        f1.write(str(R_GAP_6[i])+' ')
        f1.write(str(R_GAP_5[i])+' ')
        f1.write(str(R_GAP_3[i])+' ')
        f1.write(str(R_GAP_4[i])+' ')
        f1.write(str(R_GAP_1[i])+' ')
        f1.write(str(R_GAP_2[i])+' ')
        f1.write(str(R_GAP_8[i])+' ')
        f1.write(str(R_GAP_7[i])+'\n')
        
    f1.close()
    
    f2=open(dis+'-ratio-gamma.txt','w+')
    f2.write('Gamma K2T500 K2T1000 K5T500 K5T1000 K10T500 K10T1000 K20T500 K20T1000'+'\n')
    for i in range(len_G):
        f2.write(str(0.1*i)+' ')
        f2.write(str(Ra_GAP_6[i])+'_'+str(S_GAP_6[i])+' ')
        f2.write(str(Ra_GAP_5[i])+'_'+str(S_GAP_5[i])+' ')
        f2.write(str(Ra_GAP_3[i])+'_'+str(S_GAP_3[i])+' ')
        f2.write(str(Ra_GAP_4[i])+'_'+str(S_GAP_4[i])+' ')
        f2.write(str(Ra_GAP_1[i])+'_'+str(S_GAP_1[i])+' ')
        f2.write(str(Ra_GAP_2[i])+'_'+str(S_GAP_2[i])+' ')
        f2.write(str(Ra_GAP_8[i])+'_'+str(S_GAP_8[i])+' ')
        f2.write(str(Ra_GAP_7[i])+'_'+str(S_GAP_7[i])+'\n')
    f2.close()