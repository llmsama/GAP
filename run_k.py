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

#run the GAP Algorithm with gamma=0.5,0.8,1-1/sqrt(k), T=500, and k from 2 to 10. where T is the arrival sequence length and k is the capacity of all the bins. The demand distribution is chosen as bernoulli distribution



def run_k(dis='ber'):
    testnum=20
    testiter=5
    
#used to save rewards in each iteration
    GAP_f=np.zeros(testnum*testiter)
    GAP_e=np.zeros(testnum*testiter)
    GAP_baseline=np.zeros(testnum*testiter)
    Greedy=np.zeros(testnum*testiter)
    FF=np.zeros(testnum*testiter)
    LP=np.zeros(testnum*testiter)

    K=np.arange(2,11,1)
    len_K = len(K)
    
    #used to save average reward
    R_GAP_f=np.zeros(len_K)
    R_GAP_e=np.zeros(len_K)
    R_Greedy=np.zeros(len_K)
    R_baseline=np.zeros(len_K)
    R_FF=np.zeros(len_K)
    R_LP=np.zeros(len_K)
    
    #used to save ratio in each iteration
    Ratio_GAP_f=np.zeros(testnum*testiter)
    Ratio_GAP_e=np.zeros(testnum*testiter)
    Ratio_GAP_baseline=np.zeros(testnum*testiter)
    Ratio_Greedy=np.zeros(testnum*testiter)
    Ratio_FF=np.zeros(testnum*testiter)
    
    #used to save average LP ratio
    Ra_GAP_f=np.zeros(len_K)
    Ra_GAP_e=np.zeros(len_K)
    Ra_Greedy=np.zeros(len_K)
    Ra_baseline=np.zeros(len_K)
    Ra_FF=np.zeros(len_K)

        
    S_GAP_f=np.zeros(len_K)
    S_GAP_e=np.zeros(len_K)
    S_Greedy=np.zeros(len_K)
    S_baseline=np.zeros(len_K)
    S_FF=np.zeros(len_K)



    for k in range(2,11,1):

        for x in range(testnum):
            sequence=np.zeros((testiter,500,2))
            p=np.loadtxt('Instances/Instance'+str(k)+'/Bin10/P/p'+str(x))
            v=np.loadtxt('Instances/Instance'+str(k)+'/Bin10/V/v'+str(x))

            #p=generate_p(10)
            #v=generate_v(10,10)
            if dis=='truncnorm':
                #parameter=generate_truncnorm(10)
                parameter=np.loadtxt('Instances/Instance'+str(k)+'/Bin10/Tr/tr'+str(x))
            
            elif dis=='uni':
                #parameter=generate_uni(10,1)
                parameter=np.loadtxt('Instances/Instance'+str(k)+'/Bin10/Uni/uni'+str(x))
                
            elif dis=='ber':
                #parameter=generate_ber(10)
                parameter=np.loadtxt('Instances/Instance'+str(k)+'/Bin10/Ber/ber'+str(x))
            
            np.random.seed(1)

            for test in range(testiter):
                c=np.array([k,k,k,k,k,k,k,k,k,k],dtype=np.float64)
                if dis=='truncnorm':
                    sequence[test]=simulation_truncnorm(p,parameter,10,500)
                   
                    #print(np.shape(sequence)[0])
                    
                elif dis=='uni':               
                    sequence[test]=simulation_uni(p,parameter,10,500)
            
                   # print(np.shape(sequence)[0])
                
                elif dis=='ber':
                    sequence[test]=simulation_ber(p,parameter,10,500)
                    #print(np.shape(sequence)[0])

                
                s=sequence[test][:,1]
                LP[test+x*testiter]=solve_OPT(c,v,p,s,sequence[test])[0]
                FF[test+x*testiter]=First_Fit(v,c,sequence[test])
                Ratio_FF[test+x*testiter]=FF[test+x*testiter]/LP[test+x*testiter]
                
            np.random.seed(1)

            for test in range(testiter):
                c=np.array([k,k,k,k,k,k,k,k,k,k],dtype=np.float64)
                GAP_f[test+x*testiter]=GAP_Algorithm(c,v,p,sequence[test],100,0.5,10,0)
                Ratio_GAP_f[test+x*testiter]=GAP_f[test+x*testiter]/LP[test+x*testiter]
                
            np.random.seed(1)

            for test in range(testiter):
                c=np.array([k,k,k,k,k,k,k,k,k,k],dtype=np.float64)

                GAP_e[test+x*testiter]=GAP_Algorithm(c,v,p,sequence[test],100,0.8,10,0)
                Ratio_GAP_e[test+x*testiter]=GAP_e[test+x*testiter]/LP[test+x*testiter]
                
                
            np.random.seed(1)

            for test in range(testiter):
                c=np.array([k,k,k,k,k,k,k,k,k,k],dtype=np.float64)


                GAP_baseline[test+x*testiter]=GAP_Algorithm(c,v,p,sequence[test],100,1-1/(int(k)**(1/2)),10,0)
                Ratio_GAP_baseline[test+x*testiter]=GAP_baseline[test+x*testiter]/LP[test+x*testiter]
                
                


                Greedy[test+x*testiter]=Greedy_Algorithm(v,c,sequence[test])
                
                Ratio_Greedy[test+x*testiter]=Greedy[test+x*testiter]/LP[test+x*testiter]



        p_x=np.mean(GAP_f)
        p_y=np.mean(GAP_e)
        p_z=np.mean(GAP_baseline)
        p_w=np.mean(Greedy)
        p_l=np.mean(LP)
        p_f=np.mean(FF)

        R_baseline[int((k-2))]=p_z
        R_GAP_f[int(k-2)]=p_x
        R_GAP_e[int(k-2)]=p_y
        R_Greedy[int(k-2)]=p_w
        R_LP[int(k-2)]=p_l
        R_FF[int(k-2)]=p_f
        
        Ra_GAP_f[int((k-2))]=np.mean(Ratio_GAP_f)
        Ra_GAP_e[int(k-2)]=np.mean(Ratio_GAP_e)
        Ra_Greedy[int(k-2)]=np.mean(Ratio_Greedy)
        Ra_baseline[int(k-2)]=np.mean(Ratio_GAP_baseline)
        Ra_FF[int(k-2)]=np.mean(Ratio_FF)
        
        S_GAP_f[int(k-2)]=np.std(Ratio_GAP_f)
        S_GAP_e[int(k-2)]=np.std(Ratio_GAP_e)
        S_Greedy[int(k-2)]=np.std(Ratio_Greedy)
        S_baseline[int(k-2)]=np.std(Ratio_GAP_baseline)
        S_FF[int(k-2)]=np.std(Ratio_FF)



    f1=open(dis+'-K1f.txt','w+')
    f1.write('K FF LP'+'\n')
    for i in range(len_K):
        f1.write(str(2+i)+' ')
        f1.write(str(R_baseline[i])+' ')
        f1.write(str(R_GAP_f[i])+' ')
        f1.write(str(R_GAP_e[i])+' ')
        f1.write(str(R_Greedy[i])+' ')
        f1.write(str(R_FF[i])+' ')
        f1.write(str(R_LP[i])+'\n')
    f1.close()
    
        
    f2=open(dis+'-ratio-K1f','w+')
    f2.write('K FF'+'\n')
    for i in range(len_K):
        f2.write(str(2+i)+' ')
        f2.write(str(Ra_baseline[i])+'_'+str(S_baseline[i])+' ')
        f2.write(str(Ra_GAP_f[i])+'_'+str(S_GAP_f[i])+' ')
        f2.write(str(Ra_GAP_e[i])+'_'+str(S_GAP_e[i])+' ')
        f2.write(str(Ra_Greedy[i])+'_'+str(S_Greedy[i])+' ')
        f2.write(str(Ra_FF[i])+'_'+str(S_FF[i])+'\n')
    f2.close()