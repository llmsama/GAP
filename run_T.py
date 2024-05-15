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

#run the GAP Algorithm with gamma=0.5,0.8,1-1/sqrt(10), k=10 and T from 500 to 1000, where T is the arrival sequence length and k is the capacity of all the bins. The demand distribution is chosen to be bernoulli distribution.



def run_T(dis='ber'):
    testnum=20
    testiter=5
    
    Time=np.arange(500,1001,100)
    len_Time = len(Time)

    #used to save rewards in each iteration
    GAP_f=np.zeros(testnum*testiter)
    GAP_e=np.zeros(testnum*testiter)
    GAP_baseline=np.zeros(testnum*testiter)
    Greedy=np.zeros(testnum*testiter)
    FF=np.zeros(testnum*testiter)
    LP=np.zeros(testnum*testiter)
    
    #used to save average reward
    Ratio_GAP_f=np.zeros(testnum*testiter)
    Ratio_GAP_e=np.zeros(testnum*testiter)
    Ratio_GAP_baseline=np.zeros(testnum*testiter)
    Ratio_Greedy=np.zeros(testnum*testiter)
    Ratio_FF=np.zeros(testnum*testiter)
    

    #used to save LP ratio in each iteration
    R_GAP_f=np.zeros(len_Time)
    R_GAP_e=np.zeros(len_Time)
    R_Greedy=np.zeros(len_Time)
    R_baseline=np.zeros(len_Time)
    R_FF=np.zeros(len_Time)
    R_LP=np.zeros(len_Time)
    
    #used to save average LP ratio
    Ra_GAP_f=np.zeros(len_Time)
    Ra_GAP_e=np.zeros(len_Time)
    Ra_Greedy=np.zeros(len_Time)
    Ra_baseline=np.zeros(len_Time)
    Ra_FF=np.zeros(len_Time)
    
    S_GAP_f=np.zeros(len_Time)
    S_GAP_e=np.zeros(len_Time)
    S_Greedy=np.zeros(len_Time)
    S_baseline=np.zeros(len_Time)
    S_FF=np.zeros(len_Time)
    
    T_FF=np.zeros(len_Time)
    time_FF=np.zeros(testnum*testiter)



    for T in range(500,1001,100):
        
        if T==900:
            T=901
        
        
    
        for x in range(testnum):
            sequence=np.zeros((testiter,T,2))
            p=np.loadtxt('Instances/Instance10/Bin10/P/p'+str(x))
            v=np.loadtxt('Instances/Instance10/Bin10/V/v'+str(x))
            #p=generate_p(10)
            #v=generate_v(10,10)
            if dis=='truncnorm':
                parameter=np.loadtxt('Instances/Instance10/Bin10/Tr/tr'+str(x))
                #parameter=generate_truncnorm(10)
                
            
            elif dis=='uni':
                parameter=np.loadtxt('Instances/Instance10/Bin10/Uni/uni'+str(x))
                #parameter=generate_uni(10,1)
                
            elif dis=='ber':
                parameter=np.loadtxt('Instances/Instance10/Bin10/Ber/ber'+str(x))
                #parameter=generate_ber(10)
            #print(parameter)
            np.random.seed(1)

            for test in range(testiter):
                c=np.array([10,10,10,10,10,10,10,10,10,10],dtype=np.float64)
                if dis=='truncnorm':
                    sequence[test]=simulation_truncnorm(p,parameter,10,T)
                    #print(np.shape(sequence)[0])
                    
                elif dis=='uni':               
                    sequence[test]=simulation_uni(p,parameter,10,T)
                   # print(np.shape(sequence)[0])
                
                elif dis=='ber':
                    sequence[test]=simulation_ber(p,parameter,10,T)
                    #print(np.shape(sequence)[0])
                

                    
                s=sequence[test][:,1]
                time_s=time.time()
                FF[test+x*testiter]=First_Fit(v,c,sequence[test])
                time_e=time.time()
                
                time_FF[test+x*test]=time_e-time_s
                LP[test+x*testiter]=solve_OPT(c,v,p,s,sequence[test])[0]
                Ratio_FF[test+x*testiter]=FF[test+x*testiter]/LP[test+x*testiter]
            
            np.random.seed(1)
            for test in range(testiter):
                

                GAP_f[test+x*testiter]=GAP_Algorithm(c,v,p,sequence[test],100,0.5,10,0)
                Ratio_GAP_f[test+x*testiter]=GAP_f[test+x*testiter]/LP[test+x*testiter]
            
            np.random.seed(1)

            for test in range(testiter):
                


                GAP_e[test+x*testiter]=GAP_Algorithm(c,v,p,sequence[test],100,0.8,10,0)
                Ratio_GAP_e[test+x*testiter]=GAP_e[test+x*testiter]/LP[test+x*testiter]
            
            np.random.seed(1)
            for test in range(testiter):
                

                GAP_baseline[test+x*testiter]=GAP_Algorithm(c,v,p,sequence[test],100,1-1/(10**(1/2)),10,0)
                Ratio_GAP_baseline[test+x*testiter]=GAP_baseline[test+x*testiter]/LP[test+x*testiter]



                Greedy[test+x*testiter]=Greedy_Algorithm(v,c,sequence[test]) 
                Ratio_Greedy[test+x*testiter]=Greedy[test+x*testiter]/LP[test+x*testiter]

        
        p_x=np.mean(GAP_f)
        p_y=np.mean(GAP_e)
        p_z=np.mean(GAP_baseline)
        p_w=np.mean(Greedy)
        p_l=np.mean(LP)
        p_f=np.mean(FF)

        R_baseline[int((T-500)/100)]=p_z
        R_GAP_f[int((T-500)/100)]=p_x
        R_GAP_e[int((T-500)/100)]=p_y
        R_Greedy[int((T-500)/100)]=p_w
        R_LP[int((T-500)/100)]=p_l
        R_FF[int((T-500)/100)]=p_f
        T_FF[int((T-500)/100)]=np.mean(time_FF)
        
        Ra_GAP_f[int((T-500)/100)]=np.mean(Ratio_GAP_f)
        Ra_GAP_e[int((T-500)/100)]=np.mean(Ratio_GAP_e)
        Ra_Greedy[int((T-500)/100)]=np.mean(Ratio_Greedy)
        Ra_baseline[int((T-500)/100)]=np.mean(Ratio_GAP_baseline)
        Ra_FF[int((T-500)/100)]=np.mean(Ratio_FF)
        
        S_GAP_f[int((T-500)/100)]=np.std(Ratio_GAP_f)
        S_GAP_e[int((T-500)/100)]=np.std(Ratio_GAP_e)
        S_Greedy[int((T-500)/100)]=np.std(Ratio_Greedy)
        S_baseline[int((T-500)/100)]=np.std(Ratio_GAP_baseline)
        S_FF[int((T-500)/100)]=np.std(Ratio_FF)


    f1=open(dis+'-Tf.txt','w+')
    f1.write('T FF LP'+'\n')
    for i in range(len_Time):
        f1.write(str(500+i*100)+' ')
        f1.write(str(R_baseline[i])+' ')
        f1.write(str(R_GAP_f[i])+' ')
        f1.write(str(R_GAP_e[i])+' ')
        f1.write(str(R_Greedy[i])+' ')
        f1.write(str(R_FF[i])+' ')
        f1.write(str(R_LP[i])+'\n')
    f1.close()
    
    f2=open(dis+'-ratio-Tf','w+')
    f2.write('T FF'+'\n')
    for i in range(len_Time):
        f2.write(str(500+i*100)+' ')
        f2.write(str(Ra_baseline[i])+'_'+str(S_baseline[i])+' ')
        f2.write(str(Ra_GAP_f[i])+'_'+str(S_GAP_f[i])+' ')
        f2.write(str(Ra_GAP_e[i])+'_'+str(S_GAP_e[i])+' ')
        f2.write(str(Ra_Greedy[i])+'_'+str(S_Greedy[i])+' ')
        f2.write(str(Ra_FF[i])+'_'+str(R_FF[i])+'\n')
    f2.close()
            
        
            