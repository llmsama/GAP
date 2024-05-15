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

#run the GAP Algorithm with gamma=0.5,0.8,1-1/sqrt(10), k=10,T=500 and n from 5 to 20, where T is the arrival sequence length and k is the capacity of all the bins, n is the number of bins. The demand distribution is chosen to be bernoulli distribution.


def run_rou(dis='ber'):
    testnum=20
    testiter=5
    
    N=np.arange(5,21,1)
    len_N=len(N)
    
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
    Ratio_FF=np.zeros(testnum*testiter)
    Ratio_Greedy=np.zeros(testnum*testiter)
    

    #used to save LP ratio in each iteration
    R_GAP_f=np.zeros(len_N)
    R_GAP_e=np.zeros(len_N)
    R_baseline=np.zeros(len_N)
    R_Greedy=np.zeros(len_N)
    R_FF=np.zeros(len_N)
    R_LP=np.zeros(len_N)
    
    #used to save average LP ratio
    Ra_GAP_f=np.zeros(len_N)
    Ra_GAP_e=np.zeros(len_N)
    Ra_baseline=np.zeros(len_N)
    Ra_FF=np.zeros(len_N)
    Ra_Greedy=np.zeros(len_N)
    
    S_GAP_f=np.zeros(len_N)
    S_GAP_e=np.zeros(len_N)
    S_Greedy=np.zeros(len_N)
    S_FF=np.zeros(len_N)
    S_baseline=np.zeros(len_N)
    
    for n in range(5,21,1):
        
        for x in range(testnum):
            sequence=np.zeros((testiter,500,2))
            p=np.loadtxt('Instances/Instance10/Bin'+str(n)+'/P/p'+str(x))
            v=np.loadtxt('Instances/Instance10/Bin'+str(n)+'/V/v'+str(x))
            
            if dis=='truncnorm':
                parameter=np.loadtxt('Instances/Instance10/Bin'+str(n)+'/Tr/tr'+str(x))
            
            elif dis=='uni':
                parameter=np.loadtxt('Instances/Instance10/Bin'+str(n)+'/Uni/uni'+str(x))
            
            elif dis=='ber':
                parameter=np.loadtxt('Instances/Instance10/Bin'+str(n)+'/Ber/ber'+str(x))
            
            np.random.seed(1)
            for test in range(testiter):
                c=10*np.ones(n,dtype=np.float64)
                if dis=='truncnorm':
                    sequence[test]=simulation_truncnorm(p,parameter,10,500)
                
                elif dis=='uni':
                    sequence[test]=simulation_uni(p,parameter,10,500)
                    
                elif dis=='ber':
                    sequence[test]=simulation_ber(p,parameter,10,500)
                
                s=sequence[test][:,1]
                LP[test+x*testiter]=solve_OPT(c,v,p,s,sequence[test])[0]
                FF[test+x*testiter]=First_Fit(v,c,sequence[test])
                Ratio_FF[test+x*testiter]=FF[test+x*testiter]/LP[test+x*testiter]
            
            np.random.seed(1)
            for test in range(testiter):
                GAP_e[test+x*testiter]=GAP_Algorithm(c,v,p,sequence[test],100,0.8,10,0)
                Ratio_GAP_e[test+x*testiter]=GAP_e[test+x*testiter]/LP[test+x*testiter]
                
            np.random.seed(1)
            for test in range(testiter):
                GAP_f[test+x*testiter]=GAP_Algorithm(c,v,p,sequence[test],100,0.5,10,0)
                Ratio_GAP_f[test+x*testiter]=GAP_f[test+x*testiter]/LP[test+x*testiter]
            
            np.random.seed(1)
            for test in range(testiter):
                GAP_baseline[test+x*testiter]=GAP_Algorithm(c,v,p,sequence[test],100,1-1/(10**(1/2)),10,0)
                Ratio_GAP_baseline[test+x*testiter]=GAP_baseline[test+x*testiter]/LP[test+x*testiter]
            
                Greedy[test+x*testiter]=Greedy_Algorithm(v,c,sequence[test])
                Ratio_Greedy[test+x*testiter]=Greedy[test+x*testiter]/LP[test+x*testiter]
        
        R_GAP_f[int((n-5))]=np.mean(GAP_f)
        R_GAP_e[int((n-5))]=np.mean(GAP_e)
        R_Greedy[int((n-5))]=np.mean(Greedy)
        R_baseline[int((n-5))]=np.mean(GAP_baseline)
        R_FF[int(n-5)]=np.mean(FF)
        
        
        Ra_GAP_f[int((n-5))]=np.mean(Ratio_GAP_f)
        Ra_GAP_e[int((n-5))]=np.mean(Ratio_GAP_e)
        Ra_Greedy[int((n-5))]=np.mean(Ratio_Greedy)
        Ra_baseline[int((n-5))]=np.mean(Ratio_GAP_baseline)
        Ra_FF[int((n-5))]=np.mean(Ratio_FF)
        
        
        S_GAP_f[int((n-5))]=np.std(Ratio_GAP_f)
        S_GAP_e[int((n-5))]=np.std(Ratio_GAP_e)
        S_Greedy[int((n-5))]=np.std(Ratio_Greedy)
        S_baseline[int((n-5))]=np.std(Ratio_GAP_baseline)
        S_FF[int(n-5)]=np.std(Ratio_FF)
        
    f1=open(dis+'-rouf.txt','w+')
    f1.write('rou FF LP'+'\n')
    for i in range(len_N):
        f1.write(str(500/(10*(i+5)))+' ')
        f1.write(str(R_GAP_f[i])+' ')
        f1.write(str(R_GAP_e[i])+' ')
        f1.write(str(R_baseline[i])+' ')
        f1.write(str(R_Greedy[i])+' ')
        f1.write(str(R_FF[i])+' ')
        f1.write(str(R_LP[i])+'\n')
    f1.close()
    
    f2=open(dis+'-ratio-rouf','w+')
    f2.write('rou FF'+'\n')
    for i in range(len_N):
        f2.write(str(500/(10*(i+5)))+' ')
        f2.write(str(Ra_GAP_f[i])+'_'+str(S_GAP_f[i])+' ')
        f2.write(str(Ra_GAP_e[i])+'_'+str(S_GAP_e[i])+' ')
        f2.write(str(Ra_baseline[i])+'_'+str(S_baseline[i])+' ')
        f2.write(str(Ra_Greedy[i])+'_'+str(S_Greedy[i])+'\n')
        f2.write(str(Ra_FF[i])+'_'+str(S_FF[i])+'\n')
    f2.close()