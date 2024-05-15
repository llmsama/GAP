import numpy as np
import random
import gurobipy as gp
from gurobipy import Model,quicksum,GRB
import matplotlib.pyplot as plt
import time
import scipy.stats as stats
from Algorithm import *


#run GAP Algorithm with k=5,k=10,T=500,T=1000 and gamma=1-1/sqrt(k) for different epsilon from 0 to 0.5, where T is the arrival sequence length and k is the capacity of all the bins.

testnum=20
testiter=5

Epsilon=np.arange(0,0.55,0.05)
len_E=len(Epsilon)

#used to save the rewards of GAP algorithm in each iteration
GAP_1=np.zeros(testnum*testiter)
GAP_2=np.zeros(testnum*testiter)
GAP_3=np.zeros(testnum*testiter)
GAP_4=np.zeros(testnum*testiter)

#used to save the rewards of LP algorithm in each iteration
LP_1=np.zeros(testnum*testiter)
LP_2=np.zeros(testnum*testiter)
LP_3=np.zeros(testnum*testiter)
LP_4=np.zeros(testnum*testiter)

#used to save the LP ratio in each iteration
Ratio_GAP_1=np.zeros(testnum*testiter)
Ratio_GAP_2=np.zeros(testnum*testiter)
Ratio_GAP_3=np.zeros(testnum*testiter)
Ratio_GAP_4=np.zeros(testnum*testiter)

#used to save the average lp ratio
Ra_GAP_1=np.zeros(len_E)
Ra_GAP_2=np.zeros(len_E)
Ra_GAP_3=np.zeros(len_E)
Ra_GAP_4=np.zeros(len_E)

#used to save the average rewards
R_GAP_1=np.zeros(len_E)
R_GAP_2=np.zeros(len_E)
R_GAP_3=np.zeros(len_E)
R_GAP_4=np.zeros(len_E)

c_1=np.array([5,5,5,5,5,5,5,5,5,5],dtype=np.float64)
c_2=np.array([10,10,10,10,10,10,10,10,10,10],dtype=np.float64)


for e in Epsilon:
    
    for x in range(testnum):
        
        p=generate_p(10)
        v=generate_v(10,10)
        parameter=generate_ber(10)
        
        for test in range(testiter):

            sequence_1=simulation_ber(p,parameter,10,500)
            sequence_2=simulation_ber(p,parameter,10,1000)
            s_1=sequence_1[:,1]
            s_2=sequence_2[:,1]
        

            GAP_1[test+x*testiter]=GAP_Algorithm(c_1,v,p,sequence_1,100,1-1/(5**(1/2)),10,e)
            LP_1[test+x*testiter]=solve_OPT(c_1,v,p,s_1,sequence_1)[0]

            
            GAP_2[test+x*testiter]=GAP_Algorithm(c_1,v,p,sequence_2,100,1-1/(5**(1/2)),10,e)
            LP_2[test+x*testiter]=solve_OPT(c_1,v,p,s_2,sequence_2)[0]
        
            GAP_3[test+x*testiter]=GAP_Algorithm(c_2,v,p,sequence_1,100,1-1/(10**(1/2)),10,e)
            LP_3[test+x*testiter]=solve_OPT(c_2,v,p,s_1,sequence_1)[0]

            GAP_4[test+x*testiter]=GAP_Algorithm(c_2,v,p,sequence_2,100,1-1/(10**(1/2)),10,e) 
            LP_4[test+x*testiter]=solve_OPT(c_2,v,p,sequence_2)[0]

            Ratio_GAP_1[test+x*testiter]=GAP_1[test+x*testiter]/LP_1[test+x*testiter]
            Ratio_GAP_2[test+x*testiter]=GAP_2[test+x*testiter]/LP_2[test+x*testiter]
            Ratio_GAP_3[test+x*testiter]=GAP_3[test+x*testiter]/LP_3[test+x*testiter]
            Ratio_GAP_4[test+x*testiter]=GAP_4[test+x*testiter]/LP_4[test+x*testiter]

        
            print(GAP_1[test+x*testiter],GAP_2[test+x*testiter],GAP_3[test+x*testiter],GAP_4[test+x*testiter])
        
    p_x=np.mean(GAP_1)
    p_y=np.mean(GAP_2)
    p_z=np.mean(GAP_3)
    p_w=np.mean(GAP_4)

    R_GAP_1[int((e/0.05))]=p_x
    R_GAP_2[int(e/0.05)]=p_y
    R_GAP_3[int(e/0.05)]=p_z
    R_GAP_4[int(e/0.05)]=p_w

    Ra_GAP_1[int((e/0.05))]=np.mean(Ratio_GAP_1)
    Ra_GAP_2[int((e/0.05))]=np.mean(Ratio_GAP_2)
    Ra_GAP_3[int((e/0.05))]=np.mean(Ratio_GAP_3)
    Ra_GAP_4[int((e/0.05))]=np.mean(Ratio_GAP_4)
    

