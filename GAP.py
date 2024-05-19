#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import gurobipy as gp
from gurobipy import Model,quicksum,GRB
import matplotlib.pyplot as plt
import time
import scipy.stats as stats
#from Data-generation import *

#GAP Algorithm

#solving the Optimization Problem first
def solve_OPT(c,v,p,s,sequence):
    m=np.shape(c)[0]
    T=np.shape(sequence)[0]
    Total_type=np.shape(p)[0]
    
    M=gp.Model()
    
    x=M.addVars(Total_type,m,lb=0,name='x')
    M.addConstrs(quicksum(s[t]*x[t,j] for t in range(Total_type))<=c[j] for j in range(m))
    M.addConstrs(quicksum(x[t,j] for j in range(m))<=p[t]*T for t in range(Total_type))
    
    M.setObjective(quicksum(v[t][j]*x[t,j] for j in range(m) for t in range(Total_type)),GRB.MAXIMIZE)
    
    M.optimize()
    
    result=[]
    for v in M.getVars():
        if v.Varname[0]=='x':
            result.append(v.X)
    return M.objVal,result
#main part of the GAP Algorithm
def GAP_Algorithm(c,v,p,sequence,empty_run,gamma,D,epsilon):
    m=np.shape(c)[0]
    Total_type=np.shape(p)[0]
    s=np.zeros(Total_type)#s is the sample mean for each type
    sample_n=np.zeros(Total_type)#save the type number in empty run
    c_left=c.copy()
    v_total=0
    
    for run in range(empty_run):
        
        t0=int(sequence[run][0])
        index_set=np.argwhere(c_left==max(c_left)) #所有bin中剩下的capacity最多的bin的index set
        candidate=np.argmax(v[t0][index_set])
        c_M=index_set[candidate] #bin中剩下capacity最多且reward最大的bin的index
        if c_left[c_M]>=1:
            c_left[c_M]=c_left[c_M]-sequence[run][1]
            v_total=v_total+v[t0][c_M]
            sample_n[t0]=sample_n[t0]+1
            s[t0]=s[t0]+sequence[run][1]
    
        


    
    s=s/sample_n
    
    
    new_sequence=sequence[empty_run:]
    empty_sequence=sequence[:empty_run]
    T=np.shape(new_sequence)[0]

    x=np.array(solve_OPT(c_left,v,p,s,new_sequence)[1]).reshape(Total_type,m)

    CDF_v=np.ones((m,D+1))
    for j in range(m):
        for y in range(D+1):
            CDF_v[j][y]=X_distribution(x,j,y/D,empty_sequence,epsilon)
            
    w=np.zeros(m)

    plan=np.zeros((T,m))
    F=[]
    
    bin_list=list(range(m))
    bin_list.append(-1)
    
    for j in range(m):
        F.append(np.ones((2,int(D*c_left[j]))))

    
    for i in range(T):
        t0=int(new_sequence[i][0])
        p_t0=x[t0]/(p[t0]*T)
        p_t0=np.append(p_t0,max(0,1-sum(p_t0)))
        #np.random.seed(1)
        j_star=np.random.choice(bin_list,p=p_t0)
        
        if j_star==-1:
            continue
        
        if len(F[j_star][i%2])>0:
        
            X=np.zeros((m,D))
            theta=np.zeros(m)
            accept=np.zeros(m)
            lost=0




            if i==0:

                if np.sum(F[j_star][i%2]>=gamma)==0:
                    theta[j_star]=int(D*c_left[j]-1)
                else :
                    theta[j_star]=int(np.argwhere(F[j_star][i%2]>=gamma)[0])

                if w[j_star]<(int(theta[j_star])/D):
                    accept[j_star]=1
                elif w[j_star]==(int(theta[j_star])/D):
                    p0=gamma/F[j_star][i%2][int(theta[j_star])]
                    #np.random.seed(1)
                    accept[j_star]=np.random.choice([1,0],p=[p0,1-p0])


                if accept[j_star]==1:
                    lost=new_sequence[i][1]
                    if w[j_star]+lost<=c_left[j_star]:
                        w[j_star]=w[j_star]+lost
                        v_total=v_total+v[t0][j_star]
                        plan[i][j_star]=1
                    else:
                        accept[j_star]=0

            else:

                for j in range(m):
                    for k in range(int(D*c_left[j])):
                        F[j][i%2][k]=F[j][(i-1)%2][k]-G(F[j][(i-1)%2],k,gamma)
                        for y in range(D):
                            F[j][i%2][k]=F[j][i%2][k]+(CDF_v[j][y+1]-CDF_v[j][y])*G(F[j][(i-1)%2],k-y-1,gamma)
                        F[j][i%2][k]=F[j][i%2][k]+CDF_v[j][0]*G(F[j][(i-1)%2],k,gamma)






                if np.sum(F[j_star][i%2]>=gamma)==0:
                    theta[j_star]=int(D*c_left[j]-1)
                else :
                    theta[j_star]=int(np.argwhere(F[j_star][i%2]>=gamma)[0])



                if w[j_star]<(int(theta[j_star])/D):
                    accept[j_star]=1

                elif w[j_star]==(int(theta[j_star])/D):
                    if int(theta[j_star])==0:
                        p0=min(gamma/F[j_star][i%2][int(theta[j_star])],1)
                    else:
                        p0=(gamma-F[j_star][i%2][int(theta[j_star])-1])/(F[j_star][i%2][int(theta[j_star])]-F[j_star][i%2][int(theta[j_star])-1])
                    #np.random.seed(1)
                    accept[j_star]=np.random.choice([1,0],p=[p0,1-p0])


                if accept[j_star]==1:
                    lost=new_sequence[i][1]
                    if w[j_star]+lost<=c_left[j_star]:
                        w[j_star]=w[j_star]+lost
                        v_total=v_total+v[t0][j_star]
                        plan[i][j_star]=1
                    else:
                        accept[j_star]=0

        
        
    return v_total
#update of the empirical cumulative distribution function
def X_distribution(x,j,y,sequence,epsilon):
    T=np.shape(sequence)[0]
    Total_type=np.shape(x)[0]

    
    outcome=0
    

    for t in range(Total_type):
        CDF=0
        count=0
        for time in range(T):
            if sequence[time][0] == t:
                count=count+1
                if sequence[time][1]<=y:
                    CDF=CDF+1
                    
        if count==0:
            CDF=0
        else:
            CDF=CDF/count

        if y<1:
            CDF=max(0,CDF-epsilon)
        outcome=outcome+(x[t][j]/T)*CDF
    outcome=outcome+1-(x.sum(axis=0)[j]/T)
    
    return outcome
#function G
def G(F,w,gamma):
    if w<0:
        return 0
    else:
        return min(F[w],gamma)


#Greedy Algorithm
def Greedy_Algorithm(v,c,sequence):
    T=np.shape(sequence)[0]
    Total_bin=np.shape(c)[0]
    c_left=c.copy()
    total_reward=0
    plan=np.zeros([T,Total_bin])
    for i in range(T):
        if max(c_left)<1:
            break
        reward_max=0
        bin_number=0
        for j in range(Total_bin):
            if (c_left[j]>=1) and (v[int(sequence[i][0])][j]>reward_max):
                reward_max=v[int(sequence[i][0])][j]
                bin_number=j
        
        total_reward=total_reward+reward_max
        c_left[bin_number]=c_left[bin_number]-sequence[i][1]
            
        plan[i][bin_number]=1
    #print('c_left: ',plan)
        
    return total_reward

def First_Fit(v,c,sequence):
    T=np.shape(sequence)[0]
    Total_bin=np.shape(c)[0]
    c_left=c.copy()
    total_reward=0
    
    for i in range(T):
        if max(c_left)<1:
            break
        bin_number=0
        for j in range(Total_bin):
            if c_left[j]>=1:
                bin_number=j
                break
        total_reward=total_reward+v[int(sequence[i][0])][int(bin_number)]
        c_left[bin_number]=c_left[bin_number]-sequence[i][1]
        
    return total_reward


# In[2]:


#generate arrivale propablity, Total_type is the number of customer types.
def generate_p(Total_type):
    p=np.zeros(Total_type)
    seed=np.zeros(Total_type)
    for i in range(Total_type):
        seed[i]=np.random.choice(range(1,Total_type+1),p=np.ones(Total_type)/Total_type)
    for i in range(Total_type):
        p[i]=seed[i]/sum(seed)
    return p
#generate the reward, Total_type is the number of customer types, Total_bin is the number of bins.
def generate_v(Total_type,Total_bin):
    v=np.zeros((Total_type,Total_bin))
    for i in range(Total_type):
        for j in range(Total_bin):
            v[i][j]=random.uniform(10,20)
    return v



#generate different types of demand following bernoulli distribution
def generate_ber(Total_type):
    parameter=np.zeros(Total_type)
    for i in range(Total_type):
        p=random.uniform(0,1)
        parameter[i]=p
    return parameter
#generate a sequence of bernoulli arrival, with total length T
def simulation_ber(p,parameter,Total_type,T):
    sequence=np.zeros((T,2))
    #np.random.seed(1)
    for i in range(T):
        t0=np.random.choice(range(Total_type),p=p)
        x=np.random.binomial(1,parameter[t0])
        sequence[i][0]=t0
        sequence[i][1]=x
    return sequence

#generate different types of demand following uniform distribution, c is the upper bound of parameter in uniform distribution.
def generate_uni(Total_type,c):
    parameter=np.zeros((Total_type,2))
    for i in range(Total_type):
        a=0
        b=0
        while a==b:
            a=random.uniform(0,c)
            b=random.uniform(0,c)
        parameter[i][0]=a
        parameter[i][1]=b
    return parameter
#generate a sequence of uniform arrival, with total length T
def simulation_uni(p,parameter,Total_type,T):
    sequence=np.zeros((T,2))
    np.random.seed(1)
    random.seed(1)
    for i in range(T):
        t0=np.random.choice(range(Total_type),p=p)
        x=random.uniform(parameter[t0][0],parameter[t0][1])
        sequence[i][0]=t0
        sequence[i][1]=x
    return sequence

#generate different types of demand following truncated normal distribution
def generate_truncnorm(Total_type):
    parameter=np.zeros((Total_type,2))
    for i in range(Total_type):
        mu=random.uniform(0,1)
        sigma=random.uniform(0,0.5)
        parameter[i][0]=mu
        parameter[i][1]=sigma
    return parameter
#generate a sequence of truncated normal arrival, with total length T
def simulation_truncnorm(p,parameter,Total_type,T):
    sequence=np.zeros((T,2))
    np.random.seed(1)
    random.seed(1)
    for i in range(T):
        t0=np.random.choice(range(Total_type),p=p)
        X=stats.truncnorm(-parameter[t0][0]/parameter[t0][1],(1-parameter[t0][0])/parameter[t0][1],loc=parameter[t0][0],scale=parameter[t0][1])
        sequence[i][0]=t0
        sequence[i][1]=X.rvs(1)[0]
    return sequence


# In[3]:


def run_T(dis='ber'):
    testnum=20
    testiter=5
    
    Time=np.arange(500,1001,100)
    len_Time = len(Time)

    GAP_f=np.zeros(testnum*testiter)
    GAP_e=np.zeros(testnum*testiter)
    GAP_baseline=np.zeros(testnum*testiter)
    Greedy=np.zeros(testnum*testiter)
    FF=np.zeros(testnum*testiter)
    LP=np.zeros(testnum*testiter)
    
    Ratio_GAP_f=np.zeros(testnum*testiter)
    Ratio_GAP_e=np.zeros(testnum*testiter)
    Ratio_GAP_baseline=np.zeros(testnum*testiter)
    Ratio_Greedy=np.zeros(testnum*testiter)
    Ratio_FF=np.zeros(testnum*testiter)
    

    R_GAP_f=np.zeros(len_Time)
    R_GAP_e=np.zeros(len_Time)
    R_Greedy=np.zeros(len_Time)
    R_baseline=np.zeros(len_Time)
    R_FF=np.zeros(len_Time)
    R_LP=np.zeros(len_Time)
    
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
            
        
            


# In[ ]:





# In[4]:


def run_k(dis='ber'):
    testnum=20
    testiter=5

    GAP_f=np.zeros(testnum*testiter)
    GAP_e=np.zeros(testnum*testiter)
    GAP_baseline=np.zeros(testnum*testiter)
    Greedy=np.zeros(testnum*testiter)
    FF=np.zeros(testnum*testiter)
    LP=np.zeros(testnum*testiter)

    K=np.arange(2,11,1)
    len_K = len(K)
    
    R_GAP_f=np.zeros(len_K)
    R_GAP_e=np.zeros(len_K)
    R_Greedy=np.zeros(len_K)
    R_baseline=np.zeros(len_K)
    R_FF=np.zeros(len_K)
    R_LP=np.zeros(len_K)
    
        
    Ratio_GAP_f=np.zeros(testnum*testiter)
    Ratio_GAP_e=np.zeros(testnum*testiter)
    Ratio_GAP_baseline=np.zeros(testnum*testiter)
    Ratio_Greedy=np.zeros(testnum*testiter)
    Ratio_FF=np.zeros(testnum*testiter)
    
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


# In[5]:


def run_epsilon(dis='ber'):
    testnum=20
    testiter=5

    Epsilon=np.arange(0,0.55,0.05)
    len_E=len(Epsilon)

    GAP_1=np.zeros(testnum*testiter)
    GAP_2=np.zeros(testnum*testiter)
    GAP_3=np.zeros(testnum*testiter)
    GAP_4=np.zeros(testnum*testiter)
    
    LP_1=np.zeros(testnum*testiter)
    LP_2=np.zeros(testnum*testiter)
    LP_3=np.zeros(testnum*testiter)
    LP_4=np.zeros(testnum*testiter)
    
    Ratio_GAP_1=np.zeros(testnum*testiter)
    Ratio_GAP_2=np.zeros(testnum*testiter)
    Ratio_GAP_3=np.zeros(testnum*testiter)
    Ratio_GAP_4=np.zeros(testnum*testiter)
    
    Ra_GAP_1=np.zeros(len_E)
    Ra_GAP_2=np.zeros(len_E)
    Ra_GAP_3=np.zeros(len_E)
    Ra_GAP_4=np.zeros(len_E)

    R_GAP_1=np.zeros(len_E)
    R_GAP_2=np.zeros(len_E)
    R_GAP_3=np.zeros(len_E)
    R_GAP_4=np.zeros(len_E)
    
    S_GAP_1=np.zeros(len_E)
    S_GAP_2=np.zeros(len_E)
    S_GAP_3=np.zeros(len_E)
    S_GAP_4=np.zeros(len_E)

    c_1=np.array([5,5,5,5,5,5,5,5,5,5],dtype=np.float64)
    c_2=np.array([10,10,10,10,10,10,10,10,10,10],dtype=np.float64)


    for e in Epsilon:

        for x in range(testnum):

            p=generate_p(10)
            v=generate_v(10,10)
            
            if dis=='truncnorm':
                parameter=generate_truncnorm(10)
            
            elif dis=='uni':
                parameter=generate_uni(10,1)
                
            else:
                parameter=generate_ber(10)


            for test in range(testiter):
                
                if dis=='truncnorm':
                    sequence_1=simulation_truncnorm(p,parameter,10,500)
                    sequence_2=simulation_truncnorm(p,parameter,10,1000)
                    s_1=sequence_1[:,1]
                    s_2=sequence_2[:,1]
                    
                elif dis=='uni':
                    sequence_1=simulation_uni(p,parameter,10,500)
                    sequence_2=simulation_uni(p,parameter,10,1000)
                    s_1=sequence_1[:,1]
                    s_2=sequence_2[:,1]
                    
                else:
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
                LP_4[test+x*testiter]=solve_OPT(c_2,v,p,s_2,sequence_2)[0]
                
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
        
        S_GAP_1[int((e/0.05))]=np.std(Ratio_GAP_1)
        S_GAP_2[int((e/0.05))]=np.std(Ratio_GAP_2)
        S_GAP_3[int((e/0.05))]=np.std(Ratio_GAP_3)
        S_GAP_4[int((e/0.05))]=np.std(Ratio_GAP_4)
        
    f1=open(dis+'-epsilon.txt','w+')
    f1.write('Epsilon K5T500 K5T1000 K10T500 K10T1000'+'\n')
    for i in range(len_E):
        f1.write(str(0.05*i)+' ')
        f1.write(str(R_GAP_1[i])+' ')
        f1.write(str(R_GAP_2[i])+' ')
        f1.write(str(R_GAP_3[i])+' ')
        f1.write(str(R_GAP_4[i])+'\n')
    f1.close()
    
    f2=open(dis+'-ratio-epsilon.txt','w+')
    f2.write('Epsilon K5T500 K5T1000 K10T500 K10T1000'+'\n')
    for i in range(len_E):
        f2.write(str(0.05*i)+' ')
        f2.write(str(Ra_GAP_1[i])+'_'+str(S_GAP_1[i])+' ')
        f2.write(str(Ra_GAP_2[i])+'_'+str(S_GAP_2[i])+' ')
        f2.write(str(Ra_GAP_3[i])+'_'+str(S_GAP_3[i])+' ')
        f2.write(str(Ra_GAP_4[i])+'_'+str(S_GAP_4[i])+'\n')
    f2.close()


# In[6]:


def run_gamma(dis='ber'):
    testnum=20
    testiter=5

    Gamma=np.arange(0,1.1,0.1)
    len_G=len(Gamma)

    GAP_1=np.zeros(testnum*testiter)
    GAP_2=np.zeros(testnum*testiter)
    GAP_3=np.zeros(testnum*testiter)
    GAP_4=np.zeros(testnum*testiter)
    GAP_5=np.zeros(testnum*testiter)
    GAP_6=np.zeros(testnum*testiter)
    GAP_7=np.zeros(testnum*testiter)
    GAP_8=np.zeros(testnum*testiter)
    
    LP_1=np.zeros(testnum*testiter)
    LP_2=np.zeros(testnum*testiter)
    LP_3=np.zeros(testnum*testiter)
    LP_4=np.zeros(testnum*testiter)
    LP_5=np.zeros(testnum*testiter)
    LP_6=np.zeros(testnum*testiter)
    LP_7=np.zeros(testnum*testiter)
    LP_8=np.zeros(testnum*testiter)
    
    Ratio_GAP_1=np.zeros(testnum*testiter)
    Ratio_GAP_2=np.zeros(testnum*testiter)
    Ratio_GAP_3=np.zeros(testnum*testiter)
    Ratio_GAP_4=np.zeros(testnum*testiter)
    Ratio_GAP_5=np.zeros(testnum*testiter)
    Ratio_GAP_6=np.zeros(testnum*testiter)
    Ratio_GAP_7=np.zeros(testnum*testiter)
    Ratio_GAP_8=np.zeros(testnum*testiter)
    
    Ra_GAP_1=np.zeros(len_G)
    Ra_GAP_2=np.zeros(len_G)
    Ra_GAP_3=np.zeros(len_G)
    Ra_GAP_4=np.zeros(len_G)
    Ra_GAP_5=np.zeros(len_G)
    Ra_GAP_6=np.zeros(len_G)
    Ra_GAP_7=np.zeros(len_G)
    Ra_GAP_8=np.zeros(len_G)

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


# In[7]:


def run_rou(dis='ber'):
    testnum=20
    testiter=5
    
    N=np.arange(5,21,1)
    len_N=len(N)
    
    GAP_f=np.zeros(testnum*testiter)
    GAP_e=np.zeros(testnum*testiter)
    GAP_baseline=np.zeros(testnum*testiter)
    Greedy=np.zeros(testnum*testiter)
    FF=np.zeros(testnum*testiter)
    LP=np.zeros(testnum*testiter)
    
    Ratio_GAP_f=np.zeros(testnum*testiter)
    Ratio_GAP_e=np.zeros(testnum*testiter)
    Ratio_GAP_baseline=np.zeros(testnum*testiter)
    Ratio_FF=np.zeros(testnum*testiter)
    Ratio_Greedy=np.zeros(testnum*testiter)
    

    R_GAP_f=np.zeros(len_N)
    R_GAP_e=np.zeros(len_N)
    R_baseline=np.zeros(len_N)
    R_Greedy=np.zeros(len_N)
    R_FF=np.zeros(len_N)
    R_LP=np.zeros(len_N)
    
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



    
    


# In[8]:


run_T(dis='ber')


# In[9]:


run_T(dis='truncnorm')


# In[10]:


run_T(dis='uni')


# In[16]:





# In[ ]:




