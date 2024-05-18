import numpy as np
import random
import gurobipy as gp
from gurobipy import Model,quicksum,GRB
import matplotlib.pyplot as plt
import time
import scipy.stats as stats
from Data_generation import *


#The Lp Algorithm

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
    return M.objVal, result

#main part of the GAP Algorithm
#c is the capacity of bins
#v is the reward 
#p is the probability of each type
#sequence is the arriving sequence
#D is the discretization parameter used to disretize the continuous interval

def GAP_Algorithm(c,v,p,sequence,empty_run,gamma,D,epsilon):
    m=np.shape(c)[0]
    Total_type=np.shape(p)[0]
    s=np.zeros(Total_type) #s is the sample mean for each type
    sample_n=np.zeros(Total_type) #save the type number in empty run
    c_left=c.copy()
    v_total=0
    
    
    for run in range(empty_run):

        t0=int(sequence[run][0])
        index_set=np.argwhere(c_left=max(c_left)) #find the index of which bins have the largest capacity
        candidate=np.argmax(v[t0][index_set])
        c_M=index_set[candidate] #choose the bin with largest reward from those who have largest capacity
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
    v_total=0
    plan=np.zeros((T,m))
    F=[]
    
    bin_list=list(range(m))
    bin_list.append(-1)
    
    for j in range(m):
        F.append(np.ones((2,int(D*c[j]))))
    
    for i in range(T):
        t0=int(new_sequence[i][0])
        p_t0=x[t0]/(p[t0]*T)
        p_t0=np.append(p_t0,max(0,1-sum(p_t0)))
        
        j_star=np.random.choice(bin_list,p=p_t0)
        
        if j_star==-1:
            continue
        
        X=np.zeros((m,D))
        theta=np.zeros(m)
        accept=np.zeros(m)
        lost=0
        
        
        if i==0:

            if np.sum(F[j_star][i%2]>=gamma)=0:
                theta[j_star]=int(D*c_left[j]-1)
            else:
                theta[j_star]=int(np.argwhere(F[j_star][i%2]>=gamma)[0])

            if w[j_star]<(int(theta[j_star])/D):
                accept[j_star]=1
            elif w[j_star]==(int(theta[j_star])/D):
                p0=gamma/F[j_star][i%2][int(theta[j_star])]
                accept[j_star]=np.random.choice([1,0],p=[p0,1-p0])

                
            if accept[j_star]==1:
                lost=new_sequence[i][1]
                if w[j_star]+lost<=c[j_star]:
                    w[j_star]=w[j_star]+lost
                    v_total=v_total+v[t0][j_star]
                    plan[i][j_star]=1
                else:
                    accept[j_star]=0
        
        else:
            for j in range(m):
                for k in range(int(D*c[j])):
                    F[j][i%2][k]=F[j][(i-1)%2][k]-G(F[j][(i-1)%2],k,gamma)
                    for y in range(D):
                        F[j][i%2][k]=F[j][i%2][k]+(CDF_v[j][y+1]-CDF_v[j][y])*G(F[j][(i-1)%2],k-y-1,gamma)
                    F[j][i%2][k]=F[j][i%2][k]+CDF_v[j][0]*G(F[j][(i-1)%2],k,gamma)

            
            
            if np.sum(F[j_star][i%2]>=gamma)==0:
                theta[j_star]=int(D*c[j]-1)
            else :
                theta[j_star]=int(np.argwhere(F[j_star][i%2]>=gamma)[0])



            if w[j_star]<(int(theta[j_star])/D):
                accept[j_star]=1
                    
            elif w[j_star]==(int(theta[j_star])/D):
                if int(theta[j_star])==0:
                    p0=gamma/F[j_star][i%2][int(theta[j_star])]
                else:
                    p0=(gamma-F[j_star][i%2][int(theta[j_star])-1])/(F[j_star][i%2][int(theta[j_star])]-F[j_star][i%2][int(theta[j_star])-1])
                accept[j_star]=np.random.choice([1,0],p=[p0,1-p0])
            

            if accept[j_star]==1:
                lost=new_sequence[i][1]
                if w[j_star]+lost<=c[j_star]:
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
    total_reward=0 #the reward after assigning each arrival using greedy Algorithm
    plan=np.zeros([T,Total_bin]) #the assining plan
    
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
        
    return total_reward


#FirstFit Algorithm
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


