#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
import gurobipy as gp
from gurobipy import Model,quicksum,GRB
import matplotlib.pyplot as plt
import time
import scipy.stats as stats



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
def simulation_ber(p,paramter,Total_type,T):
    sequence=np.zeros((T,2))
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
    for i in range(T):
        t0=np.random.choice(range(Total_type),p=p)
        X=stats.truncnorm(-parameter[t0][0]/parameter[t0][1],(1-parameter[t0][0])/parameter[t0][1],loc=parameter[t0][0],scale=parameter[t0][1])
        sequence[i][0]=t0
        sequence[i][1]=X.rvs(1)[0]
    return sequence

