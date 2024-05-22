#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib
import time
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import os
my_path = os.path.dirname(os.path.abspath('demand learning'))


def plot_errbar(filename):
    # 'OFF', 'GRD', 'SAM-MAX', 'SAM-LP', 'SAM-MIX'
    colors = ['b', 'g', 'r', 'violet', 'b', 'violet', 'violet']
    markers = ['x', '^', 'o', '*', 'o', '*', 'o']
    fullpath = my_path+'\\'+filename
    with open(fullpath) as f:
        first_line = f.readline().strip().split()
        if first_line[0] == 'K':
            xlabel = 'k'
            legend_loc = 'upper right'
        if first_line[0] == 'T0':
            xlabel = 'T0'
            legend_loc = 'upper left'
        xlabel = first_line[0]
        algo_name_list = first_line[1:]
        res = [[] for algo in algo_name_list]
        err = [[] for algo in algo_name_list]
        x = []
        for l in f:
            line = l.strip().split()
            x.append(float(line[0]))
            for i in range(len(algo_name_list)):
                r = line[i+1].split('_')
                res[i].append(float(r[0]))
                err[i].append(float(r[1]))
        print(x)
        print(res)
        for i in range(len(algo_name_list)):
            plt.errorbar(x, res[i], yerr=err[i], fmt='o-', color=colors[i], marker = markers[i], ms = 5, label=algo_name_list[i], ecolor=colors[i], elinewidth=2, capsize=5)
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel('Average LP ratio', fontsize=16)
        plt.xticks(x,fontsize=16)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=12, loc=legend_loc)
        plt.show()
        plt.tight_layout()
        time_now = int(time.time())
        time_local = time.localtime(time_now)
        dt = time.strftime("%Y%m%d%H%M%S",time_local)
        plt.savefig(fullpath+dt+'.eps', format='eps')
        #plt.savefig(fullpath+dt+'.png', format='png')
        plt.close()


def plot_one(filename):
    # 'GAP*', 'GAP0.5', 'GAP0.5', 'GRD'
    colors = ['b', 'g','r', 'violet', 'y', 'violet', 'g','g']
    markers = ['x', 'o', '2', 'o', 'x', 'o', 'x','o']
    fullpath = my_path+'/'+filename
    with open(fullpath) as f:
        first_line = f.readline().strip().split()
        if first_line[0] == 'K':
            xlabel = 'k'
            legend_loc = 'lower right'
        if first_line[0] == 'Epsilon':
            xlabel = r'$\epsilon$'
            legend_loc = 'upper right'
        if first_line[0] == 'T':
            xlabel = 'T'
            legend_loc = 'upper left'
        if first_line[0]=='Gamma':
            xlabel=r'$\gamma$'
            legend_loc='upper left'
        if first_line[0]=='rou':
            xlabel=r'$\rho$'
            legend_loc='upper right'
        if first_line[0]=='T0':
            xlabel=r'$T_0$'
            legend_loc='upper right'
        # xlabel = first_line[0]
        algo_name_list = first_line[1:]
        res = [[] for algo in algo_name_list]
        err = [[] for algo in algo_name_list]
        x = []
        for l in f:
            line = l.strip().split()
            x.append(float(line[0]))
            for i in range(len(algo_name_list)):
                r = line[i+1]
                # res[i].append(float(r))
                res[i].append(float(r.split('_')[0]))
                # err[i].append(float(r[1]))
        print(x)
        print(res)
        for i in range(len(algo_name_list)):
            plt.plot(x, res[i], color=colors[i], marker = markers[i], ms = 5, label=algo_name_list[i])
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel('Average LP ratio', fontsize=16)
        plt.xticks(x,fontsize=12)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=12, loc=legend_loc)
        # plt.show()
        plt.tight_layout()
        time_now = int(time.time())
        time_local = time.localtime(time_now)
        dt = time.strftime("%Y%m%d%H%M%S",time_local)
        plt.savefig(fullpath+dt+'.eps', format='eps')
        plt.savefig(fullpath+dt+'.png', format='png')
        #plt.savefig(fullpath+dt+'.eps', format='eps')
        plt.close()

# for no RCP
def plot_one_norcp(filename):
    # 'OFF', (RCP) 'GRD', 'BAT', 'SAM1', 'SAM0.5', 'SAM'
    colors = ['g', 'r', 'b', 'b', 'violet', 'violet']
    markers = ['^', 'o', '*', 'o', '*', 'o']
    with open(filename) as f:
        first_line = f.readline().strip().split()
        if first_line[0] == 'type_number':
            xlabel = 'm'
        if first_line[0] == 'density':
            xlabel = 'q'
        if first_line[0] == 'n_max':
            xlabel = r'$N^B$'
        if first_line[0] == 'p_min':
            xlabel = r'$P^G$'
        if first_line[0] == 'lam_max':
            xlabel = r'$L^P$'
        # xlabel = first_line[0]
        algo_name_list = first_line[2:]
        res = [[] for algo in algo_name_list]
        x = []
        for l in f:
            line = l.strip().split()
            x.append(float(line[0]))
            for i in range(len(algo_name_list)):
                res[i].append(float(line[i+2]))
        print(x)
        print(res)
        for i in range(len(algo_name_list)):
            if algo_name_list[i] == 'SAM1':
                continue
            plt.plot(x, res[i], color=colors[i], marker = markers[i], ms = 15, label=algo_name_list[i])
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel('Empirical Competitive Ratio', fontsize=16)
        plt.xticks(x,fontsize=16)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=12, loc='lower right')
        # plt.show()
        plt.tight_layout()
        time_now = int(time.time())
        time_local = time.localtime(time_now)
        dt = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
        plt.savefig(my_path+'/imgs/diff_'+first_line[0]+'/'+filename+dt+'.eps', format='eps')
        plt.close()

if __name__ == '__main__':
    # plot_one('ber-ratio-K1f')
    plot_one('truncnorm-ratio-Tf')
    plot_one('truncnorm-ratio-K1f')

    plot_one('uni-ratio-K1f')
    plot_one('uni-ratio-Tf')
    # plot_one('ber-ratio-Tf')
    # plot_one('ber-ratio-K1f')
    # plot_one('diffe/bere')
    # plot_one('diffe/unie')
    # plot_one('diffe/TNe')

    #plot_one('diffT/r_ber')
    #plot_one('diffT/r_uni')
    #plot_one('diffT/r_TN')

    #plot_one('diffK/r_berK')
    #plot_one('diffK/r_uniK')
    #plot_one('diffK/r_TNK')
    # plot_errbar('truncnorm-ratio-T0.txt')
    
    # plot_one('diffT/r_ber')
    # plot_one('diffT/r_uni')
    # plot_one('diffT/r_TN')

    # plot_errbar('diffT\\TN')
    # plot_errbar('diffK\\berK')
    # plot_errbar('diffK\\uniK')
    # plot_errbar('diffK\\TNK')
    # plot_errbar('t5_g10_h0.3_D2')
    # plot_errbar('t5_g10_T250_D1')
    # plot_errbar('t5_g10_T250_D2')
    # plot_errbar('t5_g10_T100_D1')
    # plot_errbar('t5_g10_T100_D2')
    # plot_errbar('temp_h0.5_D1')


# In[ ]:




