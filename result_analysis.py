import glob
import pickle
from fedbase.utils.visualize import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# file_list = glob.glob('./log/local*cifar10*dirichlet_1*')
# file_list = glob.glob('./log/central*fashion*')
dataset = 'fashion'
noniid = 'dirichlet_0.5'

for K in [3,5,10]:
    # file_list = glob.glob('./log/cfl_'+str(K)+'*fashion*dirichlet_1*')
    file_list = glob.glob('./log/cfl_'+str(K)+'*'+dataset+'*'+ noniid +'*')
    acc= []
    for i in file_list:
        with open(i, 'rb') as f:
            log = pickle.load(f)
            print(len(log['server']))
            # acc.append(log['node']['0'])
            acc.append(log['server'])
    acc_df = pd.DataFrame(acc)
    print(acc_df)
    acc_df_n = pd.DataFrame()
    for i in acc_df.columns:
        if (i+1)%(K+1) ==0:
            acc_df_tmp = acc_df[[i]]
            acc_df_tmp['round'] = (i+1)//(K+1)
            acc_df_tmp.rename(columns={i : 'acc'}, inplace=True)
            acc_df_n = pd.concat([acc_df_n, acc_df_tmp], axis=0)
    print(acc_df_n)
    sns.lineplot(x=acc_df_n["round"], y=acc_df_n["acc"], label = 'cfl_' + str(K))

for method in ['fedavg']:
    file_list = glob.glob('./log/'+ method + '_' +'*'+dataset+'*'+ noniid +'*')
    acc= []
    for i in file_list:
        with open(i, 'rb') as f:
            log = pickle.load(f)
            print(len(log['server']))
            # acc.append(log['node']['0'])
            acc.append(log['server'])
    acc_df = pd.DataFrame(acc)
    print(acc_df)
    acc_df_n = pd.DataFrame()
    for i in acc_df.columns:
        acc_df_tmp = acc_df[[i]]
        acc_df_tmp['round'] = i+1
        acc_df_tmp.rename(columns={i : 'acc'}, inplace=True)
        acc_df_n = pd.concat([acc_df_n, acc_df_tmp], axis=0)
    print(acc_df_n)
    sns.lineplot(x=acc_df_n["round"], y=acc_df_n["acc"], label = method)

sns.set_theme(style="darkgrid")
plt.show()