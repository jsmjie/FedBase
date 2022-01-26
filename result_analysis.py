import glob
import pickle
from fedbase.utils.visualize import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# # change name
# # Function to rename multiple files
# def main():   
#     folder = "./log/log_groupwise/"
#     for count, filename in enumerate(os.listdir(folder)):
#         # dst = f"Hostel {str(count)}.jpg"
#         src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
#         # dst =f"{folder}/{dst}"
         
#         # rename() function will
#         # rename all the files
#         try:
#             os.rename(src, src.replace('cfl_fashion_mnist_5', 'WCFL_5_fashion_mnist_5'))
#         except:
#             pass
# # Driver Code
# main()
# print(a)

# folder = './log/log_groupwise/'
folder = './log/'
for dataset in ['fashion_mnist', 'cifar10']:
    for noniid in ['class_2','dirichlet_0.1','dirichlet_0.5','dirichlet_1']:
    # for noniid in ['0.1','6']:
        # central
        method = 'Global'
        file_list = glob.glob(folder+ method + '_' +'*'+dataset+'*')
        if len(file_list)>0:
            acc= []
            for i in file_list:
                with open(i, 'rb') as f:
                    log = pickle.load(f)
                    # print(len(log['server']))
                    acc.append(log['node']['0'])
                    # if len(log['server'])<=101:
                    #     acc.append(log['server'])
            acc_df = pd.DataFrame(acc)
            # print(acc_df)
            acc_df_n = pd.DataFrame()
            for i in acc_df.columns:
                acc_df_tmp = acc_df[[i]]
                acc_df_tmp.loc[:,'round'] = i+1
                acc_df_tmp = acc_df_tmp.rename(columns={i : 'test acc'})
                acc_df_n = pd.concat([acc_df_n, acc_df_tmp], axis=0)
            # print(acc_df_n)
            print(method, dataset, np.mean(acc_df_n[acc_df_n['round'] >=98]['test acc'])*100, np.std(acc_df_n[acc_df_n['round'] >=98]['test acc'])*100)
            sns.lineplot(x=acc_df_n["round"], y=acc_df_n["test acc"], label = method)
        # others
        for method in ['Local', 'Fedavg', 'Ditto',  'WCFL_3', 'WCFL_5', 'WCFL_10']:
        # for method in ['Fedavg', 'Ditto', 'Local', 'WCFL_5', 'WCFL_10']:
        # for method in ['fedavg', 'ditto', 'local', 'cfl']:
            file_list = glob.glob(folder+ method + '_' +dataset+'*'+ noniid +'*')
            if len(file_list)>0:
                acc= []
                for i in file_list:
                    with open(i, 'rb') as f:
                        log = pickle.load(f)
                        # print(len(log['server']))
                        # acc.append(log['node']['0'])
                        if len(log['server'])<=101:
                            acc.append(log['server'])
                acc_df = pd.DataFrame(acc)
                # print(acc_df)
                acc_df_n = pd.DataFrame()
                for i in acc_df.columns:
                    acc_df_tmp = acc_df[[i]]
                    acc_df_tmp.loc[:,'round'] = i+1
                    acc_df_tmp = acc_df_tmp.rename(columns={i : 'test acc'})
                    acc_df_n = pd.concat([acc_df_n, acc_df_tmp], axis=0)
                # print(acc_df_n)
                print(method, dataset, noniid, np.mean(acc_df_n[acc_df_n['round'] >=98]['test acc'])*100, np.std(acc_df_n[acc_df_n['round'] >=98]['test acc'])*100)
                sns.lineplot(x=acc_df_n["round"], y=acc_df_n["test acc"], label = method)

        sns.set_theme(style="darkgrid")
        plt.title(dataset+'_'+noniid)
        # plt.show()
        local_file = './vis/' + dataset+'_'+noniid +'.png'
        Path(local_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(local_file, dpi=200)
        plt.close()