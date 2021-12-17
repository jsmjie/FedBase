import glob
import pickle
from fedbase.utils.visualize import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


# file_list = glob.glob('./log/local*cifar10*dirichlet_1*')
# file_list = glob.glob('./log/central*fashion*')
# dataset = 'cifar10'
# dataset = 'fashion_mnist'
# noniid = 'dirichlet_1'
# noniid = 'class_2'
for dataset in ['fashion_mnist', 'cifar10']:
    for noniid in ['class_2','dirichlet_0.1','dirichlet_0.5','dirichlet_1']:
        # central
        method = 'central'
        file_list = glob.glob('./log/'+ 'central' + '_' +'*'+dataset+'*')
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
                acc_df_tmp['round'] = i+1
                acc_df_tmp.rename(columns={i : 'acc'}, inplace=True)
                acc_df_n = pd.concat([acc_df_n, acc_df_tmp], axis=0)
            print(acc_df_n)
            sns.lineplot(x=acc_df_n["round"], y=acc_df_n["acc"], label = method)
        # others
        for method in ['fedavg', 'ditto', 'local', 'fedavg_finetune',  'cfl_3', 'cfl_5', 'cfl_10']:
            file_list = glob.glob('./log/'+ method + '_' +'*'+dataset+'*'+ noniid +'*')
            if len(file_list)>0:
                acc= []
                for i in file_list:
                    with open(i, 'rb') as f:
                        log = pickle.load(f)
                        print(len(log['server']))
                        # acc.append(log['node']['0'])
                        if len(log['server'])<=101:
                            acc.append(log['server'])
                acc_df = pd.DataFrame(acc)
                # print(acc_df)
                acc_df_n = pd.DataFrame()
                for i in acc_df.columns:
                    acc_df_tmp = acc_df[[i]]
                    acc_df_tmp['round'] = i+1
                    acc_df_tmp.rename(columns={i : 'acc'}, inplace=True)
                    acc_df_n = pd.concat([acc_df_n, acc_df_tmp], axis=0)
                print(acc_df_n)
                sns.lineplot(x=acc_df_n["round"], y=acc_df_n["acc"], label = method)

        sns.set_theme(style="darkgrid")
        plt.title(dataset+'_'+noniid)
        # plt.show()
        local_file = './vis/' + dataset+'_'+noniid +'.png'
        Path(local_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(local_file, dpi=200)
        plt.close()