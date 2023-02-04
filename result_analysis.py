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
import json
warnings.filterwarnings("ignore")

from platform import system
def plt_maximize():
    # See discussion: https://stackoverflow.com/questions/12439588/how-to-maximize-a-plt-show-window-using-python
    backend = plt.get_backend()
    cfm = plt.get_current_fig_manager()
    if backend == "wxAgg":
        cfm.frame.Maximize(True)
    elif backend == "TkAgg":
        if system() == "Windows":
            cfm.window.state("zoomed")  # This is windows only
        else:
            cfm.resize(*cfm.window.maxsize())
    elif backend == "QT4Agg":
        cfm.window.showMaximized()
    elif callable(getattr(cfm, "full_screen_toggle", None)):
        if not getattr(cfm, "flag_is_max", None):
            cfm.full_screen_toggle()
            cfm.flag_is_max = True
    else:
        raise RuntimeError("plt_maximize() is not implemented for current backend:", backend)

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
files = [folder + f for f in os.listdir(folder)]
for f in files:
    try:
        os.rename(f, f.replace('_None',''))
    except:
        pass
# files = [f.replace('__','_') for f in files]
# for dataset in ['fashion_mnist', 'cifar10']:
for dataset in ['fashion_mnist', 'cifar10', 'medmnist_pathmnist', 'medmnist_tissuemnist']:
# for dataset in ['cifar10']:
# for dataset in [ 'medmnist_octmnist']:
    # client-wise
    for noniid in ['200_0.1_dirichlet','200_2_class']:
    # cluster-wise
    # for noniid in ['10_dirichlet','3_class']:
        print(dataset, noniid)
    # for noniid in ['0.1','6']:
        # central
        # method = 'central'
        # file_list = glob.glob(folder+ method + '_' +'*'+dataset+'*')
        # if len(file_list)>0:
        #     acc= []
        #     for i in file_list:
        #         with open(i, 'rb') as f:
        #             log = json.load(f)
        #             # print(len(log['server']))
        #             acc.append(log['node']['0'])
        #             # if len(log['server'])<=101:
        #             #     acc.append(log['server']
        #     acc_df = pd.DataFrame(acc)
        #     # print(acc_df)
        #     acc_df_n = pd.DataFrame()
        #     for i in acc_df.columns:
        #         acc_df_tmp = acc_df[[i]]
        #         acc_df_tmp.loc[:,'round'] = i+1
        #         acc_df_tmp['test acc'] = acc_df_tmp[i].apply(lambda x: x[0])
        #         acc_df_tmp['test macro f1'] = acc_df_tmp[i].apply(lambda x: x[1])
        #         acc_df_n = pd.concat([acc_df_n, acc_df_tmp], axis=0)
        #     # print(acc_df_n)
        #     print(method, dataset, round(np.mean(acc_df_n[acc_df_n['round'] >=68]['test acc'])*100,2), round(np.std(acc_df_n[acc_df_n['round'] >=68]['test acc'])*100,2)\
        #             , round(np.mean(acc_df_n[acc_df_n['round'] >=68]['test macro f1'])*100,2), round(np.std(acc_df_n[acc_df_n['round'] >=68]['test macro f1'])*100,2))
        #     sns.lineplot(x=acc_df_n["round"], y=acc_df_n["test acc"], label = method)
        # others
        # for method in ['Local', 'Fedavg', 'Ditto',  'WCFL_3', 'WCFL_5', 'WCFL_10']:
        # for method in ['local', 'fedavg', 'fedavg_finetune', 'ditto','ditto_0.95', 'fedprox','fedprox_0.1', 'wecfl_3_', 'wecfl_5_', 'wecfl_10_', 'ifca_3_', 'ifca_5_', 'ifca_10_', 'fesem_3_', 'fesem_5_', 'fesem_10_',\
        #     'wecfl_3_0.95', 'wecfl_5_0.95', 'wecfl_10_0.95', 'ifca_3_0.95', 'ifca_5_0.95', 'ifca_10_0.95', 'fesem_3_0.95', 'fesem_5_0.95', 'fesem_10_0.95']:
        # for method in ['fedavg', 'fedprox_0.1', 'fedavg_ensemble_5', 'fedavg_ensemble_10', 'fedprox_ensemble_5_0.1', 'fedprox_ensemble_10_0.1', 'ifca_5', 'ifca_10', 'fesem_5', 'fesem_10',  'wecfl_5', 'wecfl_10']:
        # for method in [i+j for i in ['ifca_5', 'ifca_10', 'fesem_5', 'fesem_10',  'wecfl_5', 'wecfl_10'] for j in ['','_1','_0.1','_0.01','_0.001']]:
        for method in ['ifca_res_5','wecfl_res_5', 'ifca_res_10','wecfl_res_10']:
        # for method in ['fedavg', 'fedprox', 'ditto', 'ifca_10', 'fesem_10', 'wecfl_10', 'fedavg_ensemble_10','fedprox_ensemble_10']:
        # for method in ['fedavg_ensemble_5','fedprox_ensemble_5','fedavg_ensemble_10','fedprox_ensemble_10']:
        # for method in ['fedprox_ensemble_5_0.95','fedprox_ensemble_10_0.95']:
        # for method in ['fedavg','fedprox']:
        # for method in ['wecfl_3', 'wecfl_5', 'wecfl_10', 'wecfl_3_0.95', 'wecfl_5_0.95', 'wecfl_10_0.95', 'fesem_3_0.95', 'fesem_5_0.95', 'fesem_10_0.95']:
        # for method in ['wecfl_5_0.95']:
        # for method in ['fedavg', 'ditto', 'local', 'cfl']:
            file_list = glob.glob(folder + method + '_' +dataset+'*'+ noniid +'*')
            if len(file_list)>0:
                acc= []
                for i in file_list:
                    with open(i, 'rb') as f:
                        try:
                            log = json.load(f)
                            # print(len(log['server']))
                            # acc.append(log['node']['0'])
                            if len(log['server'])<=101:
                                acc.append(log['server'])
                        except:
                            pass
                acc_df = pd.DataFrame(acc)
                # print(acc_df)
                acc_df_n = pd.DataFrame()
                for i in acc_df.columns:
                    acc_df_tmp = acc_df[[i]]
                    acc_df_tmp.loc[:,'round'] = i+1
                    acc_df_tmp['test acc'] = acc_df_tmp[i].apply(lambda x: x[0])
                    acc_df_tmp['test macro f1'] = acc_df_tmp[i].apply(lambda x: x[1])
                    # print(acc_df_tmp)
                    # acc_df_tmp = acc_df_tmp.rename(columns={i : 'test acc'})
                    acc_df_n = pd.concat([acc_df_n, acc_df_tmp], axis=0)
                # print(acc_df_n)
                print(method, dataset, noniid, "{}$\pm${}".format(round(np.mean(acc_df_n[acc_df_n['round'] >=68]['test acc'])*100,2), round(np.std(acc_df_n[acc_df_n['round'] >=68]['test acc'])*100,2))\
                    ,' & ' , "{}$\pm${}".format(round(np.mean(acc_df_n[acc_df_n['round'] >=68]['test macro f1'])*100,2), round(np.std(acc_df_n[acc_df_n['round'] >=68]['test macro f1'])*100,2)))
                # acc_df_n.to_csv('./vis/nips2022/' + method + '_'+ noniid + dataset +'.csv')
                # sns.lineplot(x=acc_df_n["round"], y=acc_df_n["test acc"], label = method)
                # sns.lineplot(x=acc_df_n["round"], y=acc_df_n["test macro f1"], label = method)

        sns.set_theme(style="darkgrid")
        plt.title(dataset+'_'+noniid)
        # plt.show()
        local_file = './vis/' + dataset+'_'+noniid +'.png'
        Path(local_file).parent.mkdir(parents=True, exist_ok=True)
        plt_maximize()
        plt.savefig(local_file)
        plt.close()
        

# file_list = glob.glob('./vis/nips2022/'+'*.csv')
# print(file_list)
# label_list = ['Ditto', 'Fedavg+(10)', 'Fedprox+(10)', 'FeSEM(10)', 'IFCA(10)', 'WeCFL(10)', 'WeCFL(10)', 'WeCFL(3)', 'WeCFL(5)']
# sns.set_theme(style="darkgrid")
# # list_index = [7,8,6]
# list_index = [4,3,5]
# for i in list_index:
# # for f in file_list:
#     acc_df_n = pd.read_csv(file_list[i])
#     sns.lineplot(x=acc_df_n["round"], y=acc_df_n["test acc"], label = label_list[i])
#     # sns.lineplot(x=acc_df_n["round"], y=acc_df_n["test macro f1"], label = label_list[i])    
# # plt.title('Test Accuracy on CIFAR10 under <3,2>-class setting', fontsize=16)
# plt.xlabel('Round', fontsize=16)
# # plt.ylabel('Test Macro F1', fontsize=16)
# plt.ylabel('Test Accuracy', fontsize=16)
# # plt.ylim((0.8,1))
# plt.legend(loc =2)

# # plt.show()
# local_file = './vis/nips2022/' + 'acc' +'.png'
# # local_file = './vis/nips2022/' + 'f1' +'.png'
# Path(local_file).parent.mkdir(parents=True, exist_ok=True)

# # plt_maximize()
# plt.savefig(local_file)
# plt.close()

# for i in list_index:
# # for f in file_list:
#     acc_df_n = pd.read_csv(file_list[i])
#     # sns.lineplot(x=acc_df_n["round"], y=acc_df_n["test acc"], label = label_list[i])
#     sns.lineplot(x=acc_df_n["round"], y=acc_df_n["test macro f1"], label = label_list[i])    
# # plt.title('Test Macro F1 on CIFAR10 under <3,2>-class setting', fontsize=16)
# plt.xlabel('Round', fontsize=16)
# plt.ylabel('Test Macro F1', fontsize=16)
# # plt.ylim((0.5,1))
# # plt.ylabel('Test Accuracy', fontsize=16)

# # plt.show()
# # local_file = './vis/nips2022/' + 'acc' +'.png'
# local_file = './vis/nips2022/' + 'f1' +'.png'
# Path(local_file).parent.mkdir(parents=True, exist_ok=True)
# # plt_maximize()
# plt.savefig(local_file)
# plt.close()