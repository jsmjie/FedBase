import glob
import pickle
from fedbase.utils.visualize import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# file_list = glob.glob('./log/local*cifar10*dirichlet_1*')
file_list = glob.glob('./log/central*cifar10*')
# file_list = glob.glob('./log/cfl*cifar10*class*')
acc= []
for i in file_list:
    with open(i, 'rb') as f:
        log = pickle.load(f)
        # print(log)
        acc.append(log['node']['0'])
acc_df = pd.DataFrame(acc)
print(acc_df)
acc_df_n = pd.DataFrame()
for i in acc_df.columns:
    acc_df_tmp = acc_df[[i]]
    acc_df_tmp['round'] = i
    acc_df_tmp.rename(columns={i : 'acc'}, inplace=True)
    acc_df_n = pd.concat([acc_df_n, acc_df_tmp], axis=0)
print(acc_df_n)

sns.set_theme(style="darkgrid")
sns.lineplot(x=acc_df_n["round"], y=acc_df_n["acc"])
plt.show()