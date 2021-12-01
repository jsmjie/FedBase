# from fedbase import main_global
# import os

# # os.chdir(os.path.dirname(os.path.abspath(__file__)))
# main_global.run_global('./data/','mnist')

from fedbase.baselines.central import central
from fedbase.baselines.fedavg import fedavg
from fedbase.baselines.ditto import ditto
from fedbase.model.model import CNNCifar, CNNMnist
from fedbase.nodes.node import node
import torch.optim as optim
import torch.nn as nn

# fedavg('./data/','mnist', 64, 30, CNNMnist(), nn.CrossEntropyLoss(), 'SGD', 100, 3)
ditto('./data/','mnist', 64, 30, CNNMnist(), nn.CrossEntropyLoss(), 'SGD', 100, 3, 0.95)