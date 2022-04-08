# FedBase
An easy, modularized, DIY Federated Learning framework with many baselines for individual researchers.

## Installation
[fedbase @ pypi](https://pypi.org/project/fedbase/)
```python
pip install --upgrade fedbase
```

## Baselines
1. Centralized training
2. Local training
3. FedAvg, [Communication-Efficient Learning of Deep Networksfrom Decentralized Data](https://arxiv.org/abs/1602.05629)
4. FedAvg + Finetune
5. Fedprox, [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127)
5. Ditto, [Ditto: Fair and Robust Federated Learning Through Personalization](https://arxiv.org/abs/2012.04221)
6. WeCFL, [On the Convergence of Clustered Federated Learning](https://arxiv.org/abs/2202.06187)
7. IFCA, [An Efficient Framework for Clustered Federated Learning](https://arxiv.org/abs/2006.04088)
8. To be continued...

## Three steps to achieve FedAvg!
1. Data partition
2. Nodes and server simulation
3. Train and test

## Design philosophy
1. Dataset
    1. Dataset
        1. MNIST
        2. CIFAR-10
        3. Fashion-MNIST
        4. ...
    2. Dataset partition
        1. IID
        2. Non-IID
            1. Dirichlet distribution
            2. N-class
            3. ...
        3. Fake data
        4. ...
    <!-- 3. Batch_size -->
2. Node
    1. Local dataset
    2. Model
    3. Objective
    4. Optimizer
    5. Local update
    6. Test
3. Server
    1. Model
    2. Aggregate
    3. Distribute
4. Server & Node
    1. Topology
    2. Client sampling
    3. Exchange message
5. Baselines
    1. Global
    2. Local
    3. FedAvg
6. Visualization

## How to develop your own FL with fedbase?