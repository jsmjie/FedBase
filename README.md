# FedBase
An easy, modularized, DIY Federated Learning framework with many baselines for individual researchers.

# Three steps to achieve FedAvg!
1. Data partition
2. Nodes and server simulation
3. Train and test

# Design philosophy
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
            2. N classes
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

# How to develop your own FL with fedbase?

# Baselines
1. Centralized train
2. Local train
3. FedAvg
4. Ditto
5. Clustered FL
6. ...

# To be continued...