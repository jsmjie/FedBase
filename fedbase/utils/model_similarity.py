import torch


# KL divergence variant log p/q,abs(log p-log q)
def blackbox_mc(model, testset):
    outputs = model(testset)
    return outputs.sum(axis=1)/len(testset)

# model similarity
def similarity(model_1,model_2):
    return abs(log(blackbox_mc(model_1))-log(blackbox_mc(model_2)))
