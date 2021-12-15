import torch

def save_checkpoint(model_structure, model_parameter, optimizer_parameter, path):
    checkpoint = {'model_structure': model_structure,
                  'model_parameter': model_parameter, 'optimizer_parameter': optimizer_parameter}
    torch.save(checkpoint, path)

def load_checkpoint(path):
    checkpoint = torch.load(path)
    model = checkpoint['model_structure']
    model.load_state_dict(checkpoint['model_parameter'])
    # for parameter in model.parameters():
    #     parameter.requires_grad = False
    # optimizer = checkpoint['optimizer']
    # model.eval()
    return model, checkpoint['optimizer_parameter']


# KL divergence variant log p/q,abs(log p-log q)
def blackbox_mc(model, testset):
    outputs = model(testset)
    return outputs.sum(axis=1)/len(testset)

# model similarity
def similarity(model_1,model_2):
    return abs(log(blackbox_mc(model_1))-log(blackbox_mc(model_2)))
