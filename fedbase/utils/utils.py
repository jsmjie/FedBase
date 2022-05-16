import glob
from torch.utils.data import Dataset
import torch.utils.data as data
import torch

def unpack_args(func):
    from functools import wraps
    @wraps(func)
    def wrapper(args):
        if isinstance(args, dict):
            return func(**args)
        else:
            return func(*args)
    return wrapper

def find_files(dir,*args):
    files=glob.glob(dir)
    print(files)

def get_targets(dataset):
    """Get the targets of a dataset without any target target transforms(!)."""
    # if isinstance(dataset, TransformedDataset):
    #     return get_targets(dataset.dataset)
    if isinstance(dataset, data.Subset):
        targets = get_targets(dataset.dataset)
        return torch.as_tensor(targets)[dataset.indices]
    if isinstance(dataset, data.ConcatDataset):
        return torch.cat([get_targets(sub_dataset) for sub_dataset in dataset.datasets])
    try:
        if torch.is_tensor(dataset.targets)==False:
            return torch.as_tensor(dataset.targets)
        else:
            return dataset.targets
    except:
        # print(dataset.labels)
        if torch.is_tensor(dataset.labels)==False:
            return torch.as_tensor(dataset.labels)
        else:
            return dataset.labels
    # if isinstance(
    #         dataset, (datasets.MNIST, datasets.ImageFolder,)
    # ):
    #     return torch.as_tensor(dataset.targets)
    # if isinstance(dataset, datasets.SVHN):
    #     return dataset.labels

    # raise NotImplementedError(f"Unknown dataset {dataset}!") 
# find_files('./log/central*cifar10*')