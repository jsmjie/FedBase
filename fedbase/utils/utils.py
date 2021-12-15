import glob

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

# find_files('./log/central*cifar10*')