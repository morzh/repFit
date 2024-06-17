from multiprocessing import Pool
import torch


def run_pool_torch(func, args: list, n_process: int = 4):
    """ Run multiprocessing with  shared memory file leaks prevention"""
    torch.multiprocessing.set_start_method('spawn')
    with Pool(n_process) as p:
        results = p.map(func, args)
    return results
