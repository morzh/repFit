from multiprocessing import Pool
from itertools import repeat
import torch


def run_pool(func, args: list, n_process: int = 10):
    n_process = min(len(args), n_process)
    if n_process < 1:
        raise ValueError(f"n_process must be more 0. {n_process=}")
    with Pool(n_process) as p:
        results = p.map(func, args)
    return results


def run_pool_steady_camera_filter(func, filepaths: list, output_folder: str, parameters: dict, n_process: int = 12):
    with Pool(n_process) as pool:
        results = pool.starmap(func, zip(filepaths, repeat(output_folder), repeat(parameters)))
    return results


def run_pool_torch(func, args: list, n_process: int = 4):
    """ Run multiprocessing with  shared memory file leaks prevention"""
    torch.multiprocessing.set_start_method('spawn')
    with Pool(n_process) as p:
        results = p.map(func, args)
    return results
