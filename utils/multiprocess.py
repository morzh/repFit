from multiprocessing import Pool
from itertools import repeat
from functools import partial
import torch


def run_pool(func, args: list, n_process: int = 10):
    n_process = min(len(args), n_process)
    if n_process < 1:
        raise ValueError(f"n_process must be more 0. {n_process=}")
    with Pool(n_process) as p:
        results = p.map(func, args)
    return results


def apply_args_and_kwargs(fn, *args, **kwargs):
    return fn(args, kwargs)


def run_pool_steady_camera_filter(func, filepaths: list, output_folder: str, number_processes: int = 12, **kwargs):
    def raise_error(error):
        raise error
    pool = Pool(number_processes)
    for filepath in filepaths:
        pool.apply_async(func, args=(filepath, output_folder), kwds=kwargs, error_callback=raise_error)
    pool.close()
    pool.join()


def run_pool_torch(func, args: list, n_process: int = 4):
    """ Run multiprocessing with  shared memory file leaks prevention"""
    torch.multiprocessing.set_start_method('spawn')
    with Pool(n_process) as p:
        results = p.map(func, args)
    return results
