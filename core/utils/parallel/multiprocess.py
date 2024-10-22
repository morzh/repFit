import torch
from multiprocessing import Pool
from jedi.inference.gradual.typing import Callable


def run_pool(func, args: list, number_processes: int = 10):
    """
    Description:
        Run multiprocessing with shared memory file leaks prevention.

    :param func:
    :param args:
    :param number_processes: number of processes in parallel pool.
    """
    number_processes = min(len(args), number_processes)
    if number_processes < 1:
        raise ValueError(f"n_process must be more 0. {number_processes=}")
    with Pool(number_processes) as p:
        results = p.map(func, args)
    return results


def apply_args_and_kwargs(fn, *args, **kwargs) -> Callable:
    """
    Description:

    """
    return fn(args, kwargs)


def run_pool_steady_camera_filter(func, filepaths: list, output_folder: str, number_processes: int = 12, **kwargs) -> None:
    """
    Description:
        Run multiprocessing with shared memory file leaks prevention.

    :param func:
    :param filepaths:
    :param output_folder:
    :param number_processes: number of processes in parallel pool.
    """
    def raise_error(error):
        raise error
    pool = Pool(number_processes)
    for filepath in filepaths:
        pool.apply_async(func, args=(filepath, output_folder), kwds=kwargs, error_callback=raise_error)
    pool.close()
    pool.join()



def run_pool_single_persons_filter(func, filepaths: list, output_folder: str, number_processes: int = 4, **kwargs) -> None:
    """
    Description:
        Run multiprocessing with shared memory file leaks prevention.

    :param func:
    :param filepaths:
    :param output_folder:
    :param number_processes: number of processes in parallel pool.
    """
    def raise_error(error):
        raise error
    pool = Pool(number_processes)
    for filepath in filepaths:
        pool.apply_async(func, args=(filepath, output_folder), kwds=kwargs, error_callback=raise_error)
    pool.close()
    pool.join()


def run_pool_torch(func, args: list, number_processes: int = 4) -> None:
    """
    Description:
        Run multiprocessing with shared memory file leaks prevention.

    :param func:
    :param args:
    :param number_processes: number of processes in parallel pool.
    """
    torch.multiprocessing.set_start_method('spawn')
    with Pool(number_processes) as p:
        results = p.map(func, args)
    return results
