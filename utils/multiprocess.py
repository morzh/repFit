from multiprocessing import Pool
from itertools import repeat


def run_pool_steady_camera_filter(func, filepaths: list, output_folder: str, parameters: dict, n_process: int = 12):
    with Pool(n_process) as pool:
        results = pool.starmap(func, zip(filepaths, repeat(output_folder), repeat(parameters)))
    return results
