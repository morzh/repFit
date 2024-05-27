from multiprocessing import Pool


def run_pool(func, args: list, n_process: int = 12):
    with Pool(n_process) as p:
        results = p.map(func, args)
    return results
