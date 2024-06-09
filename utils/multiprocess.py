from multiprocessing import Pool


def run_pool(func, args: list, n_process: int = 10):
    n_process = min(len(args), n_process)
    with Pool(n_process) as p:
        results = p.map(func, args)
    return results
