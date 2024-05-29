from typing import Union
from pathlib import Path
import pickle


def write_pickle(obj, fpath: Union[str, Path]):
    with open(fpath, 'wb') as file:
        pickle.dump(obj, file)

