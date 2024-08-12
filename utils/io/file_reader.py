from typing import Union, Any
from pathlib import Path
import pickle
import json

def write_pickle(obj, fpath: Union[str, Path]):
    with open(fpath, 'wb') as file:
        pickle.dump(obj, file)


def read_pickle(fpath: Union[str, Path]) -> Any:
    with open(fpath, 'rb') as file:
        result = pickle.load(file)
    return result


def write_json(obj, fpath: Union[str, Path]):
    with open(fpath, 'w', encoding='utf8') as file:
        json.dump(obj, file, ensure_ascii=True)
