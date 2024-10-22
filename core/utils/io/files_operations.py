import os
from typing import Union, Any
from pathlib import Path
import pickle
import json

import yaml


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




def read_yaml(yaml_filepath: str) -> dict:
    """
    Description:
        Read yaml file

    :param yaml_filepath: filepath to .yaml file

    :return: dictionary with yaml data
    """
    parameters = None
    if not os.path.exists(yaml_filepath):
        raise FileNotFoundError('Parameters YAML file does not exist')

    with open(yaml_filepath) as f:
        try:
            parameters = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
    if parameters is None:
        raise ValueError('Something wrong with the YAML file')

    return parameters


def check_filename_entry_in_folder(folder, filename_entry) -> bool:
    """
    Description:
        Check if file with given filename_entry exists in folder

    :param folder: folder to check in
    :param filename_entry:  filename entry

    """
    for s in os.listdir(folder):
        if s.find(filename_entry) > -1:
            return True

    return False
