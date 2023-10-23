import json, os

def fetch_paths():
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'paths.json'), 'r') as f:
        paths = json.load(f)
    return paths

def get_path(key:str):
    return fetch_paths()[key]