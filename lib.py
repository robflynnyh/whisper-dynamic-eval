import json, os, argparse

def fetch_paths():
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'paths.json'), 'r') as f:
        paths = json.load(f)
    return paths

def get_path(key:str):
    return fetch_paths()[key]

def get_args(parser:argparse.ArgumentParser):
    parser.add_argument('-m', '--model', type=str, default='base.en', help='Whisper model name')
    parser.add_argument('-split', '--split', type=str, default='test', help='test or dev split')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser