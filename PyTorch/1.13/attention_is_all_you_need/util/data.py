import json
import os
import pandas as pd
import typing

class BenchmarkData:
    def __init__(self):
        self._benchmark_data = {}
        self._machine_info = {}
        self._commit_info = {}
        self._names_all = set()
        self._names_common = set()
        self._tags = []
        self._json_raw = []

    def add_json_data(self, tag, json_data):
        names = set([b['name'] for b in json_data['benchmarks']])
        self._names_all.update(names)
        if len(self._benchmark_data) == 0:
            self._names_common.update(names)
        else:
            self._names_common.intersection_update(names)
        self._benchmark_data[tag] = {b['name']: b for b in json_data['benchmarks']}
        self._machine_info[tag] = json_data['machine_info']
        self._commit_info[tag] = json_data['commit_info']
        self._tags.append(tag)
        self._json_raw.append(json_data)

    def tags(self):
        return list(self._benchmark_data.keys())

    def benchmark_names(self, mode='common', keyword_filter=None):
        """
        Return the names of benchmarks across the dataset.
        
        mode:
            'common': intersection across dataset files - useful for comparison plot
            'all': union across dataset files
            'outliers': union - intersection across dataset files
        """
        if mode == 'common':
            names = self._names_common
        elif mode == 'all':
            names = self._names_all
        elif mode == 'outliers':
            names = self._names_all - self._names_common

        if keyword_filter is not None:
            if isinstance(keyword_filter, str):
                keyword_filter = [keyword_filter]
            for kw in keyword_filter:
                names = [n for n in names if kw in n]
        
        return names 

    def as_dataframe(self, name, max_data=100):
        df = pd.DataFrame()
        for i, tag in enumerate(self._benchmark_data):
            benchmark = self._benchmark_data[tag][name]
            df = df.append(pd.DataFrame()
                .assign(time=benchmark['stats']['data'][:max_data])
                .assign(tag=tag)
                .assign(file_idx=i)
                .assign(git_repo=self._commit_info[tag]['project'])
                .assign(git_commit=self._commit_info[tag]['id'])
                .assign(torch=self._machine_info[tag]['pytorch_version'])
                .assign(torchtext=self._machine_info[tag]['torchtext_version'])
                .assign(torchvision=self._machine_info[tag]['torchvision_version'])
                .assign(date=self._commit_info[tag]['time']), ignore_index=True)
        return df


def load_data_dir(data_dir, most_recent_files:int =None, use_history_file=True):
    """
    load all the files in the given data dir, up to N most recent.  
    if use_history_file=True, find most recent files using order in history file. 
    """
    history_file = os.path.join(data_dir, 'history')
    if os.path.isfile(history_file):
        with open(history_file) as hf:
            history = hf.read().splitlines()
            files = [os.path.join(data_dir, f) for f in history]
    else:
        files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.splitext(f)[1] == '.json'])

    if most_recent_files is not None:
        files = files[:most_recent_files]
    return load_data_files(files)


def load_data_files(files: typing.List[str]):
    data = BenchmarkData()
    for fname in files:
        try:
            with open(fname) as f:
                data.add_json_data(fname, json.load(f))
        except:
            print(f"Error loading {fname}")
            raise
    return data
