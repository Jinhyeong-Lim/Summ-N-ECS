import os.path
import json
from tqdm import tqdm
import multiprocessing
from typing import Dict, List, Tuple
import sys
from segmentor_core import TargetSplitterCore
from utils.dataset import *
from tools import add_sys_path
import pandas as pd
add_sys_path()


class TargetSegmentor(object):
    def __init__(self, cfg, data: Dict[str, List] = None, labels: Dict[str, List] = None, load_from_file=True):
        self.cur_stage = cfg.cur_stage
        self.cfg = getattr(cfg, f"stage{self.cur_stage}")
        self.output_path = cfg.train.output_path
        self.splitter = TargetSplitterCore()

        self.data = data
        self.labels = labels

        # store segmented targets, and the unordered segment
        self.target = {'train': [], 'val': [], 'test': []}
        self.best_label_with_scores = {'train': [], 'val': [], 'test': []}

    def segment(self) -> Tuple[Dict[str, list],Dict[str, list]]:
        for data_type in ['train', 'val', 'test']:
            # we use multiprocessing to accelerate the split process
            tasks = list(zip(self.data[data_type], self.labels[data_type], range(len(self.data[data_type]))))
            cores = min(multiprocessing.cpu_count(), self.cfg.cores_used)
            pool = multiprocessing.Pool(processes=cores)
            for i, (new_sents, new_tar) in tqdm(enumerate(pool.starmap(self.splitter.fast_rouge, tasks))):
                self.target[data_type].append(new_tar.strip())
                self.best_label_with_scores[data_type].append(new_sents)

        return self.target, self.best_label_with_scores

    def save_icsi(self):
        stage_path = "./Summ-N-ECS/icsi/stage_1/"

        for data_type in ['train', 'val', 'test']:
            source_output_path = stage_path + f"{data_type}_whole.csv"
            count_output_path = stage_path + f"{data_type}_count.json"
            data = pd.read_csv(source_output_path)
            data["summary"] = self.target[data_type]
            data.to_csv(source_output_path, index=False)

