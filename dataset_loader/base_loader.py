import os
import sys
from typing import List, Dict, Tuple
from dataset import *
from utils.dataset import write_list_asline


class LoaderBase(object):
    def __init__(self,cfg):
        self.data = {"train":[], "test":[], "val":[]}
        self.label = {"train":[], "test":[], "val":[]}
        self.cfg = cfg
        self.is_dialogue = True

    # load dataset and return a list of source and target
    def load(self) -> Tuple[Dict[str, list], Dict[str, list]]:
        raise NotImplementedError

