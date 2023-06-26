import json
import os
from typing import Dict, List, Tuple
import sys
from utils.dataset import *
from models.data_segment.segmentor_core import SourceSplitterCore
from nltk import sent_tokenize
import evaluate
import pandas as pd
from evaluate import load
import random


class SourceSegmentor(object):
    def __init__(self, cfg, data: Dict[str,List] = None, labels: Dict[str, List] = None, load_from_file=True):
        self.cur_stage = cfg.cur_stage
        self.cfg = getattr(cfg, f"stage{self.cur_stage}")
        self.output_path = cfg.train.output_path
        self.splitter = SourceSplitterCore(1024)

        # load source and targets
        self.data = data
        self.labels = labels

        # store segmented source and targets
        self.split_source = {'train':[], 'val':[], 'test':[]}
        self.dupli_target = {'train':[], 'val':[], 'test':[]}
        self.source = {'train': [], 'val': [], 'test': []}
        self.summ = {'train': [], 'val': [], 'test': []}
        self.count = {'train':[], 'val':[], 'test':[]}

    def segment_1(self, query=None) -> Tuple[Dict[str, list], Dict[str, list],
                                            Dict[str, List[int]]]:
        rouge = evaluate.load('rouge')

        q = 0
        cnt = []
        pos = []
        summ = []
        neg = []
        kkk = 0
        for data_type in ['train', 'val', 'test']:
            for i, (trans, target) in enumerate(zip(self.data[data_type], self.labels[data_type])):
                print(q)
                tmp = {}
                q+=1
                split_trans = self.splitter.segment_one_sample_stride(trans)
                tt = []
                for qw in split_trans:
                    tt.append(len(' '.join(qw).split()))
                
                print("Segment 길이 : ", tt)
            
                split_trans = [' '.join(x).lower() for x in split_trans if len(' '.join(x).split()) >= 700]
                split_target = sent_tokenize(target)
                
                print("조절 후  Segment 개수 : ", len(split_trans))
                tt = []
                for qw in split_trans:
                    tt.append(len(qw.split()))

                for tar in split_target:
                    tars = [tar.lower()] * len(split_trans)
                    ## --------- Rouge ---------
                    score = rouge.compute(predictions=split_trans,
                                          references=tars, tokenizer=lambda x: word_tokenize(x), use_aggregator=False, use_stemmer=True)

                    score =[x + y + z for x, y, z in zip(score["rouge1"], score["rouge2"], score["rougeL"])]

                    positive = split_trans[score.index(max(score))]

                    if positive.strip() in tmp:
                        tmp[positive.strip()] = tmp[positive.strip()] + " " + tar
                    else:
                        tmp[positive.strip()] = tar
                
                for positive, tar in tmp.items():
                    tars = [tar.lower()] * len(split_trans)
                    score = rouge.compute(predictions=split_trans,
                                          references=tars, tokenizer=lambda x: word_tokenize(x), use_aggregator=False, use_stemmer=True)

                    score =[x + y + z for x, y, z in zip(score["rouge1"], score["rouge2"], score["rougeL"])]

                    negative = split_trans[score.index(min(score))]
                    if positive == negative:
                        print("What happend!!!!!!!!!!!!!!!!!!!!")
                        score[score.index(min(score))] = 999
                        negative = split_trans[score.index(min(score))]
                    print("Negative length : ", score.index(min(score)), len(positive.split()), len(negative.split()))  

                    if data_type != 'train':
                        self.source[data_type].append(positive.strip())
                    else:
                        self.source[data_type].append(positive.strip() + " </s></s> " +  " " + negative.strip())
                    self.summ[data_type].append(tar)

                self.count[data_type].append(len(self.source[data_type]))

        return self.source, self.summ, self.count

    def save_1(self):
        stage_path = "/tmp/pycharm_project/Summ-N/icsi/stage_1/"

        for data_type in ['train', 'val', 'test']:
            source_output_path = stage_path + f"{data_type}_rouge12L.csv"
            count_output_path = stage_path + f"{data_type}_count.json"

            data = pd.DataFrame()
            data["text"] = self.source[data_type]
            data["summary"] = self.summ[data_type]
            data.to_csv(source_output_path, index=True)

            with open(count_output_path, 'w', encoding='utf-8') as file:
                json.dump(self.count[data_type], file)

    def segment(self, query=None) -> Tuple[Dict[str, list], Dict[str, list],
                                            Dict[str, List[int]]]:
        for data_type in ['train', 'val', 'test']:
            for i, (trans, target) in enumerate(zip(self.data[data_type], self.labels[data_type])):
                split_trans = self.splitter.segment_one_sample_stride(trans)
                split_trans = [' '.join(x).lower() for x in split_trans]

                for tran in split_trans:
                    self.split_source[data_type].append(tran.strip())
                    self.dupli_target[data_type].append(target.strip())
                
                self.count[data_type].append(len(split_trans))
        return self.split_source, self.dupli_target, self.count

    def save(self):
        stage_path = "/tmp/pycharm_project/Summ-N/icsi/stage_1/"

        for data_type in ['train', 'val', 'test']:
            source_output_path = stage_path + f"{data_type}_whole.csv"
            count_output_path = stage_path + f"{data_type}_whole_count.json"

            data = pd.DataFrame()
            assert len(self.dupli_target[data_type]) == len(self.split_source[data_type])
            data["text"] = self.split_source[data_type]
            data["summary"] = self.dupli_target[data_type]
            data.to_csv(source_output_path, index=True)

            with open(count_output_path, 'w', encoding='utf-8') as file:
                json.dump(self.count[data_type], file)
