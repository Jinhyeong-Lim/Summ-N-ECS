import gzip
import os
import json

from utils.dataset import *
from dataset_loader.base_loader import LoaderBase
from nltk import word_tokenize


def tokenize(sent):
    tokens = ' '.join(word_tokenize(sent))
    return tokens



class Loader(LoaderBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.is_dialogue = True

    def load(self):
        a = 0
        le = 0
        for data_type in ['train', 'val', 'test']:
            if data_type == "val":  # name of ICSI is dev not val
                data_path = os.path.join("./Summ-N-ECS/data/ICSI", "dev")
            else:
                data_path = os.path.join("./Summ-N-ECS/data/ICSI", data_type)

            samples = []
            for gz_name in os.listdir(data_path):
                if 'gz' not in gz_name:
                    continue
                sample_path = os.path.join(data_path, gz_name)
                with gzip.open(sample_path, 'rb') as file:
                    for line in file:
                        samples.append(json.loads(line))

            for sample in samples:
                meeting = []
                for turn in sample['meeting']:

                    sent = turn['role'] + ' ' + turn['speaker'] + " : "
                    sent += tokenize(' '.join(turn['utt']['word']))
                    meeting.append(sent)
                summary = ' '.join(sample['summary'])

                self.data[data_type].append(meeting)
                self.label[data_type].append(summary)
    
        return self.data, self.label, None