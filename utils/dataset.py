import os.path
import re


def write_list_asline(path, data):
    with open(path,'w',encoding='utf-8') as file:
        for sample in data:

            file.write(sample.strip() + '\n')

def read_list_asline(path):
    data = []
    with open(path,'r',encoding='utf-8')  as file:
        for line in file:
            data.append(line.strip())
    return data


def assertion_statis(source_data, target_data, prompt):
    assert len(source_data['train']) == len(target_data['train'])
    assert len(source_data['val']) == len(target_data['val'])
    assert len(source_data['test']) == len(target_data['test'])

