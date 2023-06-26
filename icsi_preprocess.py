import os
import statistics
from argparse import ArgumentParser
from configue import Configure
from tools import get_dataloader
from datasets import load_dataset
from training_args_icsi import TrainArgs
from data_segment.source_segment_icsi import SourceSegmentor
from data_segment.target_matching import TargetSegmentor



if __name__ == '__main__':
    # Parse all arguments
    parser = ArgumentParser(TrainArgs)
    training_args = parser.parse_args()
    training_args = TrainArgs()
    args = Configure.Get(training_args.cfg)

    args.train = training_args
    args.dataset.loader_name = "ICSI"
    dataset_loader = get_dataloader(args.dataset.loader_name)(args)

    source_data, target_data, query_data = dataset_loader.load()

    source_segmentor = SourceSegmentor(args, source_data, target_data)

    # Making Contrastive data 
    source, summary, count = source_segmentor.segment_1(
    query=query_data)
    source_segmentor.save_1()

    # Making Whole data like SUMM^N
    source, summary, count = source_segmentor.segment(query=query_data)
    source_segmentor.save()

    target_segmentor = TargetSegmentor(args, source, summary)
    target, _ = target_segmentor.segment()
    target_segmentor.save_icsi()
