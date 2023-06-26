import os
from re import T
import sys
from argparse import ArgumentParser
from dataclasses import dataclass, field
import pandas as pd
import evaluate
import os
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
import nltk
import statistics
from configue import Configure
from transformers import (
    BartForConditionalGeneration, PreTrainedTokenizerFast,
    MT5ForConditionalGeneration, MT5Tokenizer, MT5TokenizerFast, LEDForConditionalGeneration,
    DataCollatorForSeq2Seq, Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, HfArgumentParser, set_seed,
    EarlyStoppingCallback, AutoTokenizer
)
from typing import Optional
import json
from tools import get_dataloader
from datasets import load_dataset
from utils.dataset import *
from training_args_ami import TrainArgs



@dataclass
class RunArguments:
    train_file: str = field(default=None)
    validation_file: str = field(default=None)
    test_file: str = field(default=None)
    model_name: str = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    max_train_samples: str = field(default=None)
    max_eval_samples: str = field(default=None)
    max_predict_samples: str = field(default=None)
    max_source_length: Optional[int] = field(default=1024)
    max_target_length: Optional[int] = field(default=300)
    num_beams: Optional[int] = field(default=None) 
    ignore_pad_token_for_loss: bool = field(default=True)
    overwrite_cache: bool = field(default=False)
    preprocessing_num_workers: Optional[int] = field(default=None)
    val_max_target_length: Optional[int] = field(default=None)
    pad_to_max_length: bool = field(default=True)
    cfg: str = field(default=None)
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )



def main():
    parser = HfArgumentParser((Seq2SeqTrainingArguments, RunArguments))

    training_args, run_args = parser.parse_args_into_dataclasses()
    cur_stage = 1 

    training_args.per_device_train_batch_size = 24
    training_args.per_device_eval_batch_size = 24
    training_args.per_device_test_batch_size = 24

    data_files = {}
    if run_args.train_file is not None:
        data_files["train"] = run_args.train_file
        extension = run_args.train_file.split(".")[-1]
    if run_args.validation_file is not None:
        data_files["validation"] = run_args.validation_file
        extension = run_args.validation_file.split(".")[-1]
    if run_args.test_file is not None:
        data_files["test"] = run_args.test_file
        extension = run_args.test_file.split(".")[-1]

    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=run_args.cache_dir)
    

    if 'dia' in run_args.model_name:
        tokenizer = AutoTokenizer.from_pretrained('MingZhong/DialogLED-large-5120')
        models = LEDForConditionalGeneration.from_pretrained('MingZhong/DialogLED-large-5120')
    else:
        tokenizer = AutoTokenizer.from_pretrained(run_args.model_name)
        models = LEDForConditionalGeneration.from_pretrained(run_args.model_name)
        
    args = Configure.Get(run_args.cfg)

    args.dataset.loader_name = "ICSI"
    dataset_loader = get_dataloader(args.dataset.loader_name)(args)

    source_data, target_data, query_data = dataset_loader.load()


    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names

    padding = "max_length" if run_args.pad_to_max_length else False
    last_checkpoint = None
    max_target_length = run_args.max_target_length

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]
    predict_dataset = raw_datasets["test"]


    def preprocess_function(examples):
        inputs = examples['text']
        inputs = [x.strip().lower() for x in inputs]

        targets = examples['summary']
        targets = [x.lower() for x in targets]

        model_inputs = tokenizer(inputs, max_length=run_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and run_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    if training_args.do_predict:
        max_target_length = run_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")

        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=run_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not run_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=run_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not run_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

        with training_args.main_process_first(desc="train dataset map pre-processing"):
           predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=run_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not run_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    # Data collator
    label_pad_token_id = -100 if run_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, 
                                           label_pad_token_id=label_pad_token_id,
                                           pad_to_multiple_of=8 if training_args.fp16 else None)

    metric = evaluate.load("rouge")


    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels


    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if run_args.ignore_pad_token_for_loss:
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds,
                                references=decoded_labels, 
                                tokenizer=lambda x: word_tokenize(x), 
                                use_aggregator=False, 
                                use_stemmer=True
                                )

        result = {key: statistics.mean(value) * 100 for key, value in
                  result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}

        return result
      

    def combine(split_results, count_list):
        hypos = []
        start = 0
        for lengths in count_list:
            if start + lengths <= len(split_results):
                end = start + lengths
                sor = " </s> ".join(split_results[start:end]).replace("\n", "").strip()
                hypos.append(sor)
                start = start + lengths
            else:
                break
        return hypos


    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
    )


    if training_args.do_predict:
        for data_type in ["train", "val", "test"]:
            if data_type == "train":
                dataset = train_dataset
                predict_results = trainer.predict(
                train_dataset,
                metric_key_prefix=" ",
                max_length=run_args.val_max_target_length,
                min_length=20,
                num_beams=run_args.num_beams,
                length_penalty=2.0,
                no_repeat_ngram_size=3
                )
                summary = target_data[data_type]

            elif data_type == "val":
                dataset = eval_dataset
                predict_results = trainer.predict(
                eval_dataset,
                metric_key_prefix=" ",
                max_length=run_args.val_max_target_length,
                min_length=20,
                num_beams=run_args.num_beams,
                length_penalty=2.0,
                no_repeat_ngram_size=3
                )

            else:
                dataset = predict_dataset
                predict_results = trainer.predict(
                predict_dataset,
                metric_key_prefix=" ",
                max_length=run_args.val_max_target_length,
                min_length=20,
                num_beams=run_args.num_beams,
                length_penalty=2.0,
                no_repeat_ngram_size=3
                )
                
            summary = target_data[data_type]
            count_list = json.load(open("/tmp/pycharm_project/Summ-N/icsi/stage_1/" + f"{data_type}_whole_count.json"))
            metrics = predict_results.metrics

            max_predict_samples = (
                run_args.max_predict_samples if run_args.max_predict_samples is not None else len(dataset)
            )
            metrics["predict_samples"] = min(max_predict_samples, len(dataset))

            trainer.log_metrics("predict", metrics)
            trainer.save_metrics("predict", metrics)

            if trainer.is_world_process_zero():
                if training_args.predict_with_generate:
                    predictions = tokenizer.batch_decode(
                        predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    predictions = [pred.strip() for pred in predictions]
                    final_predictions = combine(predictions, count_list)

                    assert len(final_predictions)==len(suma), "Different length" 
                    ss_pred = pd.DataFrame()
                    ss_pred['text'] = final_predictions
                    ss_pred['summary'] = suma

                    output_prediction_file ="/tmp/pycharm_project/Summ-N/icsi/summn/coarse_summary/" + f"{data_type}.csv"
                    ss_pred.to_csv(output_prediction_file, index=True)


if __name__ == "__main__":
    main()


