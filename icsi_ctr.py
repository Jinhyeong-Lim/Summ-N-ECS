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
    BartForConditionalGeneration, LEDForConditionalGeneration,
    DataCollatorForSeq2Seq, Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, HfArgumentParser, set_seed,
    EarlyStoppingCallback, AutoTokenizer
)
from typing import Optional
import json
from tools import get_dataloader
from datasets import load_dataset
from nltk import sent_tokenize


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
    max_target_length: Optional[int] = field(default=600)
    num_beams: Optional[int] = field(default=None) 
    ignore_pad_token_for_loss: bool = field(default=True)
    overwrite_cache: bool = field(default=False)
    preprocessing_num_workers: Optional[int] = field(default=None)
    val_max_target_length: Optional[int] = field(default=None)
    pad_to_max_length: bool = field(default=True)
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )



def main():
    parser = HfArgumentParser((Seq2SeqTrainingArguments, RunArguments))
    training_args, run_args = parser.parse_args_into_dataclasses()

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

    raw_datasets = load_dataset(extension, data_files=data_files)

    if 'dia' in run_args.model_name:
        tokenizer = AutoTokenizer.from_pretrained('MingZhong/DialogLED-large-5120')
        models = LEDForConditionalGeneration.from_pretrained('MingZhong/DialogLED-large-5120')
    else:
        tokenizer = AutoTokenizer.from_pretrained(run_args.model_name)
        models = LEDForConditionalGeneration.from_pretrained(run_args.model_name)

    if training_args.do_train:
        column_names = raw_datasets["train"].column_names

    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names

    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names


    padding = "max_length" if run_args.pad_to_max_length else False
    last_checkpoint = None
    max_target_length = run_args.max_target_length


    class collator(DataCollatorForSeq2Seq):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, features, return_tensors=None):
            ctr_attention_mask = []
            ctr_input_ids = []
            dialogue = []
            summary = []
            id = []

            if "ctr_input_ids" in features[0].keys():
                for i in range(len(features)):
                    dialogue.append(features[i].pop("text"))
                    summary.append(features[i].pop("summary"))

            if return_tensors is None:
                return_tensors = self.return_tensors
            labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None

            if labels is not None:
                max_label_length = max(len(l) for l in labels)
                if self.pad_to_multiple_of is not None:
                    max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )

                padding_side = self.tokenizer.padding_side
                for feature in features:
                    remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                    if isinstance(feature["labels"], list):
                        feature["labels"] = (
                            feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                        )
                    elif padding_side == "right":
                        feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                    else:
                        feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

            features = self.tokenizer.pad(
                features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )

            # prepare decoder_input_ids
            if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
            ):
                decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
                features["decoder_input_ids"] = decoder_input_ids
            

            if len(ctr_attention_mask) > 0:
                features["ctr_attention_mask"] = ctr_attention_mask
                features["ctr_input_ids"] = ctr_input_ids
            
            return features


    def preprocess_function(examples):
        inputs = examples['text']
        inputs = [x.strip().lower() for x in inputs]
        ctr = None
        if " </s></s> " in inputs[0]:
            inputs1 = [x.split("</s></s>")[0].lower().strip() for x in inputs]
            ctr = [x.split("</s></s>")[1].lower().strip() for x in inputs]
            inputs = inputs1

        else:
            print("?????")

        targets = examples['summary']
        targets = [x.lower() for x in targets]

        model_inputs = tokenizer(inputs, max_length=run_args.max_source_length, padding=padding, truncation=True)
        if ctr:
            ctr_model_inputs = tokenizer(ctr, max_length=run_args.max_source_length, padding=padding, truncation=True)
            model_inputs["ctr_input_ids"] = ctr_model_inputs["input_ids"]
            model_inputs["ctr_attention_mask"] = ctr_model_inputs["attention_mask"]
    
        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and run_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        if "Unnamed: 0" in model_inputs:
            kkk = model_inputs.pop("Unnamed: 0")
        return model_inputs


    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]
    predict_dataset = raw_datasets["test"]


    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")

        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                desc="Running tokenizer on train dataset",
                load_from_cache_file=False
            )

    if training_args.do_eval:
        max_target_length = run_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")

        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function_eval,
                batched=True,
                load_from_cache_file=False,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = run_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")

        with training_args.main_process_first(desc="train dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function_eval,
                batched=True,
                load_from_cache_file=False,
                desc="Running tokenizer on test dataset",
            )


    # Data collator
    label_pad_token_id = -100 if run_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = collator(tokenizer, model=models)
    
    metric = evaluate.load("rouge")


    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
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
                                references=decoded_labels, tokenizer=lambda
                x: tokenizer.tokenize(x), use_aggregator=False)

        result = {key: statistics.mean(value) * 100 for key, value in
                  result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}

        return result


    class BartTrainer(Seq2SeqTrainer):
        def __init__(self, loss_name, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def get_train_dataloader(self):
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")

            train_dataset = self.train_dataset
            data_collator = self.data_collator

            if isinstance(train_dataset, torch.utils.data.IterableDataset):
                if self.args.world_size > 1:
                    train_dataset = IterableDatasetShard(
                        train_dataset,
                        batch_size=self._train_batch_size,
                        drop_last=self.args.dataloader_drop_last,
                        num_processes=self.args.world_size,
                        process_index=self.args.process_index,
                    )

                return DataLoader(
                    train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    collate_fn=data_collator,
                    num_workers=0,
                    pin_memory=self.args.dataloader_pin_memory,
                )

            train_sampler = self._get_train_sampler()

            return DataLoader(
                train_dataset,
                batch_size=self._train_batch_size,
                sampler=train_sampler,
                collate_fn=data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=0,
                pin_memory=self.args.dataloader_pin_memory,
            )

        def compute_loss(self, model, inputs, return_outputs=False):
                ctr_inputs = {}
                m = nn.Softmax(dim=0)
                if self.label_smoother is not None and "labels" in inputs:
                    labels = inputs.pop("labels")
                    
                else:
                    labels = None
                
                if "ctr_input_ids" in inputs:
                    id = inputs.pop("Unnamed: 0")
                    ctr_inputs["input_ids"] = inputs.pop("ctr_input_ids")
                    ctr_inputs["attention_mask"] = inputs.pop("ctr_attention_mask")
                    ctr_inputs["decoder_input_ids"] = inputs.pop("decoder_input_ids")
                    inputs["decoder_input_ids"] = ctr_inputs["decoder_input_ids"]

                outputs = model(**inputs)
                if "ctr_input_ids" in inputs:
                    ctr_outputs = model(**ctr_inputs)

                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index]

                if labels is not None:
                    high_loss = self.label_smoother(outputs, labels)
                    if "ctr_input_ids" in inputs:
                        low_loss = self.label_smoother(ctr_outputs, labels)
                        delta = 1.0
                        ctr_loss = max(0, delta - (low_loss/(high_loss+low_loss) - high_loss/(high_loss+low_loss)))
                
                else:
                    if isinstance(outputs, dict) and "loss" not in outputs:
                        raise ValueError(
                            "The model did not return a loss from the inputs, only the following keys: "
                            f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                        )
                    # We don't use .loss here since the model may return tuples instead of ModelOutput.
                    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

                loss = high_loss * 1.0 + ctr_loss * 1.0

                return (loss, outputs) if return_outputs else loss

    

    # Initialize our Trainer
    trainer = BartTrainer(
        model=models,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
    )


    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train()
        trainer.save_model()  

        metrics = train_result.metrics
        max_train_samples = (
            run_args.max_train_samples if run_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(
            max_length=run_args.val_max_target_length, 
            num_beams=run_args.num_beams,
            min_length=20,
            metric_key_prefix=" ", 
            length_penalty=2.0,
            no_repeat_ngram_size=3
        )
        max_eval_samples = run_args.max_eval_samples if run_args.max_eval_samples is not None else len(eval_dataset)
        max_eval_samples = len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


    if training_args.do_predict:
        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix=" ",
            max_length=run_args.val_max_target_length,
            min_length=20,
            num_beams=6,
            length_penalty=2.0,
            no_repeat_ngram_size=3
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            run_args.max_predict_samples if run_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)


if __name__ == "__main__":
    main()


