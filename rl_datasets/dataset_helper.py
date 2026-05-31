import os.path
import random
from typing import Callable, Optional

import numpy as np
from accelerate.commands.config.config_args import cache_dir
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, IterableDatasetDict
from transformers import AutoTokenizer
import logging
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from accelerate import PartialState

class BaseDatasetHelper:
    """
    This helper class can be used to load datasets to fine-tune a model to code snippets.
    It is optimized to datasets with code corpus
    """

    def __init__(self,
                 dataset_name: str,
                 tokenizer_name: str,
                 input_feature_name: str,
                 target_feature_name: str,
                 cache_dir: str,
                 streaming: bool = False,
                 filter_fn: Optional[Callable] = None,
                 train_size: int = 5000,
                 test_size: int = 500,
                 special_tokenization_function: Optional[Callable] = None,
                 tokenize_targets = True,
                 tokenized_dataset_path: str = None,
                 save_dir: str = "savings/tokenized_dataset",
                 tensor_type: str = "pt",
                 load_subset_feature: str = None,
                 eval_mode = False
                 ):
        """
        initiates the class

        :param dataset_name: the dataset to load. this can be either a file_name or huggingface repo
        :param tokenizer_name: pretrained tokenizer to load
        :param input_feature_name: the string features which you want to tokenize and feed it to the model @see: Example
        :param target_feature_name: the string target features which are used as Labels to train the model @see: Example
        :param tokenized_dataset_path: preprocessed dataset path to load
        :param save_dir: save path to save the tokenized dataset
        :param tensor_type: either "pt" or "jax" or "tf" to specify the return tensor of the tokenizer
        :param load_subset_feature: if the dataset has multiple languages you can specify the language to fine tune the model on

        Example:
            if you want to fine tune a model to generate code from input text using CodeSearchNet Dataset:
            input_feature_names = "func_documentation_string"
            target_feature_names = "func_code_string"
            this will fine tune the model to get an input text and generate a code
        """
        logging.basicConfig(level=logging.INFO)
        self.streaming = streaming
        self.model_input_size = 512
        self.name = dataset_name

        if load_subset_feature is None:
            self.raw_dataset = load_dataset(dataset_name, cache_dir=cache_dir, streaming=streaming)
        else:
            try:
                self.raw_dataset = load_dataset(dataset_name, load_subset_feature, cache_dir=cache_dir, streaming=streaming)
                self.subset_feature = load_subset_feature
            except Exception as e:
                print(e)
                raise " Couldn't find the feature language! Exiting"
        if streaming:
            self.raw_dataset = self.raw_dataset.shuffle(seed=41, buffer_size=10000)
        else:
            self.raw_dataset = self.raw_dataset["train"]

        if filter_fn is not None:
            self.raw_dataset = self.raw_dataset.filter(filter_fn)

        if isinstance(self.raw_dataset, Dataset):
            # Fallback for non-streaming data
            self.raw_dataset = self.raw_dataset.train_test_split(test_size=0.2)
        else:
            base_stream = self.raw_dataset["train"]

            finite_rows = list(base_stream.take(train_size + test_size))

            train_list = finite_rows[:train_size]
            test_list = finite_rows[train_size:]

            self.raw_dataset = DatasetDict({
                "train": Dataset.from_list(train_list),
                "test": Dataset.from_list(test_list)
            })


        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
        self.tokenize_targets = tokenize_targets
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

        self.input_feature_name = input_feature_name
        self.target_feature_name = target_feature_name
        self.special_tokenization_function = special_tokenization_function

        if tensor_type == "pt" or tensor_type == "jax" or tensor_type == "tf":
            self.tensor_type = tensor_type
        else:
            logging.info(" Tensor type is not defined! Using default tensor type pt")
            self.tensor_type = "pt"

        self.eval_mode = eval_mode
        self.save_dir = save_dir

        if tokenized_dataset_path is not None and os.path.exists(tokenized_dataset_path):
            self.tokenized_dataset = load_from_disk(tokenized_dataset_path)
        else:
            self.tokenized_dataset = self.tokenize_dataset()


    def _get_features(self, feature_names: list[str], example):
        """
        it is used in @tokenize_function to get the related features given the input and target names: @see __init__()

        :param feature_names: the Feature names to extract from dataset
        :param example: the Dataset batches
        :return: extracted Features
        """
        try:
            # check for multiple input or target features
            features = [example[feature_name] for feature_name in feature_names]
            sequences = []
            for idx_batch, _ in enumerate(features[0]):
                sub_sequences = []
                for num_features, _ in enumerate(features):
                    sub_sequences.append(features[num_features][idx_batch])

                sequences.append(tuple(sub_sequences))

            if len(sequences[0]) == 1:
                sequences = np.array(sequences).flatten().tolist()

        except Exception as e:
            print(e)
            raise "couldn't find the specified feature name in the Dataset!"

        return sequences

    def tokenize_function(self, batch):
        """
        tokenizes the input features

        :param batch: dataset batch
        :return: tokenized dataset batch
        """
        input_seq = self._get_features([self.input_feature_name], batch)
        target_seq = self._get_features([self.target_feature_name], batch)

        if not self.special_tokenization_function is None:
            self.special_tokenization_function(input_seq, target_seq, self.tokenizer)
            batch[self.input_feature_name] = input_seq
            batch[self.target_feature_name] = target_seq

        if self.tokenize_targets:
            inputs = self.tokenizer(input_seq, padding="max_length", truncation=True, return_tensors=self.tensor_type, max_length=self.model_input_size)
            targets = self.tokenizer(target_seq, padding="max_length", truncation=True, return_tensors=self.tensor_type, max_length=self.model_input_size)

            inputs["decoder_input_ids"] = targets["input_ids"]
            inputs["labels"] = targets["input_ids"]
        else:
            inputs = self.tokenizer(input_seq, padding=True, truncation=False, return_tensors=self.tensor_type)
        return inputs

    def tokenize_dataset(self):
        """
        tokenizes the whole dataset using @tokenize_function
        this function is enough when you want to use the Trainer API from huggingface

        :return: tokenized dataset
        """
        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        with PartialState().local_main_process_first():
            tokenized_dataset = self.raw_dataset.map(self.tokenize_function, batched=True)

        return tokenized_dataset

    def preprocess_dataset(self, format="torch"):
        """
        further adjust the dataset in order that the model can make sense of it.
        it is used for times you want to write the whole optimization function your self

        :param format: the format of the dataset
        :return: preprocessed dataset
        """
        model_features = ["attention_mask", "input_ids", "labels", "token_type_ids", "indices", "unit_tests"]

        if self.tokenized_dataset is None:
            raise ValueError("dataset isn't tokenized! this should be handled already! There might be a Problem with cache directory!")

        processed_dataset = self.tokenize_dataset()

        if "label" in processed_dataset["train"].features:
            processed_dataset = processed_dataset.rename_column("label", "labels")

        for feature in model_features:
            if not feature in processed_dataset["train"].features:
                logging.warning(f"feature {feature} not found in dataset features. It may cause problem during training if it is needed as model input!")

        for feature in processed_dataset["train"].features:
            if not (feature in model_features):
                processed_dataset = processed_dataset.remove_columns(feature)

        processed_dataset.set_format(format)

        return processed_dataset
