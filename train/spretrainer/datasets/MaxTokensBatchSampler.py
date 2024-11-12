"""
Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import Sampler, RandomSampler, SequentialSampler, Dataset
from typing import Iterator, List, Union
from sentence_transformers import SentenceTransformer

from . import BatchedLabelSampler
from spretrainer.utils import distributed, cache


class MaxTokensBatchSampler(Sampler[List[int]]):
    """
    Samples variable-size batches of indices.
    The sampler will try to allocate as many samples as possible per batch as long as the total number of
    tokens in the batch is smaller or equal to the provided maximum value `max_total_tokens`.

    This is useful because most of the times sequences are shorter than `max_seq_length` so it means we
    can allocate more samples in memory while being sure we do not require more memory than the worst case,
    that is, when all the sequences are of size `max_seq_length`, for instance, when setting
    `max_total_tokens = batch_size * max_seq_length`.

    :param data_source (Dataset): dataset to sample from
    :param model (SentenceTransformer): SentenceTransformer model to use for tokenizing the input text
    :param max_total_tokens (int): expected total maximum number of tokens per batch.
    :param shuffle (bool, str): How to shuffle the data, valid values are "label", "random", False:
        - "label": will try to shuffle samples so that batches have no repeated label (useful for contrastive learning)
        - "random": randomly shuffle the data.
        - False/None: do not shuffle, keep data as is.
    :param unique_labels (bool): If `True` .
    """
    def __init__(self, data_source: Dataset, model: SentenceTransformer,
                 max_total_tokens: int = 64 * 128, shuffle: Union[bool, str] = "label",
                 show_progress: bool = True) -> None:
        if not isinstance(max_total_tokens, int) or max_total_tokens <= 0:
            raise ValueError(f"max_total_tokens should be a positive integer value, but got max_total_tokens={max_total_tokens}")

        @cache.memory.cache
        def get_seq_lengths(tokenizer_name: str, data_source_path: str):
            """
                Return the number of tokens in each sentence in data_source.
                Nota: `tokenizer_name` and `data_source_path` are only used for
                      input -> output mapping for the cache
            """
            seq_lengths = np.empty(len(data_source), dtype=int)  # TODO: int, according to len(data_source) choose best type (int16, int32, etc.) to save space in cache
            for ix, input_example in tqdm(enumerate(data_source), desc="Tokenizing dataset",
                                      total=len(data_source), leave=True,
                                      disable=not show_progress):
                # TODO: improve (e.g. pass entire batch to process, then extract the longest per pairs of sentences)
                seq_lengths[ix] = model.tokenize(input_example.texts)['input_ids'].shape[1]
            return seq_lengths

        self.max_total_tokens = max_total_tokens

        if not distributed.is_main_process():
            distributed.barrier()  # if not the main process wait

        # TODO: improve this identifier? `tokenizer.name_or_path` is not good since resume from checkpoint changes its value (cached again as different)
        tokenizer_identifier = f"{type(model.tokenizer).__name__}_{model.tokenizer.vocab_size}_{model.tokenizer.model_max_length}"
        self.seq_lengths = get_seq_lengths(tokenizer_identifier,  # model.tokenizer.name_or_path,
                                           data_source.path if hasattr(data_source, "path") else None)

        if distributed.is_main_process():
            distributed.barrier()  # if main process, release waiting processes

        distributed.barrier()  # another barrier in case get_seq_lengths() wasn't cached (e.g. cache limit reached)

        if not shuffle:
            self.sampler = SequentialSampler(self.seq_lengths)
        elif shuffle == "random" or shuffle is True:
            self.sampler = RandomSampler(self.seq_lengths, generator=torch.default_generator)
        elif shuffle == "label":
            self.sampler = BatchedLabelSampler(data_source, batch_size=1, num_labels=1)
        else:
            raise ValueError(f"The provided value ('{shuffle}') for the `shuffle` argument is invalid. "
                             "Valid values are 'random', 'label', or False'.")

        self.batches = self.__get_batches__()

    def __get_batches__(self):
        batches = []
        current_batch = []
        current_max_seq_length = 0
        for sample_ix in self.sampler:
            seq_length = self.seq_lengths[sample_ix]
            current_max_seq_length = max(current_max_seq_length, seq_length)
            if (len(current_batch) + 1) * current_max_seq_length <= self.max_total_tokens:
                current_batch.append(sample_ix)
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [sample_ix]
                current_max_seq_length = seq_length
        if current_batch:
            batches.append(current_batch)

        # To avoid deadlocks, we force number of batches to
        # be equally distributed across all processes
        n_processes = distributed.get_world_size()
        return batches[:n_processes * (len(batches) // n_processes)]

    def __iter__(self) -> Iterator[List[int]]:
        yield from self.batches
        self.batches = self.__get_batches__()

    def __len__(self) -> int:
        return len(self.batches)
