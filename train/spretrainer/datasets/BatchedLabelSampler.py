"""
Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import numpy as np

from tqdm import tqdm
from torch.utils.data import Sampler
from typing import Iterator, Sized

from spretrainer.utils import distributed, cache


class BatchedLabelSampler(Sampler[int]):
    """
    Samples elements randomly but trying to have only `num_labels` different labels inside each batch.

    :param data_source (Dataset): dataset to sample from
    :param batch_size (int): number of samples per batch
    :param num_labels (int): expected number of unique labels per batch
    """
    def __init__(self, data_source: Sized, batch_size: int = 128, num_labels: int = None) -> None:
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")

        if not isinstance(num_labels, int) or num_labels <= 0:
            raise ValueError(f"num_labels should be a positive integer value, but got num_labels={num_labels}")

        @cache.memory.cache
        def get_label2sample_ixs(labels: np.ndarray):
            labels_unique = np.unique(labels)
            label2sample_ixs = [None] * len(labels_unique)
            # TODO: make it more efficiente! (it takes close to 3 hours with 6M samples in all_dataset)
            for label_ix, label in tqdm(enumerate(labels_unique),
                                        desc="Group indexes by label", total=len(labels_unique)):
                label2sample_ixs[label_ix] = np.where(labels == label)[0]
            return label2sample_ixs

        self.samples_per_batches = batch_size // num_labels
        self.num_samples = len(data_source)

        labels = np.array([ie.label for ie in data_source])  # TODO: set proper dtype according to len(data_source)

        labels_unique = np.unique(labels)
        if len(labels_unique) > 1:

            if not distributed.is_main_process():
                distributed.barrier()  # if not the main process wait
            self.label2sample_ixs = get_label2sample_ixs(labels)
            if distributed.is_main_process():
                distributed.barrier()  # if main process, release waiting processes
        else:
            self.label2sample_ixs = None
            self.samples_per_batches = 1

    def __iter__(self) -> Iterator[int]:
        if self.label2sample_ixs is None:
            sample_indexes = np.random.permutation(self.num_samples)
        else:
            label2sample_ixs = self.label2sample_ixs.copy()
            for label_ix, indexes in enumerate(label2sample_ixs):
                if self.samples_per_batches == 1:
                    label2sample_ixs[label_ix] = iter(np.random.permutation(indexes))
                else:
                    label2sample_ixs[label_ix] = iter(np.split(np.random.permutation(indexes),
                                                               range(self.samples_per_batches,
                                                                     len(indexes),
                                                                     self.samples_per_batches)))

            sample_indexes = []
            # TODO: improve implementation, not efficient!
            while label2sample_ixs:

                for label_ix in np.random.permutation(len(label2sample_ixs)):
                    if label2sample_ixs[label_ix]:
                        try:
                            sample_indexes.append(next(label2sample_ixs[label_ix]))
                        except StopIteration:
                            label2sample_ixs[label_ix] = None
                        # sample_indexes.append(label2sample_ixs[label_ix].pop())

                label2sample_ixs = [ixs for ixs in label2sample_ixs if ixs]

        yield from sample_indexes if self.samples_per_batches == 1 else np.concatenate(sample_indexes)

    def __len__(self) -> int:
        return self.num_samples
