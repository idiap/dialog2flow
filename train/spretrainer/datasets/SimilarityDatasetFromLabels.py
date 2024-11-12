"""
Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import random
import logging
import numpy as np

from torch.utils.data import Dataset
from sentence_transformers import InputExample

from collections.abc import Iterable
from itertools import permutations

logger = logging.getLogger(__name__)


class SimilarityDatasetFromLabels(Dataset):
    """

    """
    def __init__(self, data: Iterable, shuffle: bool = True, labels_as_ix: bool = False, balance_labels:str = 'none'):
        """
            data:Iterable
            balance_labels: 'none', 'extend', 'reduce'
        """
        if balance_labels != 'none':
            logger.info(f"Building similarity dataset from labeled instance using '{balance_labels}' strategy to balance the classes")

        data = list(data)
        self.balanced = balance_labels
        self.shuffle = shuffle
        self.y = np.array([label for _, label in data])
        self.x = [text for text, _ in data]
        self.ix2label, counts = np.unique(self.y, return_counts=True)
        self._sorted_counts = sorted(counts, reverse=True)
        if labels_as_ix:
            self.label2ix = {label:ix for ix, label in enumerate(self.ix2label)}
            self.y = np.array([self.label2ix[label] for label in self.y])
            self.labels = np.arange(self.ix2label.shape[0])
        else:
            self.label2ix = None
            self.labels = self.ix2label
        self.regenerate_pairs()

    def regenerate_pairs(self):
        self.samples = []
        max_limit = self._sorted_counts[1] if self.balanced == 'reduce' else None
        for ix, label in enumerate(self.labels):
            label_ixs = np.where(self.y == label)[0]
            sampling_ix = np.random.permutation(label_ixs)[:max_limit]
            if self.balanced == 'none' or self.balanced == 'reduce':
                for ix in range(len(sampling_ix)):
                    if ix + 1 > len(sampling_ix) - 1:
                        break
                    ix0 = sampling_ix[ix]
                    ix1 = sampling_ix[ix + 1]
                    self.samples.append(
                        InputExample(texts=[self.x[ix0], self.x[ix1]], label=label)
                    )
            elif self.balanced == 'extend':
                for ix in range(self._sorted_counts[0]):
                    if (ix + 1) % len(sampling_ix) == 0:
                        sampling_ix = np.random.permutation(label_ixs)
                    ix0 = sampling_ix[ix % len(sampling_ix)]
                    ix1 = sampling_ix[(ix + 1) % len(sampling_ix)]
                    self.samples.append(
                        InputExample(texts=[self.x[ix0], self.x[ix1]], label=label)
                    )
            else:
                raise ValueError(
                    f"Not a valid `balance_labels` value. Received {self.balanced}, expected either 'none', 'extend', or 'reduce'"
                )

        if self.shuffle:
            random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
