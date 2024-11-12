"""
Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
from torch.utils.data import Dataset
from sentence_transformers import InputExample

from collections.abc import Iterable


class SimilarityDataset(Dataset):
    """

    """
    num_labels = 0

    def __init__(self, data: Iterable, is_regression: bool = False, normalize_value: bool = True):
        """data:Iterable
        normalize_value: if regression whether or not to normalize returned values
        """

        self.samples = []
        self.label2ix = {}
        self.ix2label = {}

        max_value, min_value = float("-inf"), float("inf")
        for (sent0, sent1, label) in data:
            if is_regression:
                # print("label", label)
                label = float(label)

                max_value = max(max_value, label)
                min_value = min(min_value, label)
            else:
                if label not in self.label2ix:
                    self.label2ix[label] = len(self.label2ix)
                    self.ix2label[self.label2ix[label]] = label
                label = self.label2ix[label]

            self.samples.append(InputExample(texts=[sent0, sent1], label=label))

        # if regression normalize values between [0, 1]
        if is_regression and normalize_value:
            for sample in self.samples:
                sample.label = (sample.label - min_value) / (max_value - min_value)

        self.num_labels = len(self.label2ix)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
