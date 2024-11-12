"""
Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import torch
import pytest
import random
import numpy as np

from transformers import AutoTokenizer

from sentence_transformers import InputExample
from spretrainer.datasets import SimilarityDataReader, SimilarityDataset, \
                                 SimilarityDatasetContrastive, SimilarityDatasetFromLabels
from spretrainer.datasets import BatchedLabelSampler, MaxTokensBatchSampler

SEED = 13
PATH_DATASET_LABEL = "tests/data/dataset_labels.csv"
PATH_DATASET_LABEL_SPLITS = "tests/data/dataset_labels_splits.csv"
PATH_DATASET_REGRESSION = "tests/data/dataset_regression.csv"
PATH_DATASET_DIALOGUE = "tests/data/dataset_dialogue.csv"
PATH_DATASET_DIALOGUE_CONTRASTIVE = "tests/data/dataset_dialogue_contrastive.csv"
PATH_DATASET_LABELED = "tests/data/dataset_text_labels.csv"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class DummySentenceTransformer:
    def __init__(self, model_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(self, texts):
        return self.tokenizer(texts, return_tensors="pt")


def test_dataset_raw():
    # Error cases
    with pytest.raises(KeyError):
        data = SimilarityDataset(
            SimilarityDataReader.read_csv(PATH_DATASET_LABEL, col_sent0="sentence1", col_sent1="sentence2")
        )
    with pytest.raises(KeyError):
        data = SimilarityDataset(
            SimilarityDataReader.read_csv(PATH_DATASET_LABEL, delimiter='\t')
        )

    # Load while dataset as is
    data = SimilarityDataset(
        SimilarityDataReader.read_csv(PATH_DATASET_LABEL, col_sent0="sent1", col_sent1="sent2")
    )
    assert len(data) == 8
    assert data[0].texts == ["hello", "world"]
    assert data.ix2label[data[0].label] == "positive"
    assert data[-2].texts == ["jsalt", "ai"]
    assert data.ix2label[data[-2].label] == "positive"

    # Load only a subset, the data for a given split ("train")
    data = SimilarityDataset(
        SimilarityDataReader.read_csv(PATH_DATASET_LABEL_SPLITS,
                                      col_sent0="sent1", col_sent1="sent2",
                                      col_split="split", use_split="train")
    )
    assert len(data) == 4
    assert data[-1].texts == ["apple", "car"]
    assert data.ix2label[data[-1].label] == "negative"

    # Load regression data (i.e. ground truth is a number) with normalized values (default)
    data = SimilarityDataset(
        SimilarityDataReader.read_csv(PATH_DATASET_REGRESSION, col_sent0="sent1", col_sent1="sent2"),
        is_regression=True
    )
    assert len(data) == 7
    assert data[0].texts == ["hello", "world"]
    # check that values are returned normalized
    assert data[0].label == 5 / 5
    assert data[-1].label == 3 / 5

    # Load regression data without normalizing values (raw values)
    data = SimilarityDataset(
        SimilarityDataReader.read_csv(PATH_DATASET_REGRESSION, col_sent0="sent1", col_sent1="sent2"),
        is_regression=True, normalize_value=False
    )
    assert len(data) == 7
    # check that values are returned as given
    assert data[0].label == 5
    assert data[-1].label == 3
    assert isinstance(data[-1].label, float)


def test_dataset_contrastive():
    # with pytest.raises(KeyError):
    #     data = SimilarityDatasetContrastive(PATH_DATASET_LABEL)
    with pytest.raises(KeyError):
        data = SimilarityDatasetContrastive(
            SimilarityDataReader.read_csv(PATH_DATASET_LABEL, delimiter='\t')
        )

    # positive and explicit negatives
    data = SimilarityDatasetContrastive(
        SimilarityDataReader.read_csv(PATH_DATASET_LABEL, col_sent0="sent1", col_sent1="sent2", col_label="value"),
        label_pos="positive", label_neg="negative"
    )

    assert len(data) == 6
    assert data[0].texts == ["hello", "world", "moon"]
    assert data[1].texts == ["world", "hello", "moon"]
    assert data[2].texts == ["apple", "orange", "car"]
    assert data[3].texts == ["orange", "apple", "car"]
    assert data[4].texts == ["orange", "apple", "truck"]
    assert data[5].texts == ["apple", "orange", "truck"]

    # only positive pairs (non-existing label colum) other pairs will be considered as negative (inside the batch)
    data = SimilarityDatasetContrastive(
        SimilarityDataReader.read_csv(PATH_DATASET_LABEL, col_sent0="sent1", col_sent1="sent2", col_label="non-existing")
    )
    assert len(data) == 8
    assert data[0].texts == ["hello", "world"]
    assert data[1].texts == ["hello", "moon"]
    assert data[2].texts == ["apple", "orange"]
    assert data[3].texts == ["apple", "car"]
    assert data[4].texts == ["orange", "truck"]
    assert data[5].texts == ["person", "dog"]
    assert data[6].texts == ["jsalt", "ai"]

    data = SimilarityDatasetContrastive(
        SimilarityDataReader.read_csv(PATH_DATASET_LABEL, col_sent0="sent1", col_sent1="sent2", col_label=None)
    )
    assert len(data) == 8


def test_samplers():
    data = SimilarityDataReader.read_csv(PATH_DATASET_DIALOGUE, col_sent0="turn", col_sent1=None, col_label="dialogue_id")
    data = [InputExample(texts=[text], label=label) for text, label in data]
    sampler = BatchedLabelSampler(data, batch_size=5, num_labels=2)

    assert len(sampler) == len(data)

    sampling = [ix for ix in sampler]
    assert data[sampling[0]].label == data[sampling[1]].label
    assert data[sampling[1]].label != data[sampling[2]].label
    assert data[sampling[2]].label == data[sampling[3]].label
    assert [ix for ix in sampler] != [ix for ix in sampler]

    model = DummySentenceTransformer("bert-base-uncased")
    batch_size, max_seq_length = 2, 64
    # When shuffle is True / "random"
    batch_sampler = MaxTokensBatchSampler(data, model,
                                          max_total_tokens=batch_size * max_seq_length,
                                          shuffle=True)
    batches_0 = [batch_ixs for batch_ixs in batch_sampler]
    batches_1 = [batch_ixs for batch_ixs in batch_sampler]
    assert len(batches_0) == 5  # num total batches == 4
    assert len(batches_0[0]) != len(batches_0[2])  # variable-size batches

    assert batches_0 != batches_1

    # When shuffle is "label"
    batch_sampler = MaxTokensBatchSampler(data, model,
                                          max_total_tokens=batch_size * max_seq_length,
                                          shuffle="label")
    batches_0 = [batch_ixs for batch_ixs in batch_sampler]
    batches_1 = [batch_ixs for batch_ixs in batch_sampler]
    # Check indexes in the same batch correspond to different labels
    assert data[batches_0[0][0]].label != data[batches_0[0][1]].label
    assert data[batches_0[0][1]].label != data[batches_0[0][2]].label
    assert batches_0 != batches_1  # is shuffle working?

    # When shuffle is False, must be equal
    batch_sampler = MaxTokensBatchSampler(data, model,
                                          max_total_tokens=batch_size * max_seq_length,
                                          shuffle=False)
    batches_0 = [batch_ixs for batch_ixs in batch_sampler]
    batches_1 = [batch_ixs for batch_ixs in batch_sampler]

    assert batches_0[0][0] == 0
    assert batches_0[0][1] == 1
    assert batches_0[-1][-1] == len(data) - 1

    assert batches_0 == batches_1

    # When shuffle is "label" (only one label)
    for sample in data:
        sample.label = 1
    batch_sampler = MaxTokensBatchSampler(data, model,
                                          max_total_tokens=batch_size * max_seq_length,
                                          shuffle="label")
    batches_0 = [batch_ixs for batch_ixs in batch_sampler]
    batches_1 = [batch_ixs for batch_ixs in batch_sampler]
    assert batches_0 != batches_1


def test_dataset_labeled():
    # Shuffle off
    data = SimilarityDatasetFromLabels(
        SimilarityDataReader.read_csv(PATH_DATASET_LABELED,
                                      col_sent0="sent", col_sent1=None,
                                      col_label="value"),
        shuffle=False
    )
    assert len(data) == 8
    assert data[1].label == data[2].label
    assert data[1].texts[1] == data[2].texts[0]

    old_x = data[0].texts
    data.regenerate_pairs()
    assert old_x != data[0].texts

    # Shuffle on
    data = SimilarityDatasetFromLabels(
        SimilarityDataReader.read_csv(PATH_DATASET_LABELED,
                                      col_sent0="sent", col_sent1=None,
                                      col_label="value"),
        shuffle=True,
        balance_labels='none'
    )
    old_y = data[0].label
    data.regenerate_pairs()
    assert old_y != data[0].label

    # Original labels
    data = SimilarityDatasetFromLabels(
        SimilarityDataReader.read_csv(PATH_DATASET_LABELED,
                                      col_sent0="sent", col_sent1=None,
                                      col_label="value"),
    )
    # check label is a string
    assert isinstance(data[0].label, str)

    # Index labels
    data = SimilarityDatasetFromLabels(
        SimilarityDataReader.read_csv(PATH_DATASET_LABELED,
                                      col_sent0="sent", col_sent1=None,
                                      col_label="value"),
        labels_as_ix=True
    )
    # check label is an integer (not longer a string)
    assert not isinstance(data[0].label, str)

    # Dataset balancing strategies
    # No strategy (default)
    # Shuffle on
    data = SimilarityDatasetFromLabels(
        SimilarityDataReader.read_csv(PATH_DATASET_LABELED,
                                      col_sent0="sent", col_sent1=None,
                                      col_label="value"),
        balance_labels='none'
    )
    assert sum(data._sorted_counts) - len(data._sorted_counts)  # not balanced

    # balanced dataset by expanding smaller label groups
    data = SimilarityDatasetFromLabels(
        SimilarityDataReader.read_csv(PATH_DATASET_LABELED,
                                      col_sent0="sent", col_sent1=None,
                                      col_label="value"),
        balance_labels='extend'
    )
    assert len(data) == data._sorted_counts[0] * len(data.labels)

    # balanced reducing the greatest label groups to match the second greatest one
    data = SimilarityDatasetFromLabels(
        SimilarityDataReader.read_csv(PATH_DATASET_LABELED,
                                      col_sent0="sent", col_sent1=None,
                                      col_label="value"),
        shuffle=True,
        balance_labels='reduce'
    )
    assert len(data) == sum(data._sorted_counts[1:]) + data._sorted_counts[1] - len(data._sorted_counts)
