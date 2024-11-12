"""
Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import csv
import gzip
import json

from pathlib import Path
from typing import Union, Optional
from collections.abc import Iterator


class SimilarityDataReader:
    """
    DataReader class used to load data to train/evaluate sentence-embeddings.
    Data is expected to be a collection of either sentence pairs (sentence1, sentence2)
    or triples (sentence1, sentence2, value) where value is the ground truth value (either
    a label or similarity value).
    """
    @staticmethod
    def read_csv(
        path: str,
        delimiter: str = ",",
        col_sent0: Union[str, int] = "sent1", col_sent1: Union[str, int] = "sent2",
        col_label: Optional[str] = "value",
        col_split: str = "split", use_split: Optional[str] = None,
        label_optinal: bool = False, preprocess_fn: Optional[callable] = None,
        ignore_header: bool = False, encoding: str = 'utf8'
    ) -> Iterator[tuple[str, str, Optional[Union[str, float]]]]:
        """
        Lazily read (sent1, sent1, value) values from a CSV files.
        """
        if isinstance(path, (Path, str)):
            path = [path]

        # We are using part of the path to provide the balancing strategy
        # e.g. "path/to/data/file.csv:extend" so we need to remove it
        path = [p.split(":")[0] for p in path]

        for path_csv in path:

            # TODO: improve "if type == gzip" detection
            f_open = gzip.open if path_csv.endswith(".gz") else open

            with f_open(path_csv, 'rt', encoding=encoding) as reader:
                if isinstance(col_sent0, str):
                    csv_file = csv.DictReader(reader, delimiter=delimiter)
                else:
                    csv_file = csv.reader(reader, delimiter=delimiter)

                for row in csv_file:
                    if not use_split or row[col_split] == use_split:
                        if isinstance(row, dict):
                            label = row[col_label] if col_label and col_label in row else None
                        else:
                            if ignore_header:
                                ignore_header = False
                                continue
                            label = row[col_label]

                        if col_sent1 is not None:
                            if preprocess_fn is None:
                                yield (row[col_sent0], row[col_sent1], label)
                            else:
                                yield (preprocess_fn(row[col_sent0]), preprocess_fn(row[col_sent1]), label)
                        else:
                            yield (row[col_sent0] if preprocess_fn is None else preprocess_fn(row[col_sent0]), label)

    @staticmethod
    def read_docs(
        paths: Union[Union[Path, str], list[Union[Path, str]]],
        lines_are_documents: bool = True,
        encoding: str = "utf-8",
    ) -> Iterator[str]:
        """
        Lazily read in contents of files.
        """
        if isinstance(paths, (Path, str)):
            paths = [paths]

        for path in paths:
            with open(path, "r", encoding=encoding) as infile:
                if lines_are_documents:
                    for text in infile:
                        text = text.strip()
                        if text:
                            yield text
                else:
                    text = infile.read().strip()
                    if text:
                        yield text

    @staticmethod
    def read_jsonl(
        path: str,
        key_sent0: str = "sent1", key_sent1: str = "sent2",
        key_label: Optional[str] = "value",
        key_split: str = "split", use_split: Optional[str] = None,
        label_optinal: bool = False, preprocess_fn: Optional[callable] = None,
        encoding: str = 'utf8'
    ) -> Iterator[tuple[str, str, Optional[Union[str, float]]]]:
        """
        Lazily read in contents of jsonlist files.
        """
        if isinstance(path, (Path, str)):
            path = [path]

        for path_jsonl in path:
            # TODO: improve "if type == gzip" detection
            f_open = gzip.open if path_jsonl.endswith(".gz") else open
            with f_open(path_jsonl, "r", encoding=encoding) as infile:
                for line in infile:
                    if not line:
                        continue
                    row = json.loads(line)
                    if not use_split or row[key_split] == use_split:
                        label = row[key_label] if key_label and key_label in row else None

                        if preprocess_fn is None:
                            yield (row[key_sent0], row[key_sent1], label)
                        else:
                            yield (preprocess_fn(row[key_sent0]), preprocess_fn(row[key_sent1]), label)
