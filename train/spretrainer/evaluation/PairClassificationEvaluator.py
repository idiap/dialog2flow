"""
Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import os
import csv
import torch
import logging

from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sentence_transformers.util import batch_to_device
from sentence_transformers.evaluation import SentenceEvaluator

logger = logging.getLogger(__name__)


class PairClassificationEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on classification of sentence pairs (sent_i, sent_j) on a
    labeled dataset such as in NLI tasks.

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(self, dataloader: DataLoader, metric: str = "accuracy", metric_avg: str = "macro",
                 name: str = "", softmax_model: nn.Module = None,
                 write_csv: bool = True, show_progress_bar: bool = True):
        """
        Constructs an evaluator for the given dataset

        :param dataloader:
            the data for the evaluation
        """
        self.dataloader = dataloader
        self.name = name
        self.softmax_model = softmax_model
        self.metric = metric
        self.metric_avg = f"{metric_avg} avg"
        self.show_progress_bar = show_progress_bar

        if name:
            name = "_" + name

        self.write_csv = write_csv
        self.csv_file = f"{metric_avg}_{metric}_evaluation{name}_results.csv"
        self.csv_headers = ["epoch", "steps", f"{metric} ({self.metric_avg})"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logger.info(f"Evaluation on the {self.name} dataset" + out_txt)
        self.dataloader.collate_fn = model.smart_batching_collate
        y_true, y_pred = [], []
        for batch in tqdm(self.dataloader, desc="Evaluation", disable=not self.show_progress_bar, leave=False):
            features, label_ids = batch
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], model.device)
            label_ids = label_ids.to(model.device)
            with torch.no_grad():
                _, prediction = self.softmax_model(features, labels=None)
            y_true.extend(torch.argmax(prediction, dim=1).tolist())
            y_pred.extend(label_ids.tolist())

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        if self.metric == "accuracy":
            score = report["accuracy"]
            metric_avg = ''
        else:
            score = report[self.metric_avg][self.metric]
            metric_avg = f"({self.metric_avg})"

        logger.info(f"{self.metric.capitalize()}{metric_avg}: {score:.4f}\n")

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline='', mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, score])
            else:
                with open(csv_path, newline='', mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, score])

        return score
