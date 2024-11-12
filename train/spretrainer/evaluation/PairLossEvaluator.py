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
from sentence_transformers.util import batch_to_device
from sentence_transformers.evaluation import SentenceEvaluator

logger = logging.getLogger(__name__)


class PairLossEvaluator(SentenceEvaluator):
    """
    Evaluate a model computing the loss on the provided dataset of pairs of sentences (sent_i, sent_j).

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(self, dataloader: DataLoader, loss_model: nn.Module = None, name: str = '', write_csv: bool = True, show_progress_bar: bool = True):
        """
        Constructs an evaluator for the given dataset

        :param dataloader:
            the data for the evaluation
        """
        self.dataloader = dataloader
        self.name = name
        self.loss_model = loss_model
        self.show_progress_bar = show_progress_bar

        if name:
            name = "_" + name

        self.write_csv = write_csv
        self.csv_file = f"loss_evaluation{name}_results.csv"
        self.csv_headers = ["epoch", "steps", "loss"]

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
        loss_values = []
        for batch in tqdm(self.dataloader, desc="Evaluation", disable=not self.show_progress_bar, leave=False):
            features, labels = batch
            labels = labels.to(model._target_device)
            features = list(map(lambda batch: batch_to_device(batch, model._target_device), features))
            with torch.no_grad():
                loss_values.append(self.loss_model(features, labels).item())

        score = sum(loss_values) / float(len(loss_values)) if len(loss_values) else 0

        logger.info(f"Loss: {score:.4f}\n")

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
