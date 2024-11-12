"""
Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import os
import csv
import logging
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report

from sentence_transformers import InputExample
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import batch_to_device, cos_sim

logger = logging.getLogger(__name__)


def compute_anisotropy(embs, use_abs=False):
    sim = cos_sim(embs, embs)
    sim.fill_diagonal_(0)

    if use_abs:
        return (sim.sum().abs() / (sim.shape[0] ** 2 - sim.shape[0])).item()
    else:
        return (sim.sum() / (sim.shape[0] ** 2 - sim.shape[0])).item()


class FewShotClassificationEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on similarity-based few-shot learning classification on a
    labeled dataset. The few-shot learning is carried out by using the simple method
    introduced in "Prototypical Networks for Few-shot Learning" by Snell et al.

    In simply words, the "training" consist of computing a prototype embedding for each category by
    averaging the embedding of all the (few) training samples that belong to this category.
    Then, a sample is classified into the category whose prototype embedding is the most similar to its own.

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(self, dataset: Dataset, n_shots: int = 5, num_runs: int = 10,
                 metric: str = "accuracy", metric_avg: str = "macro", batch_size: int = 64,
                 name: str = "", write_csv: bool = True, show_progress_bar: bool = True):
        """
        Constructs an evaluator for the given dataset

        :param dataset: the dataset used for training and evaluation. During evaluation training set will be created by
                        sampling `n_shots` for each class, the rest will be used as evaluation set.
        :param n_shots: number of shots to use for training (default 5-shot).
        :param num_runs: Number of independent experiments to run before returning the metric value (avg. value accross experiments)
        """
        self.name = name
        self.n_shots = n_shots
        self.num_runs = num_runs
        self.metric = metric
        self.metric_avg = f"{metric_avg} avg"
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size

        self._labels = np.array([ie.label for ie in dataset])
        self._samples = np.array([ie.texts[0] for ie in dataset])
        del dataset

        self._labels_unique = np.unique(self._labels)

        self.dataset = [None] * len(self._labels_unique)
        for label_ix, label in enumerate(self._labels_unique):
            self.dataset[label_ix] = self._samples[self._labels == label]

        if name:
            name = "_" + name

        self.metric_name = None
        self.write_csv = write_csv
        self.csv_file = f"{metric_avg}_{metric}_few_shot_evaluation{name}_results.csv"
        self.csv_headers = ["epoch", "steps", f"{metric} ({self.metric_avg})",
                            "intra anisotropy", "inter anisotropy"]

    def get_embedding_label_pairs(self, model, dataloader):
        embeddings = []
        labels = []
        for batch in tqdm(dataloader, desc="Computing embeddings", disable=not self.show_progress_bar, leave=False):
            features, label_ids = batch
            features = batch_to_device(features[0], model.device)
            embeddings.append(model(features)['sentence_embedding'].detach().cpu().numpy())
            labels.append(label_ids.detach().numpy())

        return normalize(np.concatenate(embeddings)), np.concatenate(labels)  # (embeddings, labels)

    def evaluate_fewshot(self, model, return_report = False):
        # Creating train and eval set dynamically...
        trainset, evalset = [], []
        for label, samples in enumerate(self.dataset):
            np.random.shuffle(samples)
            x_train_label, x_eval_label = samples[:self.n_shots], samples[self.n_shots:]

            trainset.extend([InputExample(texts=[str(t)], label=label) for t in x_train_label])
            evalset.extend([InputExample(texts=[str(t)], label=label) for t in x_eval_label])

        train_loader = DataLoader(trainset, shuffle=False, batch_size=self.batch_size)
        eval_loader = DataLoader(evalset, shuffle=False, batch_size=self.batch_size)
        del trainset, evalset

        train_loader.collate_fn = model.smart_batching_collate
        eval_loader.collate_fn = model.smart_batching_collate

        # Computing support embeddings with training set
        support_embeddings, support_labels = self.get_embedding_label_pairs(model, train_loader)

        # Computing the prototype embeddings for each class/label
        num_labels = len(self.dataset)
        prototype_embeddings = np.zeros([num_labels, support_embeddings.shape[1]])
        for cat in range(num_labels):
            if len(np.where(support_labels == cat)[0]):
                prototype_embeddings[cat] = support_embeddings[np.where(support_labels == cat)[0]].mean(axis=0)
            else:
                prototype_embeddings[cat] = np.zeros(support_embeddings.shape[1])

        # Computing query embeddings with evaluation set
        x_eval_embedding, y_eval_true = self.get_embedding_label_pairs(model, eval_loader)

        # Classifying evaluation samples by distance to prototype
        sim_matrix = x_eval_embedding @ prototype_embeddings.T
        y_eval_pred = sim_matrix.argmax(axis=1)

        # Computing evaluation metrics
        report = classification_report(y_eval_true, y_eval_pred, output_dict=True, zero_division=0)

        if return_report:
            return report

        return report["accuracy"] if self.metric == "accuracy" else report[self.metric_avg][self.metric]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logger.info(f"Evaluation on the {self.name} dataset ({self.num_runs} times)" + out_txt)

        # Repeating the evaluation `self.num_runs` with different train/eval splits
        # and then reporting the average.
        score = 0
        for _ in tqdm(range(self.num_runs), desc="Few-shot Classification", leave=False):
            score += self.evaluate_fewshot(model)
        score /= self.num_runs

        metric_avg = '' if self.metric == "accuracy" else f"({self.metric_avg})"
        logger.info(f"  * {self.metric.capitalize()}{metric_avg}: {score:.4f}")

        embs = model.encode(self._samples, show_progress_bar=True, batch_size=self.batch_size)

        # Anisotropy computation
        label_centroids = np.zeros((self._labels_unique.shape[0], embs.shape[1]))
        intra_label_anisotropy = []
        for ix, label in enumerate(self._labels_unique):
            label_embs = embs[self._labels == label]
            label_centroids[ix] = label_embs.mean(axis=0)
            if label_embs.shape[0] > 2:
                # Compute intra-label anisotropy
                intra_label_anisotropy.append(compute_anisotropy(label_embs))
        intra = sorted(intra_label_anisotropy)[len(intra_label_anisotropy) // 2] if intra_label_anisotropy else None
        inter = compute_anisotropy(label_centroids)
        logger.info(f"  * Anisotropy ({len(self._labels_unique)} unique labels):")
        logger.info(f"    * Intra-label (↑): {intra:.4f}")
        logger.info(f"    * Inter-label (↓): {inter:.4f}\n")

        # Saving results
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline='', mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, score, intra, inter])
            else:
                with open(csv_path, newline='', mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, score, intra, inter])

        return score
