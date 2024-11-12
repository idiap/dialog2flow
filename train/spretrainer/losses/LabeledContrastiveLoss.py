"""
Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import torch
import logging

from torch import Tensor
from typing import Iterable, Dict, Union
from accelerate import Accelerator
from sentence_transformers import util
from sentence_transformers import SentenceTransformer

from . import BaseContrastiveLoss
from ..datasets import SimilarityDatasetFromLabels

logger = logging.getLogger(__name__)


class LabeledContrastiveLoss(BaseContrastiveLoss):
    def __init__(self, model: SentenceTransformer,
                 use_soft_labels: bool = False, temperature: float = .05,
                 soft_label_temperature: float = .35, soft_label_model: str = "multi-qa-mpnet-base-dot-v1",
                 is_symmetrical: bool = True,
                 accelerator: Accelerator = None,
                 use_contrastive_head: bool = True, use_abs: bool = False):
        """
        Soft and Vanilla Supervised Contrastive loss as described in https://arxiv.org/abs/2410.18481.
        Expects as input two texts and a label. In case of soft contrastive loss this label must be its index
        as used in the label to index mapping in `compute_label_embeddings()`.
        This is to avoid computing the label embeddings every time the loss is called.
        ).
        Then, this loss reduces the distance for all sentences embeddings in the batch having the same label (index),
        while increasing the distance among embeddings with different label.

        Args:
            model: SentenceTransformer model
            use_soft_labels: Wheather to use soft semantic labels or not (i.e. soft-contrastive loss)
            soft_label_temperature: Temperature parameter to scale the cosine similarities for the soft labels.
            soft_label_model: SentenceTransformer model to use to get the embeddings of the labels.
            temperature: Contrastive loss temperature parameter.
            is_symmetrical: Wheather to consider the loss as symmetrical between anchor and target sentences.
            accelerator: Optional Accelerator object to be used in `forward()` to gather batches across GPUs
            use_contrastive_head: Wheather to use the contrastive head or not.
            use_abs: Wheather to use the absolute value of the cosine similarity or not.

        References:
            * Paper: https://arxiv.org/abs/2410.18481

        Inputs:
            +-----------------------------------------------+------------------------------+
            | Texts                                         | Labels                       |
            +===============================================+==============================+
            | (anchor, positive/negative) pairs             | integer (label index)        |
            +-----------------------------------------------+------------------------------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer
                from torch.utils.data import DataLoader

                from spretrainer.datasets import SimilarityDatasetFromLabels
                from spretrainer.losses import LabeledContrastiveLoss


                # Our model
                my_model = SentenceTransformer(...)

                # Our supervised soft-contrastive loss
                loss_model = LabeledContrastiveLoss(model=my_model,
                                                    use_soft_labels=True)

                # Our input data
                data = [["utterance-0", "label-0"],
                        ["utterance-1", "label-1"],
                        ...
                        ["utterance-n", "label-n"]]  # (utterance, label) paris
                # Convert data to a Dataset object with InputExample()s as SentenceTransformer
                dataset = SimilarityDatasetFromLabels(data,
                                                    labels_as_ix=True,
                                                    shuffle=True)

                # We need to pre-computing label embedings for the soft-contrative loss
                loss_model.compute_label_embeddings(dataset)


                data_iterator = DataLoader(dataset, ...)

                for _ in range(n_epochs):
                    loss_model.zero_grad()
                    loss_model.train()
                    for data in data_iterator:

                        tokenized_batch, labels = data
                        loss_value = loss_model(tokenized_batch, labels)
                        loss_value.backward()
                        optimizer.step()
        """
        super(LabeledContrastiveLoss, self).__init__(model=model,
                                                     use_contrastive_head=use_contrastive_head)

        logger.info(f"Initializing labeled-contrastive loss with {'soft' if use_soft_labels else 'hard'} labels")
        if use_soft_labels:
            logger.info(f"  > Soft label temperature: {soft_label_temperature}")
            logger.info(f"  > label embedding model: {soft_label_model}")

        self.accelerator = accelerator
        self.symmetrical = is_symmetrical
        self.use_abs = use_abs
        self.use_soft_labels = use_soft_labels
        self.label2embedding = None
        self.temperature = temperature
        self.soft_label_temperature = temperature if soft_label_temperature is None else soft_label_temperature
        self.soft_label_model = soft_label_model
        self.similarity_fct = util.cos_sim
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax()

    def compute_label_embeddings(self, dataset: SimilarityDatasetFromLabels):
        if self.use_soft_labels:
            if dataset.ix2label[0].isdigit():
                # if labels are numbers makes no sense to use label embeddings...
                self.use_soft_labels = False
            else:
                self.label2embedding = SentenceTransformer(self.soft_label_model).encode(dataset.ix2label,
                                                                                         convert_to_numpy=False,
                                                                                         convert_to_tensor=True).detach().to("cpu")

    def forward(self, sentence_features: Iterable[Union[Dict[str, Tensor], Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)
                if isinstance(sentence_feature, (dict, Dict))
                else sentence_feature
                for sentence_feature in sentence_features]

        anchors, positives, labels = self.gather_batches_across_processes(reps[0], reps[1], labels)
        if self.use_abs:
            scores = self.similarity_fct(anchors, positives).abs() / self.temperature
        else:
            scores = self.similarity_fct(anchors, positives) / self.temperature

        if self.use_soft_labels:
            # TODO: if embeddings are not pre-cached, compute them on the fly and cache them as they appear
            # if self.label2embedding is None...
            if isinstance(self.label2embedding, dict):
                emb_size = next(self.label2embedding.values()).shape[1]
            else:
                emb_size = self.label2embedding.shape[1]
            label_embs = torch.zeros([labels.shape[0], emb_size])
            for label in torch.unique(labels):
                label_embs[torch.where(labels == label)] = self.label2embedding[label]
            # TODO: compute label similarity only once in compute_label_embeddings()
            #       then use the already computed values to build the ´labels_sim´ matrix
            labels_sim = util.cos_sim(label_embs, label_embs)  / self.soft_label_temperature
            targets =  torch.nn.functional.softmax(labels_sim, dim=1).to(scores.get_device())
        else:
            targets = torch.zeros_like(scores)
            for ix, label in enumerate(labels):
                targets[ix][torch.where(labels == label)[0]] = 1
            targets = targets / targets.sum(1).view(-1, 1)

        loss = self.cross_entropy_loss(scores, targets)
        if self.symmetrical:
            loss = (loss + self.cross_entropy_loss(scores.transpose(0, 1), targets)) / 2

        return self.accelerator.num_processes * loss
