"""
Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import torch

from torch import nn, Tensor
from typing import Iterable, Dict, Union
from accelerate import Accelerator
from sentence_transformers import util
from sentence_transformers import SentenceTransformer

from . import BaseContrastiveLoss


# Modified from the original code taken from sentence_bert.losses.MultipleNegativesRankingLoss
# (https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesSymmetricRankingLoss.py)
class MultipleNegativesSymmetricRankingLoss(BaseContrastiveLoss):
    """
        This loss is an adaptation of MultipleNegativesRankingLoss. MultipleNegativesRankingLoss computes the following loss:
        For a given anchor and a list of candidates, find the positive candidate.

        In MultipleNegativesSymmetricRankingLoss, we add another loss term: Given the positive and a list of all anchors,
        find the correct (matching) anchor.

        For the example of question-answering: You have (question, answer)-pairs. MultipleNegativesRankingLoss just computes
        the loss to find the answer for a given question. MultipleNegativesSymmetricRankingLoss additionally computes the
        loss to find the question for a given answer.

        Note: If you pass triplets, the negative entry will be ignored. A anchor is just searched for the positive.

        Example::

            from sentence_transformers import SentenceTransformer, losses, InputExample
            from torch.utils.data import DataLoader

            model = SentenceTransformer('distilbert-base-uncased')
            train_examples = [InputExample(texts=['Anchor 1', 'Positive 1']),
                InputExample(texts=['Anchor 2', 'Positive 2'])]
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
            train_loss = losses.MultipleNegativesSymmetricRankingLoss(model=model)
    """
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct: callable = util.cos_sim, accelerator: Accelerator = None, use_contrastive_head: bool = False):
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: Similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        :param accelerator: Optional Accelerator object to be used in `forward()` to gather batches across GPUs
        """
        super(MultipleNegativesSymmetricRankingLoss, self).__init__(model, use_contrastive_head)
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.accelerator = accelerator
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, sentence_features: Iterable[Union[Dict[str, Tensor], Tensor]], labels: Tensor):

        reps = [self.model(sentence_feature)
                if isinstance(sentence_feature, (dict, Dict))
                else sentence_feature
                for sentence_feature in sentence_features]

        anchors, positives, _ = self.gather_batches_across_processes(reps[0], reps[1])

        # if there are user-provided hard negatives n_i, i.e. when triplets are passed (a_i, p_i, n_i)
        # instead of only positive pairs, then, gather those negatives...
        if len(reps[2:]) > 0:
            hard_negatives = self.gather_batches_across_processes_single(torch.cat(reps[2:]))
            scores = self.similarity_fct(anchors, torch.cat([positives, hard_negatives])) * self.scale
        else:
            scores = self.similarity_fct(anchors, positives) * self.scale

        labels = torch.arange(len(anchors), dtype=torch.long, device=scores.device)  # (a_i, b_i) are positive pairs => "i" label
        anchor2candidate_loss = self.cross_entropy_loss(scores, labels)

        scores = scores[:, :len(positives)]  # trowing away `hard_negatives` scores, if provided
        candidate2anchor_loss = self.cross_entropy_loss(scores.transpose(0, 1), labels)

        return self.accelerator.num_processes * ((anchor2candidate_loss + candidate2anchor_loss) / 2)

    def get_config_dict(self):
        return {'scale': self.scale, 'similarity_fct': self.similarity_fct.__name__}
