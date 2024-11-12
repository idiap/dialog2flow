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
# (https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesRankingLoss.py)
class MultipleNegativesRankingLoss(BaseContrastiveLoss):
    """
        This loss expects as input a batch consisting of sentence pairs (a_1, p_1), (a_2, p_2)..., (a_n, p_n)
        where we assume that (a_i, p_i) are a positive pair and (a_i, p_j) for i!=j a negative pair.

        For each a_i, it uses all other p_j as negative samples, i.e., for a_i, we have 1 positive example (p_i) and
        n-1 negative examples (p_j). It then minimizes the negative log-likehood for softmax normalized scores.

        This loss function works great to train embeddings for retrieval setups where you have positive pairs (e.g. (query, relevant_doc))
        as it will sample in each batch n-1 negative docs randomly.

        The performance usually increases with increasing batch sizes.

        For more information, see: https://arxiv.org/pdf/1705.00652.pdf
        (Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4)

        You can also provide one or multiple hard negatives per anchor-positive pair by structering the data like this:
        (a_1, p_1, n_1), (a_2, p_2, n_2)

        Here, n_1 is a hard negative for (a_1, p_1). The loss will use for the pair (a_i, p_i) all p_j (j!=i) and all n_j as negatives.

        Example::

            from sentence_transformers import SentenceTransformer, losses, InputExample
            from torch.utils.data import DataLoader

            model = SentenceTransformer('distilbert-base-uncased')
            train_examples = [InputExample(texts=['Anchor 1', 'Positive 1']),
                InputExample(texts=['Anchor 2', 'Positive 2'])]
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
            train_loss = losses.MultipleNegativesRankingLoss(model=model)
    """
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct: callable = util.cos_sim, accelerator: Accelerator = None, use_contrastive_head: bool = False):
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: Similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        :param accelerator: Optional Accelerator object to be used in `forward()` to gather batches across GPUs
        """
        super(MultipleNegativesRankingLoss, self).__init__(model, use_contrastive_head)
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

        labels = torch.arange(len(anchors), dtype=torch.long, device=scores.device)  # (a_i, p_i) are positive pairs => "i" label

        return self.accelerator.num_processes * self.cross_entropy_loss(scores, labels)

    def get_config_dict(self):
        return {'scale': self.scale, 'similarity_fct': self.similarity_fct.__name__}
