"""
Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import logging

from . import MultipleNegativesRankingLoss

logger = logging.getLogger(__name__)


class SimCseLoss(MultipleNegativesRankingLoss):
    """
    SimCSE method introduced in "SimCSE: Simple Contrastive Learning of Sentence Embeddings" by Gao et al.
    The loss is simply equivalent to MultipleNegativesRankingLoss applied to pairs of equal sentences
    (s_i, s_i). The trick is that, despite being the same sentence they'll have slightly different
    embeddings due to the dropout in the encoder. Then the model minimizes distances between encodings
    of the same sentence while maximazing it with other sentences in the same batch.
    """
    pass
