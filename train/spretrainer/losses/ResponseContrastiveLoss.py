"""
Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import logging

from accelerate import Accelerator
from sentence_transformers import util
from sentence_transformers import SentenceTransformer

from . import MultipleNegativesRankingLoss

logger = logging.getLogger(__name__)


class ResponseContrastiveLoss(MultipleNegativesRankingLoss):
    """
    Response Contrastive Loss is introduced in the "TOD-BERT: Pre-trained Natural Language
    Understanding for Task-Oriented Dialogue" paper by Wu et al.
    It is essentially equivalent to Multiple Negatives Ranking Loss using dot product
    as similarity metric instead of dot product and no scaling.
    """
    def __init__(self, model: SentenceTransformer, accelerator: Accelerator = None, use_contrastive_head: bool = False):
        """
        :param model: SentenceTransformer model
        :param accelerator: Optional Accelerator object to be used in `forward()` to gather batches across GPUs
        """
        super(ResponseContrastiveLoss, self).__init__(model=model,
                                                      accelerator=accelerator,
                                                      scale=1,
                                                      similarity_fct=util.dot_score,
                                                      use_contrastive_head=use_contrastive_head)
