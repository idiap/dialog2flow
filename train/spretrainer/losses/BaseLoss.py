"""
Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import logging

from torch import nn, Tensor
from typing import Iterable, Dict


logger = logging.getLogger(__name__)


class BaseLoss(nn.Module):
    """
    Base class for all loss classes

    Extend this class and implement forward for custom losses.
    """
    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        pass
