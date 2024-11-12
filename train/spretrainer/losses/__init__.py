"""
Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
from .BaseLoss import BaseLoss
from .BaseContrastiveLoss import BaseContrastiveLoss

# supervised
from .SoftmaxLoss import SoftmaxLoss
from .CosineSimilarityLoss import CosineSimilarityLoss

# contrastive
from .LabeledContrastiveLoss import LabeledContrastiveLoss
from .HardNegativeSamplingLoss import HardNegativeSamplingLoss
from .MultipleNegativesRankingLoss import MultipleNegativesRankingLoss
from .MultipleNegativesSymmetricRankingLoss import MultipleNegativesSymmetricRankingLoss
from .ResponseContrastiveLoss import ResponseContrastiveLoss

# unsupervised
from .DenoisingAutoEncoderLoss import DenoisingAutoEncoderLoss
from .SimCseLoss import SimCseLoss

CONTRASTIVE_LOSSES = (ResponseContrastiveLoss,
                      MultipleNegativesRankingLoss,
                      MultipleNegativesSymmetricRankingLoss,
                      HardNegativeSamplingLoss,
                      LabeledContrastiveLoss)

UNSUPERVISED_LOSSES = (DenoisingAutoEncoderLoss, SimCseLoss)

CONTRASTIVE_LOSS_NAMES = tuple(loss.__name__ for loss in CONTRASTIVE_LOSSES)
UNSUPERVISED_LOSS_NAMES = tuple(loss.__name__ for loss in UNSUPERVISED_LOSSES)
