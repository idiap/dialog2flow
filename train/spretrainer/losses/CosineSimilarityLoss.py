"""
Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from sentence_transformers import SentenceTransformer

from . import BaseLoss


# Original code taken from sentence_bert.losses.CosineSimilarityLoss
# (https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/CosineSimilarityLoss.py)
class CosineSimilarityLoss(BaseLoss):
    """
    CosineSimilarityLoss expects, that the InputExamples consists of two texts and a float label.

    It computes the vectors u = model(input_text[0]) and v = model(input_text[1]) and measures the cosine-similarity between the two.
    By default, it minimizes the following loss: ||input_label - cos_score_transformation(cosine_sim(u,v))||_2.

    :param model: SentenceTransformer model
    :param loss_fct: Which pytorch loss function should be used to compare the cosine_similartiy(u,v) with the input_label? By default, MSE:  ||input_label - cosine_sim(u,v)||_2
    :param cos_score_transformation: The cos_score_transformation function is applied on top of cosine_similarity. By default, the identify function is used (i.e. no change).

    Example::

            from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses

            model = SentenceTransformer('distilbert-base-nli-mean-tokens')
            train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
                InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]
            train_dataset = SentencesDataset(train_examples, model)
            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
            train_loss = losses.CosineSimilarityLoss(model=model)


    """
    def __init__(self, model: SentenceTransformer, loss_fct=None, cos_score_transformation=None):
        super(CosineSimilarityLoss, self).__init__()
        self.model = model
        self.loss_fct = loss_fct if loss_fct else nn.MSELoss()
        self.cos_score_transformation = cos_score_transformation if cos_score_transformation else nn.Identity()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        output = self.cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[1]))
        return self.loss_fct(output, labels.view(-1))
