"""
Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Iterable, Dict, Union
from accelerate import Accelerator
from sentence_transformers import SentenceTransformer

from . import BaseContrastiveLoss


class HardNegativeSamplingLoss(BaseContrastiveLoss):
    """
        This loss is introduced in "Pairwise supervised contrastive learning of sentence representations" paper by Zhang et al.
        It's been also used in the "Learning Dialogue Representations from Consecutive Utterances" paper by Zhihan Zhou et al.
        which introduces Dialogue Sentence Embedding (DSE) method used by the winner team in the DSTC11 challenge (task1 and 2).

        This class expects as input a batch consisting of sentence pairs (a_1, p_1), (a_2, p_2)..., (a_n, p_n)
        and a list of binary labels (1 positive, 0 negative). Note that for each anchor sentence, all the other anchors' positive
        sentences in the batch will be considered as negative.

        That is, in addition to the provided labels, for each a_i, it uses all other p_j as negative samples, i.e.,
        for a_i, we have 1 positive example (p_i) and n-1 negative examples (p_j). This loss puts higher weights
        on the samples that are close to the anchor in the representation space. The underlying hypothesis is that
        hard negatives are more likely to occur among those that are located close to the anchor in the representation space.

        The performance is expected to increase when increasing batch sizes.

        Example::

            from spretrainer.losses import HardNegativeSamplingLoss
            from sentence_transformers import SentenceTransformer, InputExample
            from torch.utils.data import DataLoader

            model = SentenceTransformer('distilbert-base-uncased')
            train_examples = [InputExample(texts=['Anchor 1', 'Positive 1']),
                InputExample(texts=['Anchor 2', 'Positive 2'])]
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
            train_loss = losses.HardNegativeSamplingLoss(model=model)
    """
    def __init__(self, model: SentenceTransformer, temperature: float = 0.05, use_cosine_similarity: bool = True, accelerator: Accelerator = None, use_contrastive_head: bool = False):
        """
        :param model: SentenceTransformer model
        :param temperature: The temperature hyperparameter used to scale similarity (the lower the temp, the bigger the scale)
        :param use_cosine_similarity: whether or not to use cosine similarity or just dot product.
        :param accelerator: Optional Accelerator object to be used in `forward()` to gather batches across GPUs
        """
        super(HardNegativeSamplingLoss, self).__init__(model, use_contrastive_head)
        self.temperature = temperature
        self.accelerator = accelerator
        self.use_cosine_similarity = use_cosine_similarity

    def forward(self, sentence_features: Iterable[Union[Dict[str, Tensor], Tensor]], labels: Tensor):
        unique_labels = set(labels.unique().tolist())
        if unique_labels != {1} and unique_labels != {0, 1}:
            raise ValueError("`labels` is expected to contain only binary labels (either 0 or 1) with at least one positive (1),"
                             f" however {str(unique_labels)} were given.")

        reps = [self.model(sentence_feature)
                if isinstance(sentence_feature, (dict, Dict))
                else sentence_feature
                for sentence_feature in sentence_features]

        assert len(reps) == 2, f"`sentence_features` is expected to contain only pairs of sentences, {len(reps)} were given."

        if self.use_cosine_similarity:
            reps[0] = F.normalize(reps[0], dim=1)
            reps[1] = F.normalize(reps[1], dim=1)

        anchors, candidates, labels = self.gather_batches_across_processes(reps[0], reps[1], labels)
        batch_size = anchors.shape[0]

        embeddings = torch.cat([anchors, candidates], dim=0)
        labels = torch.cat([labels, labels], dim=0)
        # sim = dot product, thus embeddings must be normalized if we want sim = cosine similarity
        sim_matrix = torch.mm(embeddings, embeddings.t().contiguous())

        # [epx(sim(e_0,e_0) / t), epx(sim(e_1,e_1) / t), ..., epx(sim(e_{bs-1},e_{bs-1}) / t)]
        pos = torch.exp((anchors * candidates).sum(1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)
        pos_mask = (labels == 1).detach().type(torch.int)

        neg_mask = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.get_device()).repeat(2, 2)
        neg = torch.exp(sim_matrix / self.temperature).masked_select(neg_mask).view(2 * batch_size, -1)
        neg_imp = neg.log().exp()  # alphas (relative importance of every j for every anchor i)
        Ng = (neg_imp * neg).sum(dim=-1) / neg_imp.mean(dim=-1)

        return self.accelerator.num_processes * ((-pos_mask * torch.log(pos / (Ng + pos))).sum() / pos_mask.sum())

    def get_config_dict(self):
        return {'temperature': self.temperature, "use_cosine_similarity": self.use_cosine_similarity}
