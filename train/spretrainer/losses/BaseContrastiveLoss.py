"""
Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import torch
import logging
import torch.nn as nn

from torch import Tensor
from typing import Iterable, Dict, Union
from sentence_transformers import SentenceTransformer

from . import BaseLoss
from spretrainer.utils import distributed


logger = logging.getLogger(__name__)


class BaseContrastiveLoss(BaseLoss):
    """
    Base class for all contrastive loss classes

    Extend this class and implement forward for custom contrastive losses.
    """
    accelerator = None
    even_batches = True

    def __init__(self, model: SentenceTransformer, use_contrastive_head: bool = False):
        super(BaseLoss, self).__init__()

        self.encoder = model
        if use_contrastive_head:
            emb_size = self.encoder[0].auto_model.config.hidden_size
            feat_dim = 128  # TODO: pass as argument! (default same as DSE original paper)
            self.contrastive_head = nn.Sequential(  # same as DSE original paper
                nn.Linear(emb_size, emb_size, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(emb_size, feat_dim, bias=False))
        else:
            self.contrastive_head = None

    def model(self, sentence_features: Iterable[Union[Dict[str, Tensor], Tensor]]):
        embs = self.encoder(sentence_features)['sentence_embedding']
        return self.contrastive_head(embs) if self.contrastive_head else embs

    def gather_batches_across_processes(self, anchors, candidates, labels=None):
        """Gather anchors, candidates and labels accross multiple GPU batches into a single one."""

        if self.accelerator is not None and distributed.is_distributed():
            my_batch_size = anchors.shape[0]
            my_rank = self.accelerator.process_index

            if not self.even_batches:
                batch_sizes = [0] * self.accelerator.num_processes
                for rank in range(self.accelerator.num_processes):
                    batch_sizes[rank] = distributed.broadcast_value(my_batch_size, src=rank)

                pad_anchors = self.accelerator.pad_across_processes(anchors)
                pad_candidates = self.accelerator.pad_across_processes(candidates)
                if labels is not None:
                    pad_labels = self.accelerator.pad_across_processes(labels)
            else:
                pad_anchors = anchors
                pad_candidates = candidates
                pad_labels = labels

            with torch.no_grad():
                all_anchors = self.accelerator.gather(pad_anchors)
                all_candidates = self.accelerator.gather(pad_candidates)
                all_labels = None if labels is None else self.accelerator.gather(pad_labels)

                del pad_anchors, pad_candidates

            if self.even_batches:
                my_start, my_end = my_rank * my_batch_size, (my_rank + 1) * my_batch_size
            else:
                all_batch_size = max(batch_sizes)
                all_anchors = torch.concat([all_anchors[rank * all_batch_size: rank * all_batch_size + batch_sizes[rank]]
                                            for rank in range(self.accelerator.num_processes)])
                all_candidates = torch.concat([all_candidates[rank * all_batch_size: rank * all_batch_size + batch_sizes[rank]]
                                               for rank in range(self.accelerator.num_processes)])
                if labels is not None:
                    all_labels = torch.concat([all_labels[rank * all_batch_size: rank * all_batch_size + batch_sizes[rank]]
                                               for rank in range(self.accelerator.num_processes)])
                my_start, my_end = sum(batch_sizes[:my_rank]), sum(batch_sizes[:my_rank]) + my_batch_size

            # anchors and candidates have grads for brackprop
            all_anchors[my_start: my_end] = anchors
            all_candidates[my_start: my_end] = candidates

            return all_anchors, all_candidates, all_labels
        else:
            return anchors, candidates, labels

    def gather_batches_across_processes_single(self, tensors):
        """Gather provided `tensors` accross multiple GPU batches into a single one."""

        if self.accelerator is not None and distributed.is_distributed():
            my_batch_size = tensors.shape[0]
            my_rank = self.accelerator.process_index

            if not self.even_batches:
                batch_sizes = [0] * self.accelerator.num_processes
                for rank in range(self.accelerator.num_processes):
                    batch_sizes[rank] = distributed.broadcast_value(my_batch_size, src=rank)
                pad_tensors = self.accelerator.pad_across_processes(tensors)
            else:
                pad_tensors = tensors

            with torch.no_grad():
                all_tensors = self.accelerator.gather(pad_tensors)

            if self.even_batches:
                my_start, my_end = my_rank * my_batch_size, (my_rank + 1) * my_batch_size
            else:
                all_batch_size = max(batch_sizes)
                all_tensors = torch.concat([all_tensors[rank * all_batch_size: rank * all_batch_size + batch_sizes[rank]]
                                            for rank in range(self.accelerator.num_processes)])
                my_start, my_end = sum(batch_sizes[:my_rank]), sum(batch_sizes[:my_rank]) + my_batch_size

            # original `tensors` have grads for brackprop
            all_tensors[my_start: my_end] = tensors

            return all_tensors
        else:
            return tensors

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        # reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        # anchors, candidates, labels = self.gather_batches_across_processes(reps[0], reps[1], labels)
        pass
