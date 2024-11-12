"""
Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import random

from torch.utils.data import Dataset
from sentence_transformers import InputExample

from collections.abc import Iterable


class SimilarityDatasetContrastive(Dataset):
    num_labels = 0

    def __init__(self, data: Iterable, label_pos: str = None, label_neg: str = None):

        only_positives = not label_pos or not label_neg  # if there's no labels, then we assume only possitive pairs are given
        valid_labels = [label_pos, label_neg]

        self.samples = []
        anchor2pos_neg_list = {}
        for (sent0, sent1, label) in data:
            sent0, sent1 = sent0.strip(), sent1.strip()

            if not only_positives and label is None:
                only_positives = True

            if label is None:
                self.samples.append(InputExample(texts=[sent0, sent1], label=1))
                # self.samples.append(InputExample(texts=[sent1, sent0], label=1))
            elif label not in valid_labels and only_positives:
                self.samples.append(InputExample(texts=[sent0, sent1], label=int(label)))
            elif label in valid_labels:

                if sent0 not in anchor2pos_neg_list:
                    anchor2pos_neg_list[sent0] = [set(), set()]  # neg, pos
                if sent1 not in anchor2pos_neg_list:
                    anchor2pos_neg_list[sent1] = [set(), set()]  # neg, pos

                label2ix = int(label == label_pos)

                # Similarity is symmetrical with respect to the negative (negative for both)
                anchor2pos_neg_list[sent0][label2ix].add(sent1)
                anchor2pos_neg_list[sent1][label2ix].add(sent0)

        if not only_positives:
            for anchor_sent, pos_neg_lists in anchor2pos_neg_list.items():
                if len(pos_neg_lists[0]) > 0 and len(pos_neg_lists[1]) > 0:
                    neg_list, pos_list = list(pos_neg_lists[0]), list(pos_neg_lists[1])
                    # TODO: Add more samples per anchor?
                    # adding triples (anchor_sent, positive, negative)
                    self.samples.append(InputExample(texts=[anchor_sent, random.choice(pos_list), random.choice(neg_list)]))
                    self.samples.append(InputExample(texts=[random.choice(pos_list), anchor_sent, random.choice(neg_list)]))
            # self._org_samples = [(anchor, pos_neg_lists) for anchor, pos_neg_lists in anchor2pos_neg_list.items()
            #                      if len(pos_neg_lists[0]) > 0 and len(pos_neg_lists[1]) > 0]
            del anchor2pos_neg_list

    def __len__(self):
        return len(self.samples)
        # return len(self._org_samples) * 2 # similarity simetry -> (sent_i, sent_i_positive, sent_i_negative), (sent_i_positive, sent_i, sent_i_negative)

    def __setitem__(self, idx, value):
        self.samples[idx] = value

    def __getitem__(self, idx):
        return self.samples[idx]
        # Tried lazy version on commented code blocks, but not so simple because __setitem__ is required...
        # o_idx = idx // 2
        # anchor_sent, pos_neg_lists = self._org_samples[o_idx]

        # if isinstance(pos_neg_lists[0], set):
        #     pos_neg_lists[0] = list(pos_neg_lists[0])  # pos list
        #     pos_neg_lists[1] = list(pos_neg_lists[1])  # neg list

        # return InputExample(texts=[anchor_sent if idx % 2 == 0 else random.choice(pos_neg_lists[1]),
        #                            anchor_sent if idx % 2 == 1 else random.choice(pos_neg_lists[1]),
        #                            random.choice(pos_neg_lists[0])])
