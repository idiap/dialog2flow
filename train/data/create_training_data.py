# -*- coding: utf-8 -*-
"""
Given the path to converted TOD datasets generated with `create_tod_datasets.py`,
create the CSV files used for training the Dialog2Flow sentence embedding models.

Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import os
import re
import csv
import json
import argparse
import pandas as pd

from tqdm import tqdm


OUTPUT_CSV_DA = "dialog-acts.csv"
OUTPUT_CSV_SLOTS = "slots.csv"
OUTPUT_CSV_ALL = "dialog-acts+slots.csv"
MIN_NUM_SAMPLES_CSV_ALL = 100
LABELS_SEP = ", "
CSV_COLUMNS = ["utterance", "label", "dataset"]

# e.g. create_training_data.py -i tod_datasets/
parser = argparse.ArgumentParser(prog="Convert unified TOD datasets generated with `create_tod_datasets.py` to csv files used for training.")
parser.add_argument("-i", "--path-datasets", help="Path to the folder storing all the converted datasets", default="tod_datasets/")
args = parser.parse_args()


def preprocess(text):
    text = re.sub(r"(\w)\s+'\s*", r"\1'", text)
    text = re.sub(r"(\w)\s+\?", r"\1'", text)
    text = " ".join(re.findall(r"[a-zA-Z0-9'?]+", text))
    return text.lower()


if __name__ == "__main__":
    fwriter_da = open(OUTPUT_CSV_DA, 'w', newline='')
    fwriter_slot = open(OUTPUT_CSV_SLOTS, 'w', newline='')
    fwriter_all = open(OUTPUT_CSV_ALL, 'w', newline='')

    csv_da = csv.DictWriter(fwriter_da, fieldnames=CSV_COLUMNS)
    csv_slot = csv.DictWriter(fwriter_slot, fieldnames=CSV_COLUMNS)
    csv_all = csv.DictWriter(fwriter_all, fieldnames=CSV_COLUMNS)

    csv_da.writeheader()
    csv_slot.writeheader()
    csv_all.writeheader()

    dataset_folders = [os.path.join(args.path_datasets, f)
                       for f in os.listdir(args.path_datasets)
                       if os.path.isdir(os.path.join(args.path_datasets, f))]
    for dataset_folder in tqdm(dataset_folders, desc="Datasets"):
        with open(os.path.join(dataset_folder, "data.json")) as reader:
            data = json.load(reader)

        for dialogue in tqdm(data["dialogs"].values(), desc="Dialogs", leave=False):
            for turn in dialogue:
                utterance = preprocess(turn["text"])
                main_acts, acts, slots = None, None, None
                if "dialog_acts" in turn["labels"]:
                    acts = turn["labels"]["dialog_acts"]["acts"]
                    for act in acts:
                        csv_da.writerow({
                            'utterance': utterance,
                            'label': act,
                            "dataset": os.path.basename(dataset_folder)
                        })
                    acts = LABELS_SEP.join(acts)
                if "slots" in turn["labels"]:
                    slots = LABELS_SEP.join(turn["labels"]["slots"])
                    for slot in turn["labels"]["slots"]:
                        csv_slot.writerow({
                            "utterance": utterance,
                            "label": slot,
                            "dataset": os.path.basename(dataset_folder)
                        })

                if acts or slots:
                    csv_all.writerow({
                        "utterance": utterance,
                        "label": ((f"{acts} " if acts else "") + (slots if slots else "")).strip(),
                        "dataset": os.path.basename(dataset_folder)
                    })

    fwriter_da.close()
    fwriter_slot.close()
    fwriter_all.close()

    df_all = pd.read_csv(OUTPUT_CSV_ALL)
    label_counts = df_all.label.value_counts()
    valid_labels = label_counts[label_counts > MIN_NUM_SAMPLES_CSV_ALL].index
    df_all[df_all.label.isin(valid_labels)].to_csv(OUTPUT_CSV_ALL, index=False)
