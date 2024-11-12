# -*- coding: utf-8 -*-
"""
Given the path to output results generated with ``evaluate_embeddings.py`` script, print them
as tables. More precisely, prints the table for the classification and anisotropy-based results,
as in Table 2 and 3, as well as for ranking-based results, as Table 4.

Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import os
import json
import argparse
import pandas as pd

from collections import defaultdict
from util import STR_MODEL_COLUMN, STR_AVERAGE_COLUMN, show_results

# e.g python show_embedding_results.py -i "output/results/spokenwoz"
# e.g python show_embedding_results.py -i "output/results/unified_evaluation"
parser = argparse.ArgumentParser(prog="Print obtained results with `evaluate_embeddings.py` as Tables.")
parser.add_argument("-i", "--results-path", help="Path to the folder containing the results from `evaluate_embeddings.py`")
args = parser.parse_args()

with open(os.path.join(args.results_path, "anisotropy_results.json")) as reader:
    results_anisotropy = json.load(reader)
with open(os.path.join(args.results_path, "classification_similarity_results.json")) as reader:
    results_classification = json.load(reader)

models = [model for model in results_classification.keys()]
domains = list(results_classification[next(iter(results_classification))].keys())
domains.append(STR_AVERAGE_COLUMN)

averages = defaultdict(lambda: {})
if results_anisotropy:
    dfas = {
        "intra": show_results(models, domains,
                            lambda model, domain: results_anisotropy[model][domain]["intra"]["mean"],
                            metric_name="intra",
                            print_table=False),
        "inter": show_results(models, domains,
                            lambda model, domain: results_anisotropy[model][domain]["inter"],
                            metric_name="inter", metric_is_ascending=True,
                            print_table=False),
        "delta": show_results(models, domains,
                                lambda model, domain: results_anisotropy[model][domain]["intra"]["mean"] - results_anisotropy[model][domain]["inter"],
                                metric_name="delta",
                                print_table=False),
    }

    for row_ix in range(len(dfas["delta"])):
        model_name = dfas["delta"].loc[row_ix][STR_MODEL_COLUMN]
        for metric in dfas:
            row = dfas[metric].loc[row_ix]
            averages[model_name][metric] = row[f"{STR_AVERAGE_COLUMN}_{metric}"]


if results_classification:
    print("\n=============== CLASSIFICATION RESULTS ===============\n")
    for n_shot in [1, 5]:
        df_f1 = show_results(models, domains,
                              lambda model, domain: results_classification[model][domain][f"{n_shot}-shot"]["f1-scores-mean"],
                              percentage=True, print_table=False)
        df_f1_std = show_results(models, domains,
                                  lambda model, domain: results_classification[model][domain][f"{n_shot}-shot"]["f1-scores-std"],
                                  percentage=True, print_table=False)
        df_acc = show_results(models, domains,
                               lambda model, domain: results_classification[model][domain][f"{n_shot}-shot"]["accuracy-mean"],
                               percentage=True, print_table=False)
        df_acc_std = show_results(models, domains,
                               lambda model, domain: results_classification[model][domain][f"{n_shot}-shot"]["accuracy-std"],
                               percentage=True, print_table=False)

        for row_ix, row in df_f1.iterrows():
            averages[row[STR_MODEL_COLUMN]][n_shot] = {"f1-score": {"mean": row[f"{STR_AVERAGE_COLUMN}_"]}, "accuracy": {}}
        for row_ix, row in df_f1_std.iterrows():
            averages[row[STR_MODEL_COLUMN]][n_shot]["f1-score"]["std"] = row[f"{STR_AVERAGE_COLUMN}_"]
        for row_ix, row in df_acc.iterrows():
            averages[row[STR_MODEL_COLUMN]][n_shot]["accuracy"]["mean"] = row[f"{STR_AVERAGE_COLUMN}_"]
        for row_ix, row in df_acc_std.iterrows():
            averages[row[STR_MODEL_COLUMN]][n_shot]["accuracy"]["std"] = row[f"{STR_AVERAGE_COLUMN}_"]

    rows = []
    for model in results_classification:
        if model.endswith("-ft"):
            continue

        rows.append({"model": model,
                     "1-shot f1": "-", "5-shot f1": "-",
                     "1-shot accuracy": "-", "5-shot accuracy": "-",
                     "intra-anisotropy": "-", "inter-anisotropy": "-", "intra-inter delta": "-"})
        if 1 not in averages[model]:
            continue

        results_1shot = averages[model][1]
        results_5shot = averages[model][5]

        rows[-1]["1-shot f1"] = f"{results_1shot['f1-score']['mean']:.2%} ± {results_1shot['f1-score']['std']:.2%}".replace("%", "")
        rows[-1]["5-shot f1"] = f"{results_5shot['f1-score']['mean']:.2%} ± {results_5shot['f1-score']['std']:.2%}".replace("%", "")
        rows[-1]["1-shot accuracy"] = f"{results_1shot['accuracy']['mean']:.2%} ± {results_1shot['accuracy']['std']:.2%}".replace("%", "")
        rows[-1]["5-shot accuracy"] = f"{results_5shot['accuracy']['mean']:.2%} ± {results_5shot['accuracy']['std']:.2%}".replace("%", "")

        rows[-1]["intra-anisotropy"] = f"{averages[model]['intra']:.3f}"
        rows[-1]["inter-anisotropy"] = f"{averages[model]['inter']:.3f}"
        rows[-1]["intra-inter delta"] = f"{averages[model]['delta']:.3f}"

    df = pd.DataFrame.from_dict(rows)
    print(df.to_markdown(index=False))

    print("\n\n=============== RANKING RESULTS ===============\n")

    df_ndcg = show_results(models, domains,
                            lambda model, domain: results_classification[model][domain]["ranking"]["ndcg"][f"10"]["mean"] if "ranking" in results_classification[model][domain] else -1,
                            percentage=True, print_table=False)
    df_ndcg_std = show_results(models, domains,
                            lambda model, domain: results_classification[model][domain]["ranking"]["ndcg"][f"10"]["std"] if "ranking" in results_classification[model][domain] else -1,
                            percentage=True, print_table=False)

    for row_ix, row in df_ndcg.iterrows():
        averages[row[STR_MODEL_COLUMN]]["ndcg@10"] = {"mean": row[f"{STR_AVERAGE_COLUMN}_"] * 100,
                                                      "std": df_ndcg_std[f"{STR_AVERAGE_COLUMN}_"][row_ix] * 100}

    rows = []
    for model in results_classification:
        ft_result = averages[f"{model}-ft"]['ndcg@10'] if f"{model}-ft" in averages else 0
        rows.append({"model": model,
                     "NDCG@10": f"{averages[model]['ndcg@10']['mean']:.2f} ± {averages[model]['ndcg@10']['std']:.2f}"})

    df = pd.DataFrame.from_dict(rows)
    print(df.to_markdown(index=False))
