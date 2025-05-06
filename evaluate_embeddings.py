# -*- coding: utf-8 -*-
"""
Given utterances with ground truth action annotation, a sentence embedding model, compute the
evaluation described in the paper and store the result in the provided folder. More precisely,
the evaluation is performed in a similarity-based classification, ranking-based and anisotropy-based
settings, as reported in the paper (Table 2, 3, and 4).

Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import os
import json
import torch
import argparse
import numpy as np

from tqdm.auto import tqdm
from collections import Counter
from sentence_transformers import SentenceTransformer, models
from sklearn.metrics import classification_report, ndcg_score
from sklearn.preprocessing import normalize

from util import SentenceTransformerOpenAI, SentenceTransformerDialoGPT, SentenceTransformerSbdBERT, \
                 get_turn_text, compute_anisotropy


DEFAULT_OPENAI_MODEL = "text-embedding-3-large"
DEFAULT_SYS_NAME = "system"
DEFAULT_USER_NAME = "user"
DEFAULT_TOKEN_START = "[start]"
DEFAULT_TOKEN_END = "[end]"
DEFAULT_TOP_K_RANKINGS = [10]

# e.g. python evaluate_embeddings.py -i "data/spokenwoz/trajectories.single_domain.json" -m "sergioburdisso/dialog2flow-joint-bert-base" -o "output/results/spokenwoz"
parser = argparse.ArgumentParser(prog="Evaluate the performance of the provided model in few-show classification and ranking-based settings")
parser.add_argument("-i", "--input-path", help="Path to the ground truth 'trajectories.json' file", default="data/spokenwoz/trajectories.single_domain.json")
parser.add_argument("-m", "--model", help="Sentence-Bert model used for turn embeddings", default="sergioburdisso/dialog2flow-joint-bert-base")
parser.add_argument("-o", "--output-folder", help="Folder to store evaluation results in JSON files", default="output/results/")
parser.add_argument("-d", "--target-domains", nargs='*', help="Target domains to use. If empty, all domains")
parser.add_argument("-n", "--n-shots", nargs='*', type=int, help="n nots to use", default=[1, 5])
parser.add_argument("-k", "--num-runs", type=int, help="Number of times to perform the evaluation", default=10)
parser.add_argument("-s", "--seed", type=int, help="Seed for pseudo-random number generator", default=13)
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def evaluate_fewshot(labels, embs, n_shots):
    # Creating train and eval set dynamically...
    unique_labels, counts = np.unique(labels, return_counts=True)
    counts_mask = counts >= max(n_shots * 2, n_shots + 5)
    unique_labels = unique_labels[counts_mask]
    counts = counts[counts_mask]
    num_labels = unique_labels.shape[0]

    if num_labels <= 1:
        raise ValueError(f"Decrease the n-shots value since there is not enough instances for training and evaluation")

    valid_mask = np.isin(labels, unique_labels)
    labels = labels[valid_mask]
    embs = normalize(embs[valid_mask])

    label_indexes = {label:np.where(labels == label)[0] for label in unique_labels}
    label_train_sample = {label:np.random.permutation(label_indexes[label])[:n_shots]
                          for label in unique_labels}

    train_sample = np.concatenate(list(label_train_sample.values()))
    eval_mask = np.ones_like(labels, dtype=bool)
    eval_mask[train_sample] = False
    x_train, y_train = embs[train_sample], labels[train_sample]
    x_eval, y_eval = embs[eval_mask], labels[eval_mask]

    # Computing the prototype embeddings for each class/label
    prototype_embeddings = np.zeros([num_labels, embs.shape[1]])
    for label_ix, label in enumerate(unique_labels):
        prototype_embeddings[label_ix] = x_train[np.where(y_train == label)[0]].mean(axis=0)

    # Classifying evaluation samples by distance to prototype
    sim_matrix = x_eval @ prototype_embeddings.T
    y_eval_pred = sim_matrix.argmax(axis=1)
    y_eval_pred = [unique_labels[ix] for ix in y_eval_pred]

    # Computing evaluation metrics
    report = classification_report(y_eval, y_eval_pred, output_dict=True, zero_division=0)
    f1_score = report["macro avg"]["f1-score"]
    accuracy = report["accuracy"]

    if n_shots == 1:
        unique_labels = unique_labels.tolist()
        y_eval = np.vectorize(unique_labels.index)(y_eval)

        sim_matrix = prototype_embeddings @ x_eval.T
        label_rankings = y_eval[sim_matrix.argsort(axis=1)][:,::-1]

        precision_k = np.zeros(len(DEFAULT_TOP_K_RANKINGS))
        ndcg_k = np.zeros_like(precision_k)
        y_pred_rankings = (label_rankings == np.arange(num_labels).reshape((num_labels, -1))).astype(int)
        y_true_rankings = np.zeros_like(y_pred_rankings)
        for label_ix in range(num_labels):
            y_true_rankings[label_ix, :counts[label_ix]] = 1

        for ix, k in enumerate(DEFAULT_TOP_K_RANKINGS):
            precision_k[ix] = y_pred_rankings[:, :k].mean()

            if k > 1:
                ndcg_k[ix] = ndcg_score(y_true_rankings, y_pred_rankings, k=k)
            else:
                ndcg_k[ix] = precision_k[ix]

        return f1_score, accuracy, precision_k, ndcg_k

    return f1_score, accuracy


if __name__ == "__main__":
    print("Reading conversations...")
    with open(args.input_path) as reader:
        dialogues = json.load(reader)

    path_results_anisotropy = os.path.join(args.output_folder, "anisotropy_results.json")
    if os.path.exists(path_results_anisotropy):
        with open(path_results_anisotropy) as reader:
            anisotropy_results_all = json.load(reader)
    else:
        anisotropy_results_all = {}

    path_results_classification = os.path.join(args.output_folder, "classification_similarity_results.json")
    if os.path.exists(path_results_classification):
        with open(path_results_classification) as reader:
            classification_sim_results = json.load(reader)
    else:
        classification_sim_results = {}

    model_name = os.path.basename(args.model)
    anisotropy_results_all[model_name] = {}
    if model_name not in classification_sim_results:
        classification_sim_results[model_name] = {}
    domains = {}
    for dialog_id, dialogue in dialogues.items():
        domain = next(iter(dialogue["goal"]))

        if args.target_domains and domain not in args.target_domains:
            continue
        if domain not in domains:
            domains[domain] = {"log": [], "speaker": [], "text": [],
                               "emb": None, "prediction": None}
        domains[domain]["speaker"].extend(turn["tag"].lower() for turn in dialogue["log"][1:-1])
        domains[domain]["text"].extend(get_turn_text(turn) for turn in dialogue["log"][1:-1])
        domains[domain]["log"].extend(dialogue["log"][1:-1])

    global_labels = Counter()

    for domain in tqdm(domains, desc="Domains"):
        domains[domain]["speaker"] = np.array(domains[domain]["speaker"])
        domains[domain]["text"] = np.array(domains[domain]["text"])
        domains[domain]["labels"] = np.array([get_turn_text(t, use_ground_truth=True)
                                              for t in domains[domain]["log"]])

        if "todbert_sbd" in args.model.lower():
            sentence_encoder = SentenceTransformerSbdBERT.from_pretrained(args.model, args=args)
            sentence_encoder.to(device)
        elif "dialogpt" in args.model.lower():
            sentence_encoder = SentenceTransformerDialoGPT(args.model, device=device)
        elif args.model.lower() == "chatgpt" or "openai" in args.model.lower():
            if "openai" in args.model.lower() and "/" in args.model:  # e.g. openai/text-embedding-3-large
                model = os.path.basename(args.model)
            else:
                model = DEFAULT_OPENAI_MODEL
            sentence_encoder = SentenceTransformerOpenAI(model)
        else:
            sentence_encoder = SentenceTransformer(args.model, device=device)

        domains[domain]["emb"] = sentence_encoder.encode(domains[domain]["text"], show_progress_bar=True, batch_size=128, device=device)

        # Anisotropy computation
        labels, counts = np.unique(domains[domain]["labels"], return_counts=True)
        global_labels.update({lbl:counts[ix] for ix, lbl in enumerate(labels)})

        label_centroids = np.zeros((labels.shape[0], domains[domain]["emb"].shape[1]))
        intra_label_anisotropy = []
        for ix, label in enumerate(labels):
            label_embs = domains[domain]["emb"][domains[domain]["labels"] == label]
            label_centroids[ix] = label_embs.mean(axis=0)
            if label_embs.shape[0] > 2:
                # Compute intra-label anisotropy
                intra_label_anisotropy.append(compute_anisotropy(label_embs))

        intra_label_anisotropy = np.array(intra_label_anisotropy)

        # Compute inter-label anisotropy
        anisotropy_results = {
            "intra": {
                "mean": intra_label_anisotropy.mean(),
                "median": np.median(intra_label_anisotropy),
                "std": intra_label_anisotropy.std()
            },
            "inter": compute_anisotropy(label_centroids),
        }
        anisotropy_results_all[model_name][domain] = anisotropy_results

        print("\n> Results for Anisotropy-based evaluation:")
        print(f"  - Mean Intra-label anisotropy (↑): {anisotropy_results['intra']['mean']:.3f} ± {anisotropy_results['intra']['std']:.3f}")
        print(f"  - Median Intra-label anisotropy (↑): {anisotropy_results['intra']['median']:.3f} ± {anisotropy_results['intra']['std']:.3f}")
        print(f"  - Inter-label anisotropy (↓): {anisotropy_results['inter']:.3f}")

        if domain not in classification_sim_results[model_name]:
            classification_sim_results[model_name][domain] = {}

        precision_k = np.zeros((len(DEFAULT_TOP_K_RANKINGS), args.num_runs))
        ndcg_k = np.zeros_like(precision_k)
        for n_shot in args.n_shots:
            scores = np.zeros(args.num_runs)
            accuracies = np.zeros(args.num_runs)
            for ix in tqdm(range(args.num_runs), desc="Few-shot Classification", leave=False):
                if n_shot == 1:
                    scores[ix], accuracies[ix], precision_k[:, ix], ndcg_k[:, ix] = evaluate_fewshot(domains[domain]["labels"],
                                                                                                     domains[domain]["emb"],
                                                                                                     n_shots=n_shot)
                else:
                    scores[ix], accuracies[ix] = evaluate_fewshot(domains[domain]["labels"],
                                                                  domains[domain]["emb"],
                                                                  n_shots=n_shot)
            if n_shot == 1:
                classification_sim_results[model_name][domain][f"ranking"] = {"precision": {}, "ndcg": {}}
                for ix, k in enumerate(DEFAULT_TOP_K_RANKINGS):
                    print("\n> Results for Ranking-based evaluation:")
                    print(f"  - NDCG@{k}: {ndcg_k[ix].mean() * 100:.2f} ± {ndcg_k[ix].std() * 100:.2f}")
                    classification_sim_results[model_name][domain][f"ranking"]["precision"][k] = {
                        "values": precision_k[ix].tolist(),
                        "mean": precision_k[ix].mean(),
                        "std": precision_k[ix].std()
                    }
                    classification_sim_results[model_name][domain][f"ranking"]["ndcg"][k] = {
                        "values": ndcg_k[ix].tolist(),
                        "mean": ndcg_k[ix].mean(),
                        "std": ndcg_k[ix].std()
                    }

            print(f"\n> Results for {n_shot}-shot similarity-based classification")
            print(f"  - Average MA F1 score: {scores.mean() * 100:.2f} ± {scores.std() * 100:.2f}")
            print(f"  - Average Accuracy: {accuracies.mean() * 100:.2f} ± {accuracies.std() * 100:.2f}")
            classification_sim_results[model_name][domain][f"{n_shot}-shot"] = {
                "f1-scores": scores.tolist(),
                "f1-scores-mean": scores.mean(),
                "f1-scores-std": scores.std(),

                "accuracy": accuracies.tolist(),
                "accuracy-mean": accuracies.mean(),
                "accuracy-std": accuracies.std(),
            }

    print(f"\nDone. Saving obtained results in '{args.output_folder}'.")
    os.makedirs(args.output_folder, exist_ok=True)
    with open(path_results_anisotropy, "w") as writer:
        json.dump(anisotropy_results_all, writer)
    with open(path_results_classification, "w") as writer:
        json.dump(classification_sim_results, writer)
