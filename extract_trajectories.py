# -*- coding: utf-8 -*-
"""
Given a path to a collection of dialogues, this script first cluster all the utterances in the collection
and then convert each dialogue to a sequence of a "discrete trajectory" by replacing each utterances
with its corresponding cluster id.

Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import os
import re
import csv
import json
import torch
import logging
import argparse
import numpy as np

from tqdm import tqdm
from matplotlib import pyplot as plt
from simpleneighbors import SimpleNeighbors
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from scipy.cluster.hierarchy import dendrogram, to_tree

from util import SentenceTransformerOpenAI, SentenceTransformerDialoGPT, SentenceTransformerSbdBERT, \
                 slugify, get_turn_text, init_gpt, get_cluster_label

DEFAULT_OPENAI_MODEL = "text-embedding-3-large"
DEFAULT_SYS_NAME = "system"
DEFAULT_USER_NAME = "user"
DEFAULT_USER_ALIASES = ["user", "customer", "client"]
DEFAULT_TOKEN_START = "[start]"
DEFAULT_TOKEN_END = "[end]"

# e.g python extract_trajectories.py -i data/example/ -o output/ -m "sergioburdisso/dialog2flow-joint-bert-base" -t .6 -sp
parser = argparse.ArgumentParser(prog="Convert a collection of dialogues to discrete trajectories by clustering their utterance embeddings")
parser.add_argument("-i", "--input-path", help="Path to the input dialogues. A folder with txt, tsv or json files", required=True)
parser.add_argument("-m", "--model", help="Sentence-Bert model used to generate the embeddings", default="sergioburdisso/dialog2flow-joint-bert-base")
parser.add_argument("-t", "--threshold", type=float, help="Distance threshold or the number of cluster for the Agglomerative Clustering algorithm")
parser.add_argument("-o", "--output-path", help="Folder to store the inferred trajectories.json file", default="output/")
parser.add_argument("-sp", "--show-plot", action="store_true", help="Whether to show and save the Dendrogram with the hierarchy of clusters")
parser.add_argument("-xes", "--save-xes", action="store_true", help="Whether to also save dialogues as 'action' logs (traces) XES files for process mining (e.g. using `pm4py` package)")
parser.add_argument("-l", "--generate-labels", action="store_true", help="Generate action labels for discovered clusters with ChatGPT")
parser.add_argument("-tk", "--top-k", type=int, default=5, help="Top-K utteraces to be used to generate the labels with ChatGPT")
parser.add_argument("-d", "--target-domains", nargs='*', help="Target domains to use. If empty, all domains", required=False)
parser.add_argument("-s", "--seed", help="Seed for pseudo-random number generator", default=13)

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(levelname)s: %(message)s')

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def get_txt_dialog(path_file):
    dialog = []
    warn = False
    with open(path_file) as reader:
        for line in [ln for ln in reader.read().split("\n") if ln]:
            m = re.match(r"^(\w+?):\s*(.+)", line)
            if m:
                speaker = m.group(1)
                text = m.group(2)
            else:
                if not warn:
                    logger.warning(f"Invalid format in file `{path_file}`. Expected SPEAKER:UTTERANCE in each line of file: using default speaker ('{DEFAULT_USER_NAME}').")
                    warn = True
                speaker = DEFAULT_USER_NAME
                text = line
            dialog.append({
                "tag": DEFAULT_USER_NAME if speaker.lower() in DEFAULT_USER_ALIASES else DEFAULT_SYS_NAME,
                "text": text.strip(),
                "turn": None})
    return dialog


def get_tsv_dialog(path_file):
    with open(path_file, newline='') as reader:
        csvfile = csv.reader(reader, delimiter='\t')
        n_col = len(next(csvfile))
        assert n_col == 2, f"Invalid TSV file. Expected 2 columns (SPEAKER, UTTERANCE) found {n_col}."
        reader.seek(0)
        return [{"tag": DEFAULT_USER_NAME if row[0].lower() in DEFAULT_USER_ALIASES else DEFAULT_SYS_NAME,
                "text": row[1],
                "turn": None}
                for row in csvfile]


def get_json_dialog(path_file):
    with open(path_file) as reader:
        dialogue = json.load(reader)
        assert "Transcript" in dialogue and (not dialogue["Transcript"] or "ParticipantRole" in dialogue["Transcript"][0]), \
            "Invalid JSON format. JSON file is expected to be an Amazon Transcribe's post-call analytics output file " \
            "(https://docs.aws.amazon.com/transcribe/latest/dg/tca-post-call.html#tca-output-post-call)."
        dialogue = dialogue["Transcript"]
    return [{"tag": DEFAULT_USER_NAME if turn["ParticipantRole"].lower() in DEFAULT_USER_ALIASES else DEFAULT_SYS_NAME,
             "text": turn["Content"],
             "turn": None}
            for turn in dialogue]


def plot_dendrogram(model, title, path, labels=None, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    root, nodes = to_tree(linkage_matrix, rd=True)


    def get_leaves_of(node):
        # if it is a leaf node
        if node.count == 1 and node.dist == 0:
            return set([model.labels_[node.id]])
        return get_leaves_of(node.left).union(get_leaves_of(node.right))


    def get_children_leaf_ids(cluster_id):
        node = [node for node in nodes if node.id == cluster_id][0]
        return get_leaves_of(node)


    labeled = []
    def leaf2label(id):
        # if id < n_samples:
        if labels and model.labels_[id] not in labeled:
            labeled.append(model.labels_[id])
            return labels[model.labels_[id]]
        return str(model.labels_[id])


    def link_color_func(id):
        leaves_cluster_ids = get_children_leaf_ids(id)
        if len(leaves_cluster_ids) > 1:
            return "black"
        cluster_id = list(leaves_cluster_ids)[0]
        return f"C{cluster_id}"

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix,
               leaf_label_func=leaf2label,
               link_color_func=link_color_func,
               no_labels=True,
               leaf_rotation=-90,
               **kwargs)

    ax = plt.gca()
    ax.set_ylim([0, .8])
    plt.ylabel('cosine distance', fontsize=12)
    plt.title(title)
    plt.savefig(path, dpi=300)
    plt.show()


if __name__ == "__main__":
    logger.info("Reading conversations...")
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"The provided input path is not a valid path: '{args.input_path}'")

    dialogues = {}
    if os.path.isfile(args.input_path):
        assert args.input_path.endswith(".json"), "input path should be either a single JSON file or a folder containing one file per conversation"
        with open(args.input_path) as reader:
            dialogues = json.load(reader)
    elif os.path.isdir(args.input_path):
        domain = os.path.basename(os.path.normpath(args.input_path))
        for filename in tqdm(os.listdir(args.input_path), desc="Dialogues:"):
            dialog_id, ext = os.path.splitext(filename)
            if ext == ".json":
                dialogue = get_json_dialog(os.path.join(args.input_path, filename))
            elif ext == ".tsv":
                dialogue = get_tsv_dialog(os.path.join(args.input_path, filename))
            elif ext == ".txt":
                dialogue = get_txt_dialog(os.path.join(args.input_path, filename))
            else:
                logger.warning(f"File extension '{ext}' not supported: skipping file '{filename}'")
                continue

            dialogues[dialog_id] = {
                "goal": { domain: {}},
                "log": [
                    {
                        "tag": None,
                        "text": None,
                        "turn": DEFAULT_TOKEN_START
                    },
                    {
                        "tag": None,
                        "text": None,
                        "turn": DEFAULT_TOKEN_END
                    }
                ]
            }
            dialogues[dialog_id]["log"] = dialogues[dialog_id]["log"][:1] + dialogue + dialogues[dialog_id]["log"][-1:]
    else:
        logger.error("Input path should be either a single JSON file or a folder containing one file per conversation")
        exit()

    model_name = slugify(os.path.basename(args.model))
    output_path_trajectories = os.path.join(args.output_path, f"trajectories-{model_name}.json")
    output_path_clusters_folder = os.path.join(os.path.join(args.output_path, "clusters", model_name))

    domains = {}
    if os.path.exists(output_path_trajectories):
        with open(output_path_trajectories) as reader:
            new_dialogs = json.load(reader)
    else:
        new_dialogs = {}

    unique_domains = set()
    for dialog_id, dialogue in dialogues.items():
        domain = next(iter(dialogue["goal"]))
        unique_domains.add(domain)

        if args.target_domains and domain not in args.target_domains:
            continue

        new_dialogs[dialog_id] = dialogue

        if domain not in domains:
            domains[domain] = {"log": [], "speaker": [], "text": [],
                               "emb": None, "prediction": None}
        domains[domain]["speaker"].extend(turn["tag"].lower() for turn in dialogue["log"][1:-1])
        domains[domain]["text"].extend(get_turn_text(turn) for turn in dialogue["log"][1:-1])
        domains[domain]["log"].extend(dialogue["log"][1:-1])

    multi_domain = len(unique_domains) > 1

    logger.info(f"Using model '{args.model}' model to generate the embeddings.")
    pb_domain = tqdm(domains, desc="Domains") if multi_domain else domains
    for domain in pb_domain:
        if multi_domain:
            logger.info(f"Domain: {domain.upper()}")

        domains[domain]["speaker"] = np.array(domains[domain]["speaker"])
        domains[domain]["text"] = np.array(domains[domain]["text"])
        domains[domain]["prediction"] = np.zeros_like(domains[domain]["text"], dtype=int)
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
        # GloVe can return some Zero vectors, which invalidate the use of cosine distance, seting
        # one coordinate to 1 as a quick work around to prevent division by zero error:
        domains[domain]["emb"][np.where(~np.any(domains[domain]["emb"], axis=1))[0], 0] = 1

        normalized_turn_names = {DEFAULT_USER_NAME: {}, DEFAULT_SYS_NAME: {}}
        for speaker in normalized_turn_names:
            logger.info(f"Clustering {speaker.upper()} utterances...")
            speaker_mask = domains[domain]["speaker"] == speaker
            linkage = "average"
            n_clusters = None
            n_unique_labels = None
            distance_threshold = None

            if not speaker_mask.any():
                logger.warning(f"No {speaker} utterances were found.")
                continue

            if args.threshold is None or args.threshold < 0:
                logger.info("No valid threshold or number of cluster was provided. "
                            "Trying to set the number of clusters using ground truth annotation (if available)")
                unique_labels = np.unique(domains[domain]["labels"][speaker_mask]).tolist()
                assert unique_labels != ["unknown"], "No ground truth annotation found (and `--threshold` was not provided or is invalid)."

                n_unique_labels = len(unique_labels)
                n_clusters = n_unique_labels
            elif args.threshold > 1 and args.threshold == int(args.threshold):
                n_clusters = int(args.threshold)
            else:
                distance_threshold = args.threshold

            clustering = AgglomerativeClustering(n_clusters=n_clusters,
                                                 linkage=linkage,
                                                 metric="cosine",
                                                 compute_distances=True,
                                                 distance_threshold=distance_threshold).fit(domains[domain]["emb"][speaker_mask])
            predictions = clustering.labels_

            # Getting utterance closer to the centroid
            cluster_ids = np.unique(predictions)
            cluster_topk_utts = [None] * cluster_ids.shape[0]
            centroids = np.zeros((cluster_ids.shape[0], domains[domain]["emb"][0].shape[0]))
            for ix, cluster_id in enumerate(cluster_ids):
                cluster_utts = domains[domain]["text"][speaker_mask][predictions == cluster_id]
                cluster_embs = domains[domain]["emb"][speaker_mask][predictions == cluster_id]

                index = SimpleNeighbors(domains[domain]["emb"].shape[1], metric="cosine")
                index.feed([(utt, cluster_embs[cix]) for cix, utt in enumerate(cluster_utts)])
                index.build()

                centroids[ix] = cluster_embs.mean(axis=0)
                top_k = args.top_k
                while cluster_topk_utts[ix] is None and top_k > 0:
                    try:
                        cluster_topk_utts[ix] = {"name": None, "utterances": index.nearest(centroids[ix], top_k)}
                    except ValueError:  # "Expected n_neighbors <= n_samples_fit"
                        top_k -= 1

            # Saving cluster information for later use (centroid embeddings and top-k utterances of each cluster)
            if args.generate_labels:
                init_gpt()
                for cluster in tqdm(cluster_topk_utts, desc=f"Cluster labels ({speaker.title()}):"):
                    cluster["name"] = get_cluster_label(cluster["utterances"])
            output_path_clusters = os.path.join(output_path_clusters_folder, domain) if multi_domain else output_path_clusters_folder
            os.makedirs(output_path_clusters, exist_ok=True)
            with open(os.path.join(output_path_clusters, f"centroid-embeddings.{speaker.lower()}.npy"), "wb") as writer:
                np.save(writer, centroids)
            with open(os.path.join(output_path_clusters, f"top-utterances.{speaker.lower()}.json"), "w") as writer:
                json.dump(cluster_topk_utts, writer)

            logger.info(f"# clusters: {len(np.unique(predictions))}")
            logger.info(f"# ground truth labels: {n_unique_labels}")
            logger.info(f"# Total predictions: {len(predictions)}")
            domains[domain]["prediction"][speaker_mask] = predictions
            for tid in np.unique(predictions):
                cluster_name = cluster_topk_utts[tid]["utterances"][0] if cluster_topk_utts[tid]["name"] is None else cluster_topk_utts[tid]["name"]
                normalized_turn_names[speaker][tid] = {"name": f"{tid}_" + cluster_name,
                                                       "info": cluster_topk_utts[tid],
                                                       "id": f"{speaker[0].lower()}{tid}"}

            if args.show_plot:
                plots_path = os.path.join(args.output_path, "plots")
                if multi_domain:
                    plots_path = os.path.join(plots_path, domain)
                os.makedirs(plots_path, exist_ok=True)
                output_file = os.path.join(plots_path, f"dendrogram_{model_name}.{speaker.lower()}.png")
                plot_dendrogram(clustering,
                                f"{speaker.title()} Utterances ({model_name})",
                                output_file)
                logger.info(f"Dendrogram plot for {speaker} utterances saved in `{output_file}`")

        if not domains[domain]['prediction'].any():
            logger.warning(f"No cluster predictions for '{domain}'. Skipped.")
            continue

        for ix, turn in enumerate(domains[domain]["log"]):
            turn["turn"] = normalized_turn_names[turn['tag']][domains[domain]['prediction'][ix]]

        # Saving dialogues as state sequences for graph visualization (as url hash #)
        state_sequences = {did: f'#{",".join([t["turn"]["id"] for t in d["log"][1:-1]])}'
                        for did, d in new_dialogs.items() if domain in d["goal"]}
        with open(os.path.join(output_path_clusters, f"cluster-id-sequences.json"), "w") as writer:
            json.dump(state_sequences, writer)

        if args.save_xes:
            # Saving dialogues as action, event or stages logs (traces) (XES files)
            xes_path = os.path.join(args.output_path, "xes")
            if multi_domain:
                xes_path = os.path.join(xes_path, domain)
            actions_logs = [[t["turn"]["name"] for t in d["log"][1:-1]]
                            for d in new_dialogs.values() if domain in d["goal"]]
            os.makedirs(xes_path, exist_ok=True)
            xes_path = os.path.join(xes_path, f"action_log_{model_name}.xes")
            logger.info(f"Saving event log XES file in `{xes_path}`")
            logger.warn(f"XES file convertion not implemented yet")

        for ix, turn in enumerate(domains[domain]["log"]):
            turn["turn"] = f"{turn['tag'].upper()}: {normalized_turn_names[turn['tag']][domains[domain]['prediction'][ix]]['name']}"

    os.makedirs(args.output_path, exist_ok=True)
    with open(output_path_trajectories, "w") as writer:
        json.dump(new_dialogs, writer)
