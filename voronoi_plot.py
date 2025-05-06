# -*- coding: utf-8 -*-
"""
Given a ground truth trajectories json file, generates 3D Voronoi plots, as shown
in Figure 3 in the paper, with their utterance embeddings color coded by the ground
truth action labels.

Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import os
import json
import torch
import plotly
import argparse
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from tqdm.auto import tqdm

from umap import UMAP
from umap.plot import _get_embedding
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from scipy.spatial import SphericalVoronoi, geometric_slerp

from util import SentenceTransformerOpenAI, get_turn_text


DEFAULT_OPENAI_MODEL = "text-embedding-3-large"

# e.g. python voronoi_plot.py -i data/spokenwoz/trajectories.single_domain.json -m "sergioburdisso/dialog2flow-joint-bert-base" -o "output/plots/voronoi/d2f_joint" -d hospital
# e.g. python voronoi_plot.py -i data/spokenwoz/trajectories.single_domain.json -m "openai/text-embedding-3-large" -o "output/plots/voronoi/openai" -d hospital
parser = argparse.ArgumentParser(prog="Given a ground truth trajectories json file generates 3D Voronoi plots with their utterance embeddings.")
parser.add_argument("-i", "--input-path", help="Path to the dataset ground truth 'trajectories.json' file", required=True)
parser.add_argument("-m", "--model", help="Sentence-Bert model used for turn embeddings", default="sergioburdisso/dialog2flow-joint-bert-base")
parser.add_argument("-o", "--output-path", help="Folder to store the inferred trajectories.json file", default="output/plots/voronoi/")
parser.add_argument("-d", "--target-domains", nargs='*', help="Target domains to use. If empty, all domains")
parser.add_argument("-s", "--seed", help="Seed for pseudo-random number generator", default=13)
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def sphere(x, y, z, radius, resolution=100):
    """Return the coordinates for plotting a sphere centered at (x,y,z)"""
    u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*1j]
    xx = radius * np.cos(u)*np.sin(v) + x
    yy = radius * np.sin(u)*np.sin(v) + y
    zz = radius * np.cos(v) + z
    return (xx, yy, zz)


def plot_voronoi_3d(embs, targets, speaker, labels, path, model_name):
    os.makedirs(path, exist_ok=True)

    umap2d = UMAP(n_neighbors=15,
                    n_components=2,
                    min_dist=0.0,
                    metric='cosine',
                    output_metric='haversine',
                    low_memory=False,
                    random_state=args.seed,
                    n_jobs=1).fit(embs)
    points_2d = _get_embedding(umap2d)

    points = np.zeros((points_2d.shape[0], 3))
    points[:, 0] = np.sin(points_2d[:, 0]) * np.cos(points_2d[:, 1])
    points[:, 1] = np.sin(points_2d[:, 0]) * np.sin(points_2d[:, 1])
    points[:, 2] = np.cos(points_2d[:, 0])

    unique_targets = np.unique(targets)
    centroid_points = np.zeros([unique_targets.shape[0], 3])
    for ix, target in enumerate(unique_targets):
        target_points = points[targets == target]
        centroid_points[ix] = target_points.mean(axis=0)
    centroid_points = normalize(centroid_points)

    sv = SphericalVoronoi(centroid_points, 1, np.zeros(3))

    # sort vertices (optional, helpful for plotting)
    sv.sort_vertices_of_regions()
    t_vals = np.linspace(0, 1, 100)
    fig_umap_3d = px.scatter_3d(
        points, x=0, y=1, z=2,
        title=f"UMAP 3D projection",
        color=labels,
    )

    x_sphere_surface, y_sphere_surface, z_sphere_surface = sphere(0, 0, 0, 1)
    fig_umap_3d.add_trace(
        go.Surface(x=x_sphere_surface, y=y_sphere_surface, z=z_sphere_surface,
                   colorscale=['#f0f3f3', '#f0f3f3'],
                   showscale=False,
                   lighting=dict(ambient=1),
                   opacity=0.75)
    )

    fig_umap_3d.add_trace(go.Scatter3d(
        x=centroid_points[:, 0], y=centroid_points[:, 1], z=centroid_points[:, 2],
        name='centroids',
        mode='markers',
        marker=dict(
            size=12,
            symbol="cross",
            color="black",
        )
    ))

    for region in sv.regions:
        n = len(region)
        for i in range(n):
            start = sv.vertices[region][i]
            end = sv.vertices[region][(i + 1) % n]
            result = geometric_slerp(start, end, t_vals)
            fig_umap_3d.add_trace(go.Scatter3d(
                x=result[..., 0],
                y=result[..., 1],
                z=result[..., 2],
                showlegend=False,
                mode="lines",
                # opacity=value,
                line=dict(
                    color="black",
                    width=3,
                )))

    fig_umap_3d.update_layout(scene=dict(
        xaxis=dict(backgroundcolor="white",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
                showticklabels=False,
                visible=False),
        yaxis=dict(backgroundcolor="white",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
                showticklabels=False,
                visible=False),
        zaxis=dict(backgroundcolor="white",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
                showticklabels=False,
                visible=False)
    ))
    plotly.offline.plot(fig_umap_3d, filename=os.path.join(path, f"voronoi_{speaker}.html"))


if __name__ == "__main__":
    print("Reading conversations...")
    with open(args.input_path) as reader:
        dialogues = json.load(reader)

    model_name = os.path.basename(args.model)
    domains = {}
    new_dialogs = {}
    for dialog_id, dialogue in dialogues.items():
        domain = next(iter(dialogue["goal"]))

        if args.target_domains and domain not in args.target_domains:
            continue

        new_dialogs[dialog_id] = dialogue

        if domain not in domains:
            domains[domain] = {"log": [], "speaker": [], "text": [],
                               "emb": None, "prediction": None}
        domains[domain]["speaker"].extend(turn["tag"].lower() for turn in dialogue["log"][1:-1])

        domains[domain]["text"].extend(get_turn_text(turn) for turn in dialogue["log"][1:-1])
        domains[domain]["log"].extend(dialogue["log"][1:-1])

    for domain in tqdm(domains, desc="Domains"):
        domains[domain]["speaker"] = np.array(domains[domain]["speaker"])
        domains[domain]["text"] = np.array(domains[domain]["text"])
        domains[domain]["labels"] = np.array([get_turn_text(t, use_ground_truth=True)
                                              for t in domains[domain]["log"]])

        if args.model.lower() == "chatgpt" or "openai" in args.model.lower():
            if "openai" in args.model.lower() and "/" in args.model:  # e.g. openai/text-embedding-3-large
                model = os.path.basename(args.model)
            else:
                model = DEFAULT_OPENAI_MODEL
            sentence_encoder = SentenceTransformerOpenAI(model)
        else:
            sentence_encoder = SentenceTransformer(args.model, device=device)

        print(f"Computing sentence embeddings for '{domain}' with {args.model}")
        domains[domain]["emb"] = sentence_encoder.encode(domains[domain]["text"], show_progress_bar=True, batch_size=128)

        normalized_turn_names = {"user": {}, "system": {}}
        for speaker in normalized_turn_names:
            print(f"Clustering {speaker.title()} utterances for '{domain.upper()}'")
            speaker_mask = domains[domain]["speaker"] == speaker
            linkage = "average"
            labels = domains[domain]["labels"][speaker_mask]
            unique_labels, labels_ids, counts = np.unique(labels, return_inverse=True, return_counts=True)
            print("# unique labels:", len(unique_labels))

            unique_labels = unique_labels[counts > 4]
            labels_mask = np.isin(labels, unique_labels)
            print("# unique labels after filtering:", len(unique_labels))

            labels = labels[labels_mask]
            labels_ids = labels_ids[labels_mask]
            embds = domains[domain]["emb"][speaker_mask][labels_mask]

            plot_voronoi_3d(embds, labels_ids, speaker, labels, os.path.join(args.output_path, domain), model_name)
