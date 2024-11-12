# -*- coding: utf-8 -*-
"""
Given path for the ground truth graph and the extracted graph generated with `build_graph.py`,
compute graph size difference between them and print the results, as reported in the paper (Table 6).

Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import os
import argparse
import networkx as nx

from util import STR_AVERAGE_COLUMN, show_results

# e.g python evaluate_graph.py -i "output/graph" -gt "output/graph/ground_truth"
parser = argparse.ArgumentParser(prog="Evaluate the graphs generated with `build_graph.py`.")
parser.add_argument("-gt", "--groundtruth-path", help="Path to the ground truth graphs", required=True)
parser.add_argument("-i", "--inference-path", help="Path to the generated graphs", required=True)
args = parser.parse_args()


if __name__ == "__main__":
    domains = [d for d in os.listdir(args.groundtruth_path)]
    domains.append(STR_AVERAGE_COLUMN)
    results = {}
    for domain in domains:
        path_graph = os.path.join(args.groundtruth_path, domain)
        if not os.path.isdir(path_graph):
            continue

        G_true = nx.read_graphml(os.path.join(path_graph, "graph.graphml"))
        print(f"\nGround truth '{domain.upper()}' graph loaded ({len(G_true.nodes)} nodes and {len(G_true.edges)} edges)")

        for model in os.listdir(args.inference_path):
            path_graph = os.path.join(args.inference_path, model)
            if not os.path.isdir(path_graph)  or os.path.normpath(path_graph) == os.path.normpath(args.groundtruth_path):
                continue

            path_graph = os.path.join(path_graph, domain, "graph.graphml")
            if not os.path.exists(path_graph):
                raise ValueError(f"Required generated graph does not exist in '{path_graph}'")

            G = nx.read_graphml(path_graph)

            print(f"  Extracted graph with '{model}' has {len(G.nodes)} nodes and {len(G.edges)} edges")

            if model not in results:
                results[model] = {}
            results[model][domain] = {
                "nodes": len(G.nodes),
                "true-nodes": len(G_true.nodes),
                "diff-nodes": len(G.nodes) - len(G_true.nodes),
                "diff-nodes-norm": abs((len(G_true.nodes) - len(G.nodes)) / len(G_true.nodes)),
            }

    models = list(results.keys())
    print("\n\n=============== GRAPH RESULTS ===============")
    show_results(models, [d for d in domains],
                 lambda model, domain: results[model][domain][f"diff-nodes-norm"],
                 sorted=True, metric_is_ascending=True, percentage=True,
                 value_extra_getter=lambda model, domain: results[model][domain][f"diff-nodes"] if domain in results[model] else None,
                 column_value_getter=lambda model, domain: results[model][domain][f'true-nodes'] if domain in results[model] else None)
