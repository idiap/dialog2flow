#!/bin/bash
PATH_DATASET="data/spokenwoz"
PATH_CONVERSATIONS="$PATH_DATASET/trajectories.single_domain.json"
PATH_OUTPUT="output/"

# 1) Generate the ground truth graph plot for hospital
python build_graph.py -i "$PATH_DATASET/trajectories.single_domain.json" -o "$PATH_OUTPUT/graph_plot/ground_truth" -png -d $1

# 2) Generate the graphs plots using the generated trajectories
python build_graph.py -i "$PATH_OUTPUT" -o "$PATH_OUTPUT/graph_plot" -png -d $1
