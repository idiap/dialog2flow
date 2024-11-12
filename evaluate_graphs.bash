#!/bin/bash
PATH_DATASET="data/spokenwoz"
PATH_CONVERSATIONS="$PATH_DATASET/trajectories.single_domain.json"
PATH_OUTPUT="output"

models=(
  # Dialog2Flow models
  "sergioburdisso/dialog2flow-single-bert-base"  # D2F_single
  "sergioburdisso/dialog2flow-joint-bert-base"  # D2F_joint
  # "models/d2f-hard_single"  # D2F-Hard_single ------> (UNCOMMENT line if already unzipped)
  # "models/d2f-hard_joint"  # D2F-Hard_joint ------> (UNCOMMENT line if already unzipped)

  # Baselines
  "aws-ai/dse-bert-base"  # DSE
  "sergioburdisso/space-2"  # SPACE-2
  "microsoft/DialoGPT-medium"  # DialoGPT
  "bert-base-uncased"  # BERT
  "openai/text-embedding-3-large"  # OpenAI
  "all-mpnet-base-v2"  # Sentence-BERT
  "sentence-transformers/gtr-t5-base"  # GTR-T5
  # "models/todbert_sbd"  # SBD-BERT ------> (UNCOMMENT line if already unzipped)
  "sentence-transformers/average_word_embeddings_glove.840B.300d"  # GloVe
  "TODBERT/TOD-BERT-JNT-V1"  # TOD-BERT
)

# 1) Generate the action trajectories for each model, by clustering dialog utterances
for model in "${models[@]}"; do
    python extract_trajectories.py -o "$PATH_OUTPUT/" -m "$model" -i "$PATH_CONVERSATIONS"
done

# 2) Generate the ground truth / reference graphs (one graph dormain)
python build_graph.py -i "$PATH_DATASET/trajectories.single_domain.json" -o "$PATH_OUTPUT/graph/ground_truth" -te 0.02
# 3) Generate the graphs using the generated trajectories (one graph domain)
python build_graph.py -i "$PATH_OUTPUT" -o "$PATH_OUTPUT/graph" -te 0.02
# 4) Evaluate reference graph vs. generated graph for each domain
python evaluate_graph.py -i "$PATH_OUTPUT/graph" -gt "$PATH_OUTPUT/graph/ground_truth"
