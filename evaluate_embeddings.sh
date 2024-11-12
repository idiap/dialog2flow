#!/bin/bash
PATH_DATASET="data/$1"
PATH_CONVERSATIONS="$PATH_DATASET/trajectories.single_domain.json"
PATH_OUTPUT="output/results/$1"

# auto
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

for model in "${models[@]}"; do
    python evaluate_embeddings.py -m "$model" -o "$PATH_OUTPUT" -i "$PATH_DATASET/trajectories.single_domain.json"
done
python show_embedding_results.py -i "$PATH_OUTPUT"
