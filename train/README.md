# Training

Before starting, if you haven't done it yet, make sure your environment and dependencies are all set up:
```bash
# creates the corresponding "dialog2flow" conda environment
conda env create -f conda.yml
conda activate dialog2flow
pip install -r requirements.txt
```
_(:point_right: in case your having issues during installation, we also have exported the full conda environment and requirements ([`conda-exported.yml`](conda-exported.yml)) and ([`requirements-exported.txt`](requirements-exported.txt)) for you to use the exact same environemt used the last time we tested the code)_

1. Training data: Data is located in `data` folder. You need to download the 3 missing files for training data (`dialog-acts+slots.csv`, `dialog-acts.csv`, `slots.csv`) as follows:
    ```bash
    cd data/
    wget https://huggingface.co/datasets/sergioburdisso/dialog2flow-dataset/resolve/main/dialog-acts%2Bslots.csv
    wget https://huggingface.co/datasets/sergioburdisso/dialog2flow-dataset/resolve/main/dialog-acts.csv
    wget https://huggingface.co/datasets/sergioburdisso/dialog2flow-dataset/resolve/main/slots.csv
    ```
    _(:point_right: alternatively, you can create these files from scratch, for instance, to remove samples from certain datasets in case of License issues, see Section "Training Data Generation" below)_
2. Train the models described in the paper:
    - **D2F**$_{joint}$
        ```bash
        python main.py \
            target.trainsets='["data/dialog-acts.csv:extend","data/slots.csv"]' \
            target.losses='["soft-labeled-contrastive-loss", "soft-labeled-contrastive-loss"]' \
            contrastive_learning.use_contrastive_head=True \
            contrastive_learning.softmax_temperature=0.05 \
            contrastive_learning.soft_label_temperature=0.35 \
            contrastive_learning.soft_label_model="multi-qa-mpnet-base-dot-v1" \
            model.base="bert-base-uncased" \
            model.max_seq_length=64 \
            model.pooling_mode="mean" \
            training.num_epochs=15 \
            training.batch_size=64 \
            training.chunk_size=0 \
            training.learning_rate=3e-06 \
            evaluation.metric="f1-score" \
            evaluation.devset="data/val.csv" \
            evaluation.testset=null \
            evaluation.evaluations_per_epoch=300 \
            evaluation.best_model_output_path="models/d2f_joint/" \
            wandb.project_name="d2f_joint" \
            checkpointing.path="models/d2f_joint/checkpoints" \
            checkpointing.always_save_after_each_epoch=False \
            checkpointing.saves_per_epoch=0
        ```
    - **D2F**$_{single}$
        ```bash
        python main.py \
            target.trainsets="data/dialog-acts+slots.csv" \
            target.losses="soft-labeled-contrastive-loss" \
            contrastive_learning.use_contrastive_head=True \
            contrastive_learning.softmax_temperature=0.05 \
            contrastive_learning.soft_label_temperature=0.35 \
            contrastive_learning.soft_label_model="multi-qa-mpnet-base-dot-v1" \
            model.base="bert-base-uncased" \
            model.max_seq_length=64 \
            model.pooling_mode="mean" \
            training.num_epochs=15 \
            training.batch_size=64 \
            training.chunk_size=0 \
            training.learning_rate=3e-06 \
            evaluation.metric="f1-score" \
            evaluation.devset="data/val.csv" \
            evaluation.testset=null \
            evaluation.evaluations_per_epoch=100 \
            evaluation.best_model_output_path="models/d2f_single/" \
            wandb.project_name="d2f_single" \
            checkpointing.path="models/d2f_single/checkpoints" \
            checkpointing.always_save_after_each_epoch=False \
            checkpointing.saves_per_epoch=0
        ```
    - **D2F-Hard**$_{joint}$
        ```bash
        python main.py \
            target.trainsets='["data/dialog-acts.csv:extend","data/slots.csv"]' \
            target.losses='["labeled-contrastive-loss", "labeled-contrastive-loss"]' \
            contrastive_learning.use_contrastive_head=True \
            contrastive_learning.softmax_temperature=0.05 \
            model.base="bert-base-uncased" \
            model.max_seq_length=64 \
            model.pooling_mode="mean" \
            training.num_epochs=15 \
            training.batch_size=64 \
            training.chunk_size=0 \
            training.learning_rate=3e-06 \
            evaluation.metric="f1-score" \
            evaluation.devset="data/val.csv" \
            evaluation.testset=null \
            evaluation.evaluations_per_epoch=300 \
            evaluation.best_model_output_path="models/d2f-hard_joint/" \
            wandb.project_name="d2f-hard_joint" \
            checkpointing.path="models/d2f-hard_joint/checkpoints" \
            checkpointing.always_save_after_each_epoch=False \
            checkpointing.saves_per_epoch=0
        ```
    - **D2F-Hard**$_{single}$
        ```bash
        python main.py \
            target.trainsets="data/dialog-acts+slots.csv" \
            target.losses="labeled-contrastive-loss" \
            contrastive_learning.use_contrastive_head=True \
            contrastive_learning.softmax_temperature=0.05 \
            model.base="bert-base-uncased" \
            model.max_seq_length=64 \
            model.pooling_mode="mean" \
            training.num_epochs=15 \
            training.batch_size=64 \
            training.chunk_size=0 \
            training.learning_rate=3e-06 \
            evaluation.metric="f1-score" \
            evaluation.devset="data/val.csv" \
            evaluation.testset=null \
            evaluation.evaluations_per_epoch=100 \
            evaluation.best_model_output_path="models/d2f-hard_single/" \
            wandb.project_name="d2f-hard_single" \
            checkpointing.path="models/d2f-hard_single/checkpoints" \
            checkpointing.always_save_after_each_epoch=False \
            checkpointing.saves_per_epoch=0
        ```
Models checkpoints will be saved in `models`, inside their own folders with the following structure:
```
- models/
  |- d2f_joint/
  |   +- best_model_metric_0/ (best overall checkpoint in this folder)
  |   +- checkpoints/ (all checkpoints in this folder)
  +- d2f_single/
  +- d2f-hard_joint/
  +- d2f-hard_single/
```


## Training Data Generation

You can use the `create_training_data.py` script to create the three CSV files used for training (`dialog-acts+slots.csv`, `dialog-acts.csv`, `slots.csv`) from the unified dataset, for instance, if you want to use only some of the datasets given License restrictions:
```bash
cd data/
python create_tod_datasets.py -d DATASET1 DATASET2 ... DATASETK
python create_training_data.py
```

Where each DATASET is the name of the target dataset (list given below). The  The `create_tod_datasets.py` script converts the provided list of dataset to the unified format needed to create the training files, as described in the following section.

## Unified TOD dataset

You can get access to the whole dataset in 🤗 Hugging Face by clicking [here](https://huggingface.co/datasets/sergioburdisso/dialog2flow-dataset), or alternativelly, you can created from scratch simply by running:
```bash
python create_tod_datasets.py
```
which by default will download and convert all the datasets to the unified format.
In case you want to convert only certain datasets, for instance, to avoid license issues, you can use the `-d` argument followed by the list of dataset names to use, for instance, to process only "ABCD" and "BiTOD" datasets, we can run:
```
python create_tod_datasets.py -d ABCD BiTOD
```
By default converted datasets will be saved in `tod_datasets/`, in case a different path is needed, you can use the `-o PATH` argument.

### Details of the 20 available TOD datasets

| Dataset Name               | Train  | Validation | Test  | Total  | License                                                     |
|--------------------|--------|------------|-------|--------|-------------------------------------------------------------|
| ABCD               | 8034   | 1004       | 1004  | 10042  | MIT License                                                 |
| BiTOD              | 2952   | 295        | 442   | 3689   | Apache License 2.0                                          |
| Disambiguation     | 8433   | 999        | 1000  | 10432  | MiT License                                                 |
| DSTC2-Clean        | 1612   | 506        | 1117  | 3235   | GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007          |
| FRAMES             | 1329   | -          | 40    | 1369   | GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007          |
| GECOR              | 676    | -          | -     | 676    | CC BY 4.0                                                   |
| HDSA-Dialog        | 8438   | 1000       | 1000  | 10438  | MIT License                                                 |
| KETOD              | 4247   | 545        | 532   | 5324   | MiT License                                                 |
| MS-DC              | 10000  | -          | -     | 10000  | MICROSOFT RESEARCH LICENSE TERMS                            |
| MulDoGO            | 59939  | 1150       | 2319  | 63408  | Community Data License Agreement – Permissive – Version 1.0 |
| MultiWOZ_2.1       | 8434   | 999        | 1000  | 10433  | MiT License                                                 |
| MULTIWOZ2_2        | 8437   | 1000       | 1000  | 10437  | Mit License                                                 |
| SGD                | 16142  | 2482       | 4201  | 22825  | CC BY-SA 4.0                                                |
| SimJointGEN        | 100000 | 10000      | 10000 | 120000 | No license                                                  |
| SimJointMovie      | 384    | 120        | 264   | 768    | No license                                                  |
| SimJointRestaurant | 1116   | 349        | 775   | 2240   | No license                                                  |
| Taskmaster1        | 6170   | 769        | 769   | 7708   | Attribution 4.0 International (CC BY 4.0)                   |
| Taskmaster2        | 17304  | -          | -     | 17304  | Creative Commons Attribution 4.0 License (CC BY 4.0)        |
| Taskmaster3        | 22724  | 17019      | 17903 | 57646  | Creative Commons Attribution 4.0 License (CC BY 4.0)        |
| WOZ2_0             | 600    | 200        | 400   | 1200   | Apache License 2.0                                          |


## Proposed Soft-Contrastive Loss

Details about loss implementation can be found in the [main `README.md`](../#proposed-soft-contrastive-loss).

## License

Individual datasets were originally loaded from [DialogStudio](https://huggingface.co/datasets/Salesforce/dialogstudio) and therefore, this project follows [their licensing structure](https://huggingface.co/datasets/Salesforce/dialogstudio/blob/main/README.md#license).
For detailed licensing information, please refer to the specific licenses accompanying the datasets provided in the table above.

All extra content purely authored by us is released under the MIT license:

Copyright (c) 2024 [Idiap Research Institute](https://www.idiap.ch/).

MIT License.
