"""
General purpose script to train SentenceBERT-like models.

Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import os
import json
import time
import torch
import wandb
import hydra
import random
import logging
import accelerate
import numpy as np

from torch import nn
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator

from omegaconf import DictConfig
from tqdm.autonotebook import trange
from typing import Union, List, Iterable, Tuple, Type, Dict

from sentence_transformers import models, datasets, InputExample
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers.util import fullname
from sentence_transformers.evaluation import SentenceEvaluator, SimilarityFunction, EmbeddingSimilarityEvaluator
from sentence_transformers.model_card_templates import ModelCardTemplate


from spretrainer.evaluation import PairClassificationEvaluator, PairLossEvaluator, FewShotClassificationEvaluator
from spretrainer.datasets import SimilarityDataReader, SimilarityDataset, SimilarityDatasetContrastive, BatchedLabelSampler, \
                                 MaxTokensBatchSampler, SimilarityDatasetFromLabels  # noqa: E126
from spretrainer.losses import BaseLoss, SoftmaxLoss, CosineSimilarityLoss, DenoisingAutoEncoderLoss, \
                               MultipleNegativesRankingLoss, MultipleNegativesSymmetricRankingLoss, \
                               HardNegativeSamplingLoss, SimCseLoss, ResponseContrastiveLoss, \
                               BaseContrastiveLoss, LabeledContrastiveLoss, \
                               CONTRASTIVE_LOSS_NAMES, UNSUPERVISED_LOSS_NAMES, UNSUPERVISED_LOSSES  # noqa: E126
from spretrainer.utils import distributed, cache

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt=r'%Y-%m-%d %H:%M:%S',
                    level=logging.INFO, handlers=[LoggingHandler()])

DEFAULT_OPTIMIZER = torch.optim.AdamW

# Globals to be populated in main()
CONTRASTIVE_LABEL_POS = None
CONTRASTIVE_LABEL_NEG = None
CONTRASTIVE_TEMPERATURE = None
SOFT_LABEL_TEMPERATURE = None
CSV_COL_SENT1 = None
CSV_COL_SENT2 = None
CSV_COL_LABEL = None


# TODO: add :param NAME: description to this and all the functions below...
def get_loss_class_name(loss_name: str) -> str:
    """
    Convert to original-name to UpperCamelCase class name.
    """
    return ''.join([w.capitalize() for w in loss_name.split('-')]) + "Loss"


def get_dataset_name(paths: Union[List[str], str]) -> str:
    """Given a path, or a list of paths, return string with file name(s)."""
    if isinstance(paths, str):
        paths = [paths]

    filenames = [os.path.split(path)[1] for path in paths]
    return '|'.join([filename[:filename.find('.')] for filename in filenames])


def get_default_wandb_project_name(path_trainset: str, path_devset: str,
                                   model_name: str, pooling_mode: str, loss: str) -> str:
    """Given the provided parameters return a string to identify the project."""
    project_name = f"train[{get_dataset_name(path_trainset)}]eval[{get_dataset_name(path_devset)}]" + model_name.replace("/", "-") + f"[pooling-{pooling_mode}][loss-{'|'.join(loss)}]"
    return project_name[:128]  # wandb project name can't have more than 128 characters


def get_dataset_by_loss(loss_name: str, data: Iterable, path: str, loss_model: BaseLoss) -> Dataset:
    """Get the proper Dataset object for the given loss."""
    logging.info_dist("Loading dataset...")

    # supervised
    if loss_name == SoftmaxLoss.__name__:
        dataset = SimilarityDataset(data)
    elif loss_name == CosineSimilarityLoss.__name__:
        dataset = SimilarityDataset(data, is_regression=True, normalize_value=True)
    # unsupervised
    elif loss_name == DenoisingAutoEncoderLoss.__name__:
        # Convert the list of sentences in `data` into a list of (original sentence, corrupted sentence)
        dataset = datasets.DenoisingAutoEncoderDataset(data)
    elif loss_name == SimCseLoss.__name__:
        # Convert the list of sentences in `data` into a list of repeated pairs (sentence, sentence)
        dataset = datasets.DenoisingAutoEncoderDataset(data, noise_fn=lambda s: s)  # no noise
    # labeled contrastive
    elif "labeled" in loss_name.lower():
        logging.info_dist("  > labeled dataset for selected loss")
        balance_strategy = path.split(":")[1] if ":" in path else 'none'

        dataset = SimilarityDatasetFromLabels(data,
                                              labels_as_ix=True,
                                              shuffle=True,
                                              balance_labels=balance_strategy)
        logging.info_dist("  > pre-computing label embedings for selected loss")
        loss_model.compute_label_embeddings(dataset)
    # contrastive
    elif loss_name in CONTRASTIVE_LOSS_NAMES:
        dataset = SimilarityDatasetContrastive(data, label_pos=CONTRASTIVE_LABEL_POS, label_neg=CONTRASTIVE_LABEL_NEG)

    dataset.path = path  # used for hashing the object (utils/hashable.py:23)
    return dataset


def get_dataloader_by_loss(loss_name: str, dataset: Dataset, batch_size: int, shuffle: bool = True,
                           model: SentenceTransformer = None, max_seq_length: int = None) -> DataLoader:
    """Get the proper DataLoader for the given loss."""
    sampler = None
    shuffle = True
    if "labeled" in loss_name.lower():
        # TODO: pass sampler to use as argument, part of the path of the dataset?
        # sampler = BatchedLabelSampler(dataset, batch_size=batch_size, num_labels=batch_size)
        # shuffle = None
        pass
    elif loss_name in CONTRASTIVE_LOSS_NAMES and model:
        pass
        # # data_loader = DataLoader(dataset,
        # #                          batch_sampler=MaxTokensBatchSampler(dataset, model,
        # #                                                              max_total_tokens=batch_size * max_seq_length,
        # #                                                              shuffle="label"))
        # sampler = BatchedLabelSampler(dataset, batch_size=batch_size, num_labels=batch_size)
        # shuffle = None
        # # If contrastive, make sure set(data[*].label) = {0, 1}, if not, set everything to 1 (positive)
        # # since BatchedLabelSampler already used the labels.
        # # TODO: if Loss are modified to support multiple labels, then we can remove this part
        # labels = set(sample.label for sample in dataset)
        # if labels != {0, 1} and labels != {1}:
        #     for sample in dataset:
        #         sample.label = 1
        # # return data_loader

    return DataLoader(dataset,
                      shuffle=shuffle,
                      sampler=sampler,
                      batch_size=batch_size,
                      drop_last=True)


def get_loss_by_name(loss_name: str, data: Dataset,
                     model: SentenceTransformer, model_name_or_path: str,
                     accelerator: Accelerator, use_contrastive_head: bool=False) -> BaseLoss:
    """Get the Loss object given its name."""
    # supervised
    if loss_name == SoftmaxLoss.__name__:
        return SoftmaxLoss(model=model,
                           sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                           num_labels=data.num_labels)
    elif loss_name == CosineSimilarityLoss.__name__:
        return CosineSimilarityLoss(model=model)
    # unsupervised
    elif loss_name == DenoisingAutoEncoderLoss.__name__:
        try:
            return DenoisingAutoEncoderLoss(model,
                                            encoder_name_or_path=model_name_or_path,
                                            tie_encoder_decoder=True)
        except ValueError:
            logging.info_dist(f"DenoisingAutoEncoderLoss: Model name or path '{model_name_or_path}' does "
                              "not support being as a decoder."
                              "Trying to use 'bert-base-uncased' as decoder (untied encoder-decoder setting)")
            model_size = "large" if model[0].auto_model.config.hidden_size == 1024 else "base"
            return DenoisingAutoEncoderLoss(model,
                                            encoder_name_or_path=model_name_or_path,
                                            decoder_name_or_path=f"sentence-transformers/nli-bert-{model_size}",
                                            tie_encoder_decoder=False)
    elif loss_name == SimCseLoss.__name__:
        return SimCseLoss(model=model, accelerator=accelerator)
    # contrastive labeled
    elif "labeled" in loss_name.lower():
        return LabeledContrastiveLoss(model=model, use_soft_labels="soft" in loss_name.lower(),
                                      temperature=CONTRASTIVE_TEMPERATURE,
                                      soft_label_model=SOFT_LABEL_MODEL,
                                      soft_label_temperature=SOFT_LABEL_TEMPERATURE,
                                      is_symmetrical=True,
                                      accelerator=accelerator, use_contrastive_head=use_contrastive_head)
    # contrastive
    elif loss_name == MultipleNegativesRankingLoss.__name__:
        return MultipleNegativesRankingLoss(model=model, accelerator=accelerator, use_contrastive_head=use_contrastive_head)
    elif loss_name == MultipleNegativesSymmetricRankingLoss.__name__:
        return MultipleNegativesSymmetricRankingLoss(model=model, accelerator=accelerator, use_contrastive_head=use_contrastive_head)
    elif loss_name == HardNegativeSamplingLoss.__name__:
        return HardNegativeSamplingLoss(model=model, accelerator=accelerator, use_contrastive_head=use_contrastive_head)
    elif loss_name == ResponseContrastiveLoss.__name__:
        return ResponseContrastiveLoss(model=model, accelerator=accelerator, use_contrastive_head=use_contrastive_head)
    else:
        raise ValueError(f"Loss {loss_name} not supported.")


def get_evaluator_by_metric(path_evalset: str, metric: str, metric_avg: str = "",
                            loss_model: BaseLoss = None, batch_size: int = None,
                            evaluator_name: str = '') -> SentenceEvaluator:
    """
    Get the Evaluator object for the given metric name.

    If `metric` == 'loss' then the `loss_model` should contain the concrete loss object to use for evaluation
    """

    # if it's unsupervised
    if metric == "loss" and isinstance(loss_model, UNSUPERVISED_LOSSES):
        # read raw txt file, each line is a sample sentence
        data = list(SimilarityDataReader.read_docs(path_evalset, lines_are_documents=True))
    else:
        data = SimilarityDataReader.read_csv(path_evalset, col_sent0=CSV_COL_SENT1, col_sent1=CSV_COL_SENT2, col_label=CSV_COL_LABEL)

    if metric == "correlation-score":
        evalset = SimilarityDataset(data, is_regression=True, normalize_value=True)
        return EmbeddingSimilarityEvaluator.from_input_examples(
            evalset, main_similarity=SimilarityFunction.COSINE,
            batch_size=batch_size, name=evaluator_name
        )
    elif metric in ["accuracy", "f1-score", "recall", "precision"]:
        try:
            evalset = DataLoader(SimilarityDataset(data), shuffle=False, batch_size=batch_size)
            return PairClassificationEvaluator(evalset, softmax_model=loss_model,
                                               metric=metric, metric_avg=metric_avg,
                                               name=evaluator_name)
        except KeyError:
            # If it is not a Similarity Dataset (i.e. pairs (sent1, sent2)), assume it's
            # single sentence classification task (sent, label)
            data = [InputExample(texts=[text], label=label)
                    for text, label in SimilarityDataReader.read_csv(path_evalset, col_sent0=0, col_sent1=None, col_label=1)]
            return FewShotClassificationEvaluator(data,
                                                  n_shots=5, num_runs=5,
                                                  batch_size=batch_size,
                                                  metric=metric, metric_avg=metric_avg,
                                                  name=evaluator_name)
    elif metric == "loss":
        loss_name = loss_model.__class__.__name__
        evalset = get_dataloader_by_loss(loss_name,
                                         get_dataset_by_loss(loss_name, data, path_evalset, loss_model),
                                         batch_size=batch_size, shuffle=False)
        return PairLossEvaluator(evalset, loss_model, name=evaluator_name)
    else:
        raise ValueError(f"evaluation metric '{metric}' is not supported.")


def wandb_log(epoch: int, steps: int, train_losses: List[float], metrics_result: dict = None) -> None:
    """
    Log evaluation results in WandB.

    When called after finishing each bach, `steps` will be equal to -1.
    """

    # if it's the evaluation perform automatically **after finishing the epoch**, use "epoch" as x-axis
    if metrics_result is not None and steps == -1:
        metrics = {f"{metric_name}_epoch": score for metric_name, score in metrics_result.items()}
        metrics.update({"epoch": epoch + 1})
        wandb.log(metrics)
    elif steps != -1:  # if not just use default wandb steps as x-axis
        # normalize loses
        train_losses = [sum(ll) / len(ll) if ll else 0 for ll in train_losses]

        metrics = metrics_result if metrics_result else {}
        if len(train_losses) > 1:
            metrics.update({f"train_loss_obj{ix}": avg_loss for ix, avg_loss in enumerate(train_losses)})
        elif len(train_losses) == 1:
            metrics.update({"train_loss": train_losses[0]})
        wandb.log(metrics)
        logging.info(f"wandb: new results added to the log - {metrics}")


@torch.no_grad()
def eval_during_training(model: SentenceTransformer, evaluators: Iterable[SentenceEvaluator],
                         output_path: str, save_best_model: bool, epoch: int, steps: int,
                         train_losses: list, accelerator: Accelerator) -> None:
    """Runs evaluation during the training using the provided evalutor object."""
    eval_path = output_path
    if output_path is not None and distributed.is_main_process():
        os.makedirs(output_path, exist_ok=True)

    if evaluators:
        model = accelerator.unwrap_model(model)
        metrics = {}
        for ix, evaluator in enumerate(evaluators):
            eval_path = os.path.join(output_path, f"eval_metric_{ix}")
            os.makedirs(eval_path, exist_ok=True)
            score = evaluator(model,
                              output_path=eval_path if distributed.is_main_process() else None,
                              epoch=epoch, steps=steps)
            metrics[evaluator.metric_name] = score
            if steps >= 1:
                if evaluator.metric_name != "loss" and score > model.best_score[f"metric_{ix}"]:
                    model.best_score[f"metric_{ix}"] = score
                    if save_best_model:
                        best_path = os.path.join(output_path, f"best_model_metric_{ix}")
                        os.makedirs(best_path, exist_ok=True)
                        model.save(best_path)
        if steps > 1:
            wandb_log(epoch, steps, train_losses, metrics)
    elif steps > 1:
        wandb_log(epoch, steps, train_losses)


def save_checkpoint(model: SentenceTransformer,
                    optimizers: List[Optimizer],
                    schedulers: List[lr_scheduler.LambdaLR],
                    losses: List[BaseLoss],
                    accelerator: Accelerator,
                    checkpoint_path, checkpoint_save_total_limit, global_step) -> None:

    # Save model
    if distributed.is_main_process():
        accelerator.unwrap_model(model)._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)
    accelerate.utils.wait_for_everyone()

    # Save optimizers and schedulers
    states = {}
    checkpoint_path = os.path.join(checkpoint_path, str(global_step))
    for ix, optimizer in enumerate(optimizers):
        states.update({f"optimizer_{ix}": optimizer.state_dict()})
    for ix, scheduler in enumerate(schedulers):
        states.update({f"scheduler_{ix}": scheduler.state_dict()})
    torch.save(states,
               os.path.join(checkpoint_path, f"optim_{distributed.get_rank()}.pt"))

    # Save checkpoint-able losses
    checkpoint_path = os.path.join(checkpoint_path, "losses", str(distributed.get_rank()))
    for ix, loss in enumerate(losses):
        if hasattr(loss, "save_state"):
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path, exist_ok=True)
            loss.save_state(os.path.join(checkpoint_path, f"{type(loss).__name__}_{ix}.pt"))


# Modified from the original sentence-bert model.fit() implementation
# (https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py#L575)
@distributed.record_errors
def train(
        model,
        train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
        evaluators: Iterable[SentenceEvaluator] = None,
        accelerator: Accelerator = None,
        epochs: int = 1,
        chunk_size: int = 0,
        scheduler: str = 'WarmupLinear',
        warmup_steps: int = 10000,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params: Dict[str, object] = None,
        optimizer_one_per_objective: bool = True,
        weight_decay: float = 0.01,
        evaluation_steps: int = 0,
        output_path: str = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        show_progress_bar: bool = True,
        checkpoint_path: str = None,
        checkpoint_save_steps: int = 500,
        checkpoint_save_total_limit: int = 0,
        checkpoint_save_after_each_epoch: bool = False,
        checkpoint_last_global_step: int = 0,
        checkpoint_optim_state: dict = None
):
    """
    Train the model with the given training objective(s).

    Each training objective is sampled in turn for one batch.
    We sample only as many batches from each objective as there are in the smallest one
    to make sure of equal training with each dataset.

    :param train_objectives: Tuples of (DataLoader, LossFunction). Pass more than one for multi-task learning
    :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
    :param epochs: Number of epochs for training
    :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
    :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
    :param optimizer_class: Optimizer
    :param optimizer_params: Optimizer parameters (default set {'lr': 2e-5} to if not provided)
    :param optimizer_one_per_objective: Whether or not to multiple or a single optimizer when multiple train objectives are given
    :param weight_decay: Weight decay for model parameters
    :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
    :param output_path: Storage path for the model and evaluation files
    :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
    :param max_grad_norm: Used for gradient normalization.
    :param show_progress_bar: If True, output a tqdm progress bar
    :param checkpoint_path: Folder to save checkpoints during training
    :param checkpoint_save_steps: Will save a checkpoint after so many steps
    :param checkpoint_save_total_limit: Total number of checkpoints to store
    :param checkpoint_save_after_each_epoch: Whether or not to save also checkpoints after each epoch
    :param checkpoint_last_global_step: The last global step checkpointed to continue training from
    """
    if not optimizer_params:
        optimizer_params = {'lr': 2e-5}

    info_loss_functions = []
    for dataloader, loss in train_objectives:
        info_loss_functions.extend(ModelCardTemplate.get_train_objective_info(dataloader, loss))
    info_loss_functions = "\n\n".join([text for text in info_loss_functions])

    info_fit_parameters = json.dumps({"evaluator": [fullname(evaluator) for evaluator in evaluators],
                                      "epochs": epochs, "scheduler": scheduler, "warmup_steps": warmup_steps,
                                      "optimizer_class": str(optimizer_class), "optimizer_params": optimizer_params,
                                      "weight_decay": weight_decay, "evaluation_steps": evaluation_steps,
                                      "max_grad_norm": max_grad_norm}, indent=4, sort_keys=True)
    model._model_card_text = None
    model._model_card_vars['{TRAINING_SECTION}'] = ModelCardTemplate.__TRAINING_SECTION__.replace("{LOSS_FUNCTIONS}", info_loss_functions).replace("{FIT_PARAMETERS}", info_fit_parameters)

    dataloaders = []
    for dataloader, loss in train_objectives:
        # Use smart batching (described in sentence bert paper)
        dataloader.collate_fn = model.smart_batching_collate
        if isinstance(dataloader.batch_sampler, MaxTokensBatchSampler):
            prepared_data_loader = accelerate.data_loader.prepare_data_loader(
                dataloader,
                accelerator.device,
                num_processes=accelerator.num_processes,
                process_index=accelerator.process_index,
                split_batches=accelerator.split_batches,
                put_on_device=accelerator.device_placement if accelerator.distributed_type != accelerate.utils.DistributedType.TPU else False,
                rng_types=accelerator.rng_types.copy(),
                dispatch_batches=accelerator.dispatch_batches,
                even_batches=False
            )
            dataloaders.append(prepared_data_loader)
            accelerator._dataloaders.append(prepared_data_loader)
            loss.even_batches = False
        else:
            dataloaders.append(accelerator.prepare_data_loader(dataloader))

    model.best_score = {f"metric_{ix}": float("-inf") for ix in range(len(evaluators))}
    model = accelerator.prepare_model(model)
    loss_models = [cache.GradCache(loss, chunk_size=chunk_size)  # to scale batch size to virtually any size (for in-batch negatives)
                   if chunk_size > 0 and chunk_size < dataloader.batch_size and isinstance(loss, BaseContrastiveLoss)
                   else loss
                   for dataloader, loss in train_objectives]

    steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])
    num_train_steps = int(steps_per_epoch * epochs)

    # Prepare optimizers
    optimizers = []
    schedulers = []
    for ix, loss_model in enumerate(loss_models):
        param_optimizer = [np for np in loss_model.named_parameters() if "contrastive_head" not in np[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        if hasattr(loss_model, "contrastive_head") and loss_model.contrastive_head:
            lr_scale = 100
            optimizer_grouped_parameters.append({
                'params':loss_model.contrastive_head.parameters(), 'lr': optimizer_params["lr"] * lr_scale
            })

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        scheduler_obj = SentenceTransformer._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

        if checkpoint_optim_state:
            optimizer.load_state_dict(checkpoint_optim_state[f"optimizer_{ix}"])
            scheduler_obj.load_state_dict(checkpoint_optim_state[f"scheduler_{ix}"])

        if optimizer_one_per_objective or len(optimizers) == 0:
            optimizers.append(accelerator.prepare_optimizer(optimizer))
            schedulers.append(accelerator.prepare_scheduler(scheduler_obj))

    global_step = 0
    data_iterators = [iter(dataloader) for dataloader in dataloaders]

    num_train_objectives = len(train_objectives)

    if checkpoint_last_global_step > 0:
        logging.info_dist(f"Previous checkpoint detected, training will skip the first {checkpoint_last_global_step} global steps...")

    skip_scheduler = False
    loss_values = [[] for _ in range(num_train_objectives)]
    for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
        training_steps = 0

        for ix, loss_model in enumerate(loss_models):
            loss_model.zero_grad()
            loss_model.train()
            loss_values[ix].clear()

        for _ in trange(steps_per_epoch, desc="Step", smoothing=0.05, disable=not show_progress_bar):
            for train_idx in range(num_train_objectives):
                data_iterator = data_iterators[train_idx]
                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(dataloaders[train_idx])
                    data_iterators[train_idx] = data_iterator
                    data = next(data_iterator)

                # Skip already processed steps
                if global_step <= checkpoint_last_global_step:
                    continue

                loss_model = loss_models[train_idx]
                optimizer = optimizers[:train_idx + 1][-1]
                scheduler = schedulers[:train_idx + 1][-1]

                tokenized_batch, labels = data

                loss_value = loss_model(tokenized_batch, labels)
                if not isinstance(loss_model, cache.GradCache):
                    accelerator.backward(loss_value)
                accelerator.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                optimizer.step()

                loss_values[train_idx].append(loss_value.detach().item())
                optimizer.zero_grad()

                del data, tokenized_batch, labels
                torch.cuda.empty_cache()

                if not skip_scheduler:
                    scheduler.step()

            training_steps += 1
            global_step += 1
            # Skip already processed steps
            if global_step <= checkpoint_last_global_step:
                continue

            if evaluation_steps > 0 and (training_steps - 1) % evaluation_steps == 0:
                eval_during_training(model, evaluators, output_path, save_best_model, epoch, training_steps, loss_values, accelerator)
                for ix, loss_model in enumerate(loss_models):
                    loss_model.zero_grad()
                    loss_model.train()
                    loss_values[ix].clear()

            if checkpoint_path is not None and checkpoint_save_steps is not None and checkpoint_save_steps > 0 and global_step % checkpoint_save_steps == 0:
                save_checkpoint(model, optimizers, schedulers, loss_models, accelerator,
                                checkpoint_path, checkpoint_save_total_limit, global_step)

        if global_step > checkpoint_last_global_step:
            if checkpoint_save_after_each_epoch:
                save_checkpoint(model, optimizers, schedulers, loss_models, accelerator,
                                checkpoint_path, None, f"epoch-{epoch + 1}")

            eval_during_training(model, evaluators, output_path, save_best_model, epoch, -1, loss_values, accelerator)

    if (not evaluators or evaluators[0].metric_name == "loss") and output_path is not None and distributed.is_main_process():
        accelerator.unwrap_model(model).save(output_path)

    if checkpoint_path is not None:
        save_checkpoint(model, optimizers, schedulers, loss_models, accelerator,
                        checkpoint_path, checkpoint_save_total_limit, global_step)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    global CONTRASTIVE_LABEL_POS, CONTRASTIVE_LABEL_NEG, \
           CONTRASTIVE_TEMPERATURE, SOFT_LABEL_TEMPERATURE, SOFT_LABEL_MODEL, \
           CSV_COL_SENT1, CSV_COL_SENT2, CSV_COL_LABEL

    CONTRASTIVE_LABEL_POS = cfg.contrastive_learning.label_pos
    CONTRASTIVE_LABEL_NEG = cfg.contrastive_learning.label_neg
    CONTRASTIVE_TEMPERATURE = cfg.contrastive_learning.softmax_temperature
    SOFT_LABEL_TEMPERATURE = cfg.contrastive_learning.soft_label_temperature
    SOFT_LABEL_MODEL = cfg.contrastive_learning.soft_label_model

    CSV_COL_SENT1 = cfg.datasets.csv.column_name_sent1
    CSV_COL_SENT2 = cfg.datasets.csv.column_name_sent2
    CSV_COL_LABEL = cfg.datasets.csv.column_name_ground_truth

    num_epochs = cfg.training.num_epochs
    batch_sizes = cfg.training.batch_size
    learning_rate = cfg.training.learning_rate
    evals_per_epoch = cfg.evaluation.evaluations_per_epoch
    checkpoint_saves_per_epoch = cfg.checkpointing.saves_per_epoch
    optimizer = DEFAULT_OPTIMIZER

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    if isinstance(cfg.target.trainsets, str):
        cfg.target.trainsets = [cfg.target.trainsets]
    if isinstance(cfg.target.losses, str):
        cfg.target.losses = [cfg.target.losses]
    if isinstance(batch_sizes, int):
        batch_sizes = [batch_sizes]
    if isinstance(cfg.evaluation.devset, str):
        cfg.evaluation.devset = [cfg.evaluation.devset]
    elif not cfg.evaluation.devset:
        cfg.evaluation.devset = []
    if not cfg.evaluation.testset:
        cfg.evaluation.testset = ''

    # 0. Set up distributed and cache
    device, rank, local_rank, world_size = distributed.init()
    accelerator = Accelerator()
    logging.info_dist = (lambda self, msg: self.info(msg) if accelerator.is_main_process else None).__get__(logging)  # log only if main process

    if cfg.cache.enabled:
        cache.init(cfg.cache.path, cfg.cache.size, cfg.cache.verbose)

    # 1. Set up the model
    # 1.1. Check if there's a previous checkpoint saved
    model_name_or_path = cfg.model.base
    last_global_step, last_epoch = 0, 0
    checkpoint_optim_state = None
    wandb_id, wandb_project_name, wandb_group_name = None, None, None

    if cfg.training.continue_from_last_checkpoint and os.path.exists(cfg.checkpointing.path):
        wandb_info_file = os.path.join(cfg.checkpointing.path, f"wandb_{distributed.get_rank()}.json")
        if os.path.exists(wandb_info_file):
            with open(os.path.join(cfg.checkpointing.path, f"wandb_{distributed.get_rank()}.json")) as reader:
                wandb_info = json.load(reader)
                wandb_id = wandb_info["id"]
                wandb_group_name = wandb_info["group_name"]
                wandb_project_name = wandb_info["project_name"]
                logging.info_dist(f"wandb: previous project name '{wandb_project_name}' identified to be resumed")
                if wandb_group_name:
                    logging.info_dist(f"wandb: previous group name '{wandb_group_name}' identified to be resumed")
                logging.info(f"wandb: previous run '{wandb_id}' identified to be resumed")

        checkpoints_global_steps = sorted([int(cp) for cp in os.listdir(cfg.checkpointing.path)
                                           if cp.isdigit()])
        checkpoints_epochs = sorted([int(cp[len("epoch-"):])
                                     for cp in os.listdir(cfg.checkpointing.path)
                                     if cp.startswith("epoch-")])
        if checkpoints_global_steps:
            last_global_step = checkpoints_global_steps[-1]
            model_name_or_path = os.path.join(cfg.checkpointing.path, str(last_global_step))
        elif checkpoints_epochs:
            last_epoch = checkpoints_epochs[-1]
            model_name_or_path = os.path.join(cfg.checkpointing.path, f"epoch-{last_epoch}")

    # 1.2. Load model
    # If there's a previous checkpoint to load, use it...
    load_previous_checkpoint = last_global_step + last_epoch > 0
    if load_previous_checkpoint:
        logging.info_dist(f"Loading last model checkpoint from '{model_name_or_path}' to resume previous training.")
        # load checkpointed model
        model = SentenceTransformer(model_name_or_path, device=device)

        optimizer_state_path = os.path.join(model_name_or_path, f"optim_{distributed.get_rank()}.pt")
        if os.path.exists(optimizer_state_path):
            # Load checkpointed optimizers and schedulers
            logging.info_dist(f"Loading last optimizers and schedulers checkpoint from '{optimizer_state_path}' to resume previous training.")
            checkpoint_optim_state = torch.load(optimizer_state_path)

    # Otherwise, build it from the provided base model...
    else:
        transformer_seq_encoder = models.Transformer(model_name_or_path, max_seq_length=cfg.model.max_seq_length)

        if cfg.model.special_tokens:
            transformer_seq_encoder.tokenizer.add_tokens(cfg.model.special_tokens, special_tokens=True)
            transformer_seq_encoder.auto_model.resize_token_embeddings(len(transformer_seq_encoder.tokenizer))

        # TODO: if cfg.model.pooling_mode == "mean-sqrt" => pooling_mode_mean_tokens=False, pooling_mode_mean_sqrt_len_tokens=True
        sentence_vector = models.Pooling(transformer_seq_encoder.get_word_embedding_dimension(), pooling_mode=cfg.model.pooling_mode)
        model = SentenceTransformer(modules=[transformer_seq_encoder, sentence_vector], device=device)

    # 2. Set up wandb
    if wandb_project_name is None:
        wandb_project_name = (cfg.wandb.project_name or get_default_wandb_project_name(cfg.target.trainsets, cfg.evaluation.devset,
                                                                                       cfg.model.base, cfg.model.pooling_mode, cfg.target.losses))
    if wandb_group_name is None:
        wandb_group_name = f"distributed-{distributed.broadcast_value(int(time.time()))}" if distributed.is_distributed() else None

    logging.info(f"wandb: init project '{wandb_project_name}'." + (f" Runs will be grouped by the name '{wandb_group_name}'." if wandb_group_name else ''))

    if wandb_id is None:
        wandb_id = wandb.util.generate_id()

        if cfg.checkpointing.path:
            os.makedirs(cfg.checkpointing.path, exist_ok=True)
            with open(os.path.join(cfg.checkpointing.path, f"wandb_{distributed.get_rank()}.json"), "w") as writer:
                json.dump({"id": wandb_id,
                           "group_name": wandb_group_name,
                           "project_name": wandb_project_name}, writer)

    wandb.init(
        id=wandb_id,
        project=wandb_project_name[:128],
        group=wandb_group_name,
        resume="allow",
        config=dict(cfg)
    )
    wandb.define_metric("epoch")
    wandb.define_metric(f"{cfg.evaluation.metric}_epoch", step_metric="epoch")

    wandb.watch(model, log_freq=cfg.wandb.log_freq)

    # 3. Loading datasets, data loaders and losses
    # 3.1. Training sets
    logging.info_dist(f"Reading training sets ({cfg.target.trainsets})")
    train_objectives = []
    target_losses = [get_loss_class_name(loss_name) for loss_name in cfg.target.losses]
    for ix, path in enumerate(cfg.target.trainsets):
        loss_name = target_losses[:ix + 1][-1]  # trick to avoid IndexError when there are more datasets than losses by returning the last one
        batch_size = batch_sizes[:ix + 1][-1]

        # if it's unsupervised
        if loss_name in UNSUPERVISED_LOSS_NAMES:
            # read raw txt file, each line is a sample sentence
            data = list(SimilarityDataReader.read_docs(path, lines_are_documents=True))
        # if it's labeled
        elif "labeled" in loss_name.lower():
            data = SimilarityDataReader.read_csv(path, col_sent0=0, col_sent1=None, col_label=1, ignore_header=True)
        else:
            data = SimilarityDataReader.read_csv(path, col_sent0=CSV_COL_SENT1, col_sent1=CSV_COL_SENT2, col_label=CSV_COL_LABEL)

        loss_fn = get_loss_by_name(loss_name, data, model, model_name_or_path, accelerator, cfg.contrastive_learning.use_contrastive_head)
        if load_previous_checkpoint and hasattr(loss_fn, "load_state"):
            loss_fn.load_state(os.path.join(model_name_or_path, "losses", str(distributed.get_rank()), f"{type(loss_fn).__name__}_{ix}.pt"))
        loss_fn.to(device)

        logging.info_dist("Creating dataloader...")
        train_objectives.append(
            # (data_loader, loss_fn) pair
            (get_dataloader_by_loss(loss_name,
                                    get_dataset_by_loss(loss_name, data, path, loss_fn),
                                    batch_size=batch_size,
                                    model=model,
                                    max_seq_length=cfg.model.max_seq_length),
             loss_fn)
        )
        logging.info_dist("Dataloader ready.")

    # If multiple training sets, sets will be repeated as in a round-robin queue
    # Thus, 1 epoch = 1 epoch with the smallest training set (increase epoch to cover more parts of the bigger ones)
    steps_per_epoch = min([len(data_loader) // accelerator.num_processes for data_loader, _ in train_objectives])
    if len(train_objectives) > 1:
        logging.info_dist(f"Multiple training sets: steps per epoch set to {steps_per_epoch} steps, the smallest set:")
        for ix, path in enumerate(cfg.target.trainsets):
            logging.info_dist(f"    {ix + 1}. '{path}': {len(train_objectives[ix][0]) // accelerator.num_processes} steps")

    # In case the evaluation metric is "loss",
    # We assume here the first loss provided by the user, at [0], is the one is used for evaluation
    # (Should be somehow allow the user to specify a different one? e.g. cfg.evaluation.target_loss_ix?)
    _, evaluation_loss = train_objectives[0]

    # 3.2. Evaluation/development sets
    dev_evaluators = []
    for ix, devset in enumerate(cfg.evaluation.devset):
        logging.info_dist(f"Reading development set ({devset})")
        dev_evaluator = get_evaluator_by_metric(devset,
                                                cfg.evaluation.metric,
                                                cfg.evaluation.metric_avg,
                                                evaluation_loss,
                                                batch_size=batch_sizes[0], evaluator_name=f"devset_{ix}")
        dev_evaluator.write_csv = not distributed.is_distributed() or distributed.is_main_process()
        dev_evaluator.metric_name = cfg.evaluation.metric

        dev_evaluators.append(dev_evaluator)

    # 4. Training
    logging.info_dist("Warmup steps: {}".format(cfg.training.warmup_steps))
    train(model,
          train_objectives=train_objectives,
          evaluators=dev_evaluators,
          accelerator=accelerator,
          epochs=num_epochs,
          chunk_size=cfg.training.chunk_size,
          evaluation_steps=max(steps_per_epoch // evals_per_epoch, cfg.evaluation.min_steps) if evals_per_epoch > 0 else 0,
          warmup_steps=cfg.training.warmup_steps,
          output_path=cfg.evaluation.best_model_output_path,
          optimizer_class=optimizer,
          optimizer_params={'lr': learning_rate},
          optimizer_one_per_objective=not cfg.training.use_single_optimizer,
          checkpoint_path=cfg.checkpointing.path,
          checkpoint_save_steps=max(steps_per_epoch // checkpoint_saves_per_epoch, cfg.checkpointing.min_steps) if checkpoint_saves_per_epoch else 0,
          checkpoint_save_total_limit=cfg.checkpointing.total_limit,
          checkpoint_save_after_each_epoch=cfg.checkpointing.always_save_after_each_epoch,
          checkpoint_last_global_step=last_global_step if last_global_step > 0 else last_epoch * steps_per_epoch,
          checkpoint_optim_state=checkpoint_optim_state)

    # 5. If test set, then evaluate model on it...
    if cfg.evaluation.testset and distributed.is_main_process():
        logging.info(f"Loading final model from disk ({cfg.evaluation.best_model_output_path})")
        model = SentenceTransformer(cfg.evaluation.best_model_output_path)

        torch.cuda.empty_cache()
        model.to(model._target_device)

        logging.info(f"Reading the test set ({cfg.evaluation.testset})")
        test_evaluator = get_evaluator_by_metric(cfg.evaluation.testset, cfg.evaluation.metric, cfg.evaluation.metric_avg, evaluation_loss, batch_size=batch_sizes[0], evaluator_name="testset")

        logging.info("Evaluating model on the test set data...")
        with torch.no_grad():
            test_evaluator(model, output_path=cfg.evaluation.best_model_output_path)

    distributed.destroy()
    wandb.finish()


if __name__ == '__main__':
    main()
