"""
Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import os
import torch
import torch.distributed as dist

from datetime import timedelta
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: F401
from torch.nn.parallel import DataParallel as DP  # noqa: F401
from torch.utils.data.distributed import DistributedSampler as Sampler  # noqa: F401
from torch.distributed.elastic.multiprocessing.errors import record as record_errors  # noqa: F401


BLOCKING_TIMEOUT = 6  # hours


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process():
    return get_rank() == 0


def save_if_main_process(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def print_if_main_process(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def broadcast_value(value, src=0):
    if is_distributed():
        object_list = [value]
        dist.broadcast_object_list(object_list, src=src)
        return object_list[0]
    else:
        return value


def barrier():
    if is_distributed():
        dist.barrier()


def init():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device("cuda", local_rank)
        nb_gpu = torch.cuda.device_count()
        if nb_gpu > 1:
            os.environ["NCCL_BLOCKING_WAIT"] = "1"
            dist.init_process_group("nccl", timeout=timedelta(seconds=60 * 60 * BLOCKING_TIMEOUT))
    else:
        device = torch.device("cpu")

    rank = get_rank()
    local_rank = get_local_rank()
    world_size = get_world_size()
    # torch.cuda.set_device(rank)

    return device, rank, local_rank, world_size


def destroy():
    if is_distributed() and is_main_process():
        dist.destroy_process_group()
