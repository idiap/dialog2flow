"""
Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import os
import torch

from tqdm import tqdm
from typing import List, Callable, Any, Iterable, Dict
from joblib import Memory
from itertools import repeat
from contextlib import nullcontext
from collections import UserDict

from torch import nn, Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import get_device_states, set_device_states

from . import hashable
from ..losses import BaseContrastiveLoss


class DummyMemory():
    def cache(self, function):
        return function

    def clear(self):
        pass


memory = DummyMemory()


def init(path, gigabytes_limit, verbose=0):
    global memory
    memory = Memory(os.path.join(path, "__cache__"),
                    bytes_limit=gigabytes_limit * (2 ** 30),
                    verbose=verbose)


def clear():
    memory.clear()


def get_cached(function):
    return memory.cache(function)


def get_cached_hashed(function):
    if isinstance(memory, DummyMemory):
        return function

    # TODO: double check, not sure it's working as expected
    return memory.cache(hashable.hash_args(function))


# Taken from the original GradCache implementation
# (https://github.com/luyug/GradCache/blob/main/src/grad_cache/context_managers.py#L5)
class RandContext:
    def __init__(self, *tensors):
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self):
        self._fork = torch.random.fork_rng(
            devices=self.fwd_gpu_devices,
            enabled=True
        )
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


# Adapted from the original GradCache implementation
# (https://github.com/luyug/GradCache/blob/main/src/grad_cache/grad_cache.py#L16)
class GradCache(nn.Module):
    """
    Gradient Cache class. Implements input chunking, first graph-less forward pass, Gradient Cache creation, second
    forward & backward gradient computation. Optimizer step is not included. Native torch automatic mixed precision is
    supported. User needs to handle gradient unscaling and scaler update after a gradeitn cache step.
    """
    def __init__(
            self,
            loss: BaseContrastiveLoss,
            chunk_size: int,
            split_input_fn: Callable[[Any, int], Any] = None,
            get_rep_fn: Callable[..., Tensor] = None,
            fp16: bool = False,
            scaler: GradScaler = None,
    ):
        """
        Initialize the Gradient Cache class instance.
        :param chunk_sizes: An integer indicating chunk size. Or a list of integers of chunk size for each model.
        :param loss: A BaseContrastiveLoss object. It should not in any case modify the input tensors' relations
        in the autograd graph, which are later relied upon to create the gradient cache.
        :param split_input_fn: An optional function that split generic model input into chunks. If not provided, this
        class will try its best to split the inputs of supported types. See `split_inputs` function.
        :param get_rep_fn: An optional function that takes generic model output and return representation tensors. If
        not provided, the generic output is assumed to be the representation tensor.
        :param fp16: If True, run mixed precision training, which requires scaler to also be set.
        :param scaler: A GradScaler object for automatic mixed precision training.
        """
        super(GradCache, self).__init__()

        self.model = loss.model
        self.contrastive_head = loss.contrastive_head
        self.chunk_size = chunk_size

        self.split_input_fn = split_input_fn
        self.get_rep_fn = get_rep_fn
        self.loss = loss

        if fp16:
            assert scaler is not None, "mixed precision training requires a gradient scaler passed in"

        self.fp16 = fp16
        self.scaler = scaler

        self._get_input_tensors_strict = False

    def __call__(self, *args, **kwargs):
        """
        Call the cache_step function.
        :return: Current step loss.
        """
        return self.cache_step(*args, **kwargs)

    def split_inputs(self, model_input, chunk_size: int) -> List:
        """
        Split input into chunks. Will call user provided `split_input_fn` if specified. Otherwise,
        it can handle input types of tensor, list of tensors and dictionary of tensors.
        :param model_input: Generic model input.
        :param chunk_size:  Size of each chunk.
        :return: A list of chunked model input.
        """
        # delegate splitting to user provided function
        if self.split_input_fn is not None:
            return self.split_input_fn(model_input, chunk_size)

        if isinstance(model_input, (dict, UserDict)) and all(isinstance(x, Tensor) for x in model_input.values()):
            keys = list(model_input.keys())
            chunked_tensors = [model_input[k].split(chunk_size, dim=0) for k in keys]
            return [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]

        elif isinstance(model_input, list) and all(isinstance(x, Tensor) for x in model_input):
            chunked_x = [t.split(chunk_size, dim=0) for t in model_input]
            return [list(s) for s in zip(*chunked_x)]

        elif isinstance(model_input, Tensor):
            return list(model_input.split(chunk_size, dim=0))

        elif isinstance(model_input, tuple) and list(map(type, model_input)) == [list, dict]:
            args_chunks = self.split_inputs(model_input[0], chunk_size)
            kwargs_chunks = self.split_inputs(model_input[1], chunk_size)
            return list(zip(args_chunks, kwargs_chunks))

        else:
            raise NotImplementedError(f'Model input split not implemented for type {type(model_input)}')

    def get_input_tensors(self, model_input) -> List[Tensor]:
        """
        Recursively go through model input and grab all tensors, which are then used to record current device random
        states. This method will do its best to parse types of Tensor, tuple, list, dict and UserDict. Other types will
        be ignored unless self._get_input_tensors_strict is set to True, in which case an exception will be raised.
        :param model_input: input to model
        :return: all torch tensors in model_input
        """
        if isinstance(model_input, Tensor):
            return [model_input]

        elif isinstance(model_input, (list, tuple)):
            return sum((self.get_input_tensors(x) for x in model_input), [])

        elif isinstance(model_input, (dict, UserDict)):
            return sum((self.get_input_tensors(x) for x in model_input.values()), [])

        elif self._get_input_tensors_strict:
            raise NotImplementedError(f'get_input_tensors not implemented for type {type(model_input)}')

        else:
            return []

    def get_reps(self, model_out) -> Tensor:
        """
        Return representation tensor from generic model output
        :param model_out: generic model output
        :return: a single tensor corresponding to the model representation output
        """
        if self.get_rep_fn is not None:
            return self.get_rep_fn(model_out)
        else:
            return model_out

    def forward_no_grad(
            self,
            model: nn.Module,
            model_inputs,
    ) -> [Tensor, List[RandContext]]:
        """
        The first forward pass without gradient computation.
        :param model: Encoder model.
        :param model_inputs: Model input already broken into chunks.
        :return: A tuple of a) representations and b) recorded random states.
        """
        rnd_states = []
        model_reps = []

        with torch.no_grad():
            for x in model_inputs:
                rnd_states.append(RandContext(*self.get_input_tensors(x)))
                y = self.model(x)
                model_reps.append(self.get_reps(y))

        # concatenate all sub-batch representations
        model_reps = torch.cat(model_reps, dim=0)
        return model_reps, rnd_states

    def build_cache(self, reps: Tensor, **loss_kwargs) -> [List[Tensor], Tensor]:
        """
        Compute the gradient cache
        :param reps: Computed representations from all encoder models
        :param loss_kwargs: Extra keyword arguments to the loss function
        :return: A tuple of a) gradient cache for each encoder model, and b) loss tensor
        """
        reps = [r.detach().requires_grad_() for r in reps]
        with autocast() if self.fp16 else nullcontext():
            loss = self.loss(reps, **loss_kwargs)

        if self.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        cache = [r.grad for r in reps]

        return cache, loss.detach()

    def forward_backward(
            self,
            model: nn.Module,
            model_inputs,
            cached_gradients: List[Tensor],
            random_states: List[RandContext],
            no_sync_except_last: bool = False
    ):
        """
        Run the second forward and the backward pass to compute gradient for a model.
        :param model: Encoder model.
        :param model_inputs: Chunked input to the encoder model.
        :param cached_gradients: Chunked gradient cache tensor for each input.
        :param random_states: Each input's device random state during the first forward.
        :param no_sync_except_last: If True, under distributed setup, only trigger gradient reduction across processes
        for the last sub-batch's forward-backward pass.
        """
        if no_sync_except_last:
            sync_contexts = [model.no_sync for _ in range(len(model_inputs) - 1)] + [nullcontext]
        else:
            sync_contexts = [nullcontext for _ in range(len(model_inputs))]

        for x, state, gradient, sync_context in zip(model_inputs, random_states, cached_gradients, sync_contexts):
            with sync_context():
                with state:
                    y = self.model(x)
                reps = self.get_reps(y)

                surrogate = torch.dot(reps.flatten(), gradient.flatten())
                surrogate.backward()  # self.loss.accelerator.backward(surrogate)

    def cache_step(
            self,
            model_inputs: Iterable[Dict[str, Tensor]],
            labels: Tensor,
            no_sync_except_last: bool = False
    ) -> Tensor:
        """
        Run a cached step to compute gradient over the inputs.
        :param model_inputs: Input to each encoder model. Should be in similar order as the class's model.
        :param no_sync_except_last: If True, under distributed setup, for each model, only trigger gradient reduction
        across processes for the last sub-batch's forward-backward pass.
        :param loss_kwargs: Additional keyword arguments to the loss function.
        :return: The current's loss.
        """
        all_reps = []
        all_rnd_states = []

        if no_sync_except_last:
            assert isinstance(self.model, nn.parallel.DistributedDataParallel), \
                'Some of models are not wrapped in DistributedDataParallel. Make sure you are running DDP with ' \
                'proper initializations.'

        model_inputs = [self.split_inputs(x, self.chunk_size) for x in model_inputs]

        for x in model_inputs:
            model_reps, rnd_states = self.forward_no_grad(self.model, x)
            all_reps.append(model_reps)
            all_rnd_states.append(rnd_states)

        cache, loss = self.build_cache(all_reps, labels=labels)
        cache = [c.split(self.chunk_size) for c in cache]

        for x, model_cache, rnd_states in tqdm(zip(model_inputs, cache, all_rnd_states), desc="Chunk", leave=False, total=len(model_inputs)):
            self.forward_backward(self.model, x, model_cache, rnd_states, no_sync_except_last=no_sync_except_last)

        return loss
