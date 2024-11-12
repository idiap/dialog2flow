"""
Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
from functools import wraps
from frozendict import frozendict
from typing import Iterable, Hashable


def hash_args(func):
    """
    Decorator to force function arguments to be hashable
    (useful to be compatible with cache)
    """
    def make_hashable(obj):
        if isinstance(obj, Hashable):
            return obj
        elif isinstance(obj, dict):
            return frozendict({k: make_hashable(v) for k, v in obj.items()})
        elif isinstance(obj, Iterable):
            return tuple(make_hashable(v) for v in obj)
        else:
            raise TypeError(f"hash_args decorator does not support '{type(obj)}' datatype.")

    @wraps(func)
    def wrapped(*args, **kwargs):
        args = tuple(make_hashable(arg) for arg in args)
        kwargs = {k: make_hashable(v) for k, v in kwargs.items()}
        return func(*args, **kwargs)
    return wrapped
