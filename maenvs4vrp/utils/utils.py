import torch
from typing import Tuple
from tensordict import TensorDict
from torch import Tensor


def data_equivalence(data_1, data_2, exact: bool = False) -> bool:
    # adapted from https://gymnasium.farama.org/main/_modules/gymnasium/utils/env_checker/
    """Assert equality between data 1 and 2, i.e observations, actions, info.

    Args:
        data_1: Data structure 1
        data_2: Data structure 2
        exact: Whether to compare array exactly or not if false compares with absolute and realive torrelance of 1e-5 (for more information check [np.allclose](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html)).

    Returns:
        If observation 1 and 2 are equivalent
    """
    if type(data_1) is not type(data_2):
        return False
    if isinstance(data_1, dict) or isinstance(data_1, TensorDict):
        return data_1.keys() == data_2.keys() and all(
            data_equivalence(data_1[k], data_2[k], exact) for k in data_1.keys()
        )
    elif isinstance(data_1, (tuple, list)):
        return len(data_1) == len(data_2) and all(
            data_equivalence(o_1, o_2, exact) for o_1, o_2 in zip(data_1, data_2)
        )
    elif isinstance(data_1, Tensor):
        if data_1.shape == data_2.shape and data_1.dtype == data_2.dtype:
            if data_1.dtype == object:
                return all(
                    data_equivalence(a, b, exact) for a, b in zip(data_1, data_2)
                )
            else:
                if exact:
                    return torch.all(data_1 == data_2)
                else:
                    return torch.allclose(data_1, data_2, rtol=1e-5, atol=1e-5)
        else:
            return False
    else:
        return data_1 == data_2

