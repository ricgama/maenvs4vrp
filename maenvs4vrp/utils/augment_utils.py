import torch
import os
from typing import Tuple
from tensordict import TensorDict
from torch import Tensor


def rotate_coord_augment(self, xy_data, origin=torch.tensor([0, 0]), seed=None):
        """Rotate coords around a given point.

        Args:
            coords: coords tensor

        """
        if seed is not None:
            self.generator._set_seed(seed)

        origin = origin.to(self.generator.device)
        angle = torch.FloatTensor(*self.generator.batch_size, 1).uniform_(0, 2*torch.pi).to(self.generator.device)
        adjusted_xy = (xy_data - origin)
        c, s = torch.cos(angle).to(self.generator.device), torch.sin( angle).to(self.generator.device)
        m = torch.cat([torch.cat([c, -s], axis=1).unsqueeze(-1) , torch.cat([s , c], axis=1).unsqueeze(-1)], axis=-1)
        nxy_data = origin + torch.transpose(torch.matmul(m, torch.transpose(adjusted_xy, 2, 1)), 2, 1)
        return nxy_data


def random_coord_unit_square_augment(coords):
    """
    coords: Tensor of shape (B, N, 2) - Batch, Nodes, [x, y]
    """
    B, N, _ = coords.shape
    x, y = coords.split(1, dim=2)

    # Pre-calculate all 8 possible outcomes
    # Shape: (8, B, N, 2)
    options = torch.stack([
        torch.cat([x, y], dim=-1),           # 0: (x, y)
        torch.cat([y, x], dim=-1),           # 1: (y, x)
        torch.cat([x, 1 - y], dim=-1),       # 2: (x, 1-y)
        torch.cat([y, 1 - x], dim=-1),       # 3: (y, 1-x)
        torch.cat([1 - x, y], dim=-1),       # 4: (1-x, y)
        torch.cat([1 - y, x], dim=-1),       # 5: (1-y, x)
        torch.cat([1 - x, 1 - y], dim=-1),   # 6: (1-x, 1-y)
        torch.cat([1 - y, 1 - x], dim=-1)    # 7: (1-y, 1-x)
    ], dim=0)

    # Randomly pick an index for each element in the batch
    indices = torch.randint(0, 8, (B,), device=coords.device)

    # Use advanced indexing to pick one transform per batch item
    # Result shape: (B, N, 2)
    batch_indices = torch.arange(B, device=coords.device)
    augmented_coords = options[indices, batch_indices]
    return augmented_coords

def augment_coord_by_8_fold(xy_data):
    # from: https://rl4co.readthedocs.io/en/v0.1.1/_modules/rl4co/models/zoo/pomo/augmentations.html
    # xy_data.shape = (batch_s, problem, 2)

    # [batch, graph, 2]
    x, y = xy_data.split(1, dim=2)
    # x,y shape = (batch, problem, 1)
    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1-x, y), dim=2)
    dat3 = torch.cat((x, 1-y), dim=2)
    dat4 = torch.cat((1-x, 1-y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1-y, x), dim=2)
    dat7 = torch.cat((y, 1-x), dim=2)
    dat8 = torch.cat((1-y, 1-x), dim=2)

    data_augmented = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape = (8*batch, problem, 2)
    return data_augmented

def time_shift_augm(self, time_windows, delta = 1, seed=None):

    """Add time shift to TW.

    Args:
        time_windows: time_windows tensor

    """
    if seed is not None:
        self.generator._set_seed(seed)

    time_shift = torch.FloatTensor(*self.generator.batch_size, 1).uniform_(-delta, delta).to(self.generator.device)
    ntime_windows =  time_windows+time_shift.unsqueeze(-1)
    return ntime_windows
