from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union
import torch
import torch.nn as nn
import numpy as np


class TurnOverDropout(nn.Module):
    def __init__(self, size: Union[int, Tuple], p: float = 0.5, seed: int = 777):
        super(TurnOverDropout, self).__init__()
        self.size = size
        self.p = p
        self.seed = seed
        if self.seed != 777:
            print("tod seed is changed from 777 to", self.seed)

        # make a random projection for hashing; this does not have to be updated (and even saved)
        proj = (torch.rand((size, ), generator=torch.Generator("cpu").manual_seed(seed)) + 0.1) * 1000.0
        self.proj = nn.parameter.Parameter(proj, requires_grad=False)

    def forward(self, x: torch.Tensor, indices: Optional[torch.Tensor] = None, flips: Optional[Union[torch.Tensor, bool]] = None):
        if indices is not None:
            # randomize indices as binary d-dim vectors. arbitrary function can be used.
            indices = indices + 199
            indices, proj = torch.broadcast_tensors(
                indices.float()[..., None],
                self.proj.view((1, ) * indices.ndim + (self.size, )))
            masks = (torch.sin(indices * self.proj) * 1000.0 % 2).int().float()
            if flips is not None:
                if isinstance(flips, torch.Tensor):
                    assert flips.shape == indices.shape
                    masks = torch.where(flips[..., None], 1.0 - masks, masks)
                else:
                    if flips:
                        masks = 1 - masks
                x = x * masks / (1.0 - self.p)
            else:
                x = x * masks / self.p
        else:
            assert flips is None
        return x


class SequentialWithArgs(nn.Sequential):
    def forward(self, input, *args, **kwargs):
        for module in self:
            assert hasattr(module, "forward")
            # This checks the module's forward has "indices" argument or not.
            if "indices" in module.forward.__code__.co_varnames[:module.forward.__code__.co_argcount]:
                input = module(input, *args, **kwargs)
            else:
                input = module(input)
        return input


def make_sequential(types_str, size, p=0.5):
    L = []
    for s in types_str:
        if s == "t":
            L.append(TurnOverDropout(size, p=p))
        elif s == "d":
            L.append(torch.nn.Dropout(p=0.1))
        elif s == "l":
            assert isinstance(size, int), size
            L.append(torch.nn.Linear(size, size, bias=False))
        elif s == "r":
            L.append(torch.nn.ReLU())
        else:
            raise NotImplementedError()
    return SequentialWithArgs(*L)


def make_dataset_with_index(dataset_class):
    # DatasetWithIndex(torchvision.datasets.CIFAR10)
    class DatasetWithIndex(dataset_class):
        def __getitem__(self, index):
            sample = super(DatasetWithIndex, self).__getitem__(index)
            if isinstance(sample, tuple):
                return sample + (index, )
            else:
                return sample, index
    return DatasetWithIndex


if __name__ == "__main__":
    pass
