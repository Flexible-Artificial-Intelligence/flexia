import torch
from ..constants import infinity


# default best, compare operation, delta operation relatively on the mode.
modes_options = {
    "min": (infinity, torch.lt, torch.add),
    "max": (-infinity, torch.gt, torch.sub)
}