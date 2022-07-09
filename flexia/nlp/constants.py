import torch
import numpy as np


supported_dtypes = {
    "pt": torch.tensor,
    "np": np.asarray,
    "list": list,
}