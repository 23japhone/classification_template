import torch
import random
import numpy as np


def setRandomSeed(seed: int = 42,
                  deterministic: bool = False,
                  benchmark: bool = False) -> None: 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True
        