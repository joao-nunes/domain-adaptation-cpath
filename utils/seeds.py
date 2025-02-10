import torch
import numpy as np
import random

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def same_seed_random(seed):
    random.seed(seed+5)
    np.random.seed(seed+7)
    torch.manual_seed(seed+9)
    torch.cuda.manual_seed(seed+11)
    torch.cuda.manual_seed_all(seed+13)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    # torch.use_deterministic_algorithms(True)
