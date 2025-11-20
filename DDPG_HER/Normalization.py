import numpy as np
import torch

class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape: the dimension of input data
        self.n = 0
        self.shape = shape
        self.mean = np.zeros(shape, dtype=np.float32)
        self.S = np.zeros(shape, dtype=np.float32)
        self.std = np.ones(shape, dtype=np.float32)

    def update(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        else:
            x = x.reshape(-1, self.shape)
        
        batch_mean = np.mean(x, axis=0)
        batch_std = np.std(x, axis=0)
        batch_size = x.shape[0]
        
        self.n += batch_size
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * batch_size / self.n
        self.S = self.S + batch_size * (batch_std ** 2 + delta ** 2 * batch_size * (self.n - batch_size) / (self.n ** 2))
        self.std = np.sqrt(self.S / self.n)
        

            
class Normalization:
    def __init__(self, shape, warmup_steps=5000):
        self.running_ms = RunningMeanStd(shape=shape)
        self.warmup_count = 0
        self.warmup_steps = warmup_steps 

    def update(self, x: np.ndarray):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            self.warmup_count += 1
        else:
            self.warmup_count += x.shape[0]
        
        self.running_ms.update(x)
        
    def __call__(self, x: np.ndarray):
        if self.warmup_count < self.warmup_steps:
            return x  # No normalization during warm-up phase
        # Convert mean and std to tensor for computation
        mean = self.running_ms.mean
        std = self.running_ms.std
        x_normalized = (x - mean) / (std + 1e-8)
        return x_normalized