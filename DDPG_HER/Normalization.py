import numpy as np

class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape:int):  # shape: the dimension of input data
        # Ensure shape is always a tuple to prevent TypeError
        if isinstance(shape, int):
            shape = (shape,)
        self.n = 0
        self.shape = shape
        self.mean = np.zeros(shape, dtype=np.float32)
        self.S = np.zeros(shape, dtype=np.float32)
        self.std = np.ones(shape, dtype=np.float32)

    def update(self, x):
        x = np.asarray(x, dtype=np.float32)
        # Flatten the input data to (batch_size, feature_dim)
        x = x.reshape(-1, self.shape[0])
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        # Vectorized update of mean and variance using Parallel Algorithm
        delta = batch_mean - self.mean
        tot_count = self.n + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        # Update the sum of squared differences
        new_S = self.S + batch_var * batch_count + np.square(delta) * self.n * batch_count / tot_count

        self.mean = new_mean
        self.S = new_S
        self.n = tot_count
        # Update std, using a small epsilon to prevent division by zero
        self.std = np.sqrt(self.S / self.n) if self.n > 1 else np.ones(self.shape, dtype=np.float32)
            
class Normalization:
    def __init__(self, shape, warmup_steps=5000):
        self.running_ms = RunningMeanStd(shape=shape)
        self.warmup_count = 0
        self.warmup_steps = warmup_steps 

    def update(self, x: np.ndarray):
        self.warmup_count += x.reshape(-1, self.running_ms.shape[0]).shape[0]
        self.running_ms.update(x)
        
    def __call__(self, x: np.ndarray , clip_range=None):
        if self.warmup_count < self.warmup_steps:
            return x  # No normalization during warm-up phase
        
        # --- DEBUGGING INFO ---
        #print(f"--- Normalization __call__ Debug ---")
        #print(f"Input x shape: {x.shape}")
        #print(f"Mean shape: {self.running_ms.mean.shape}")
        #print(f"Std shape: {self.running_ms.std.shape}")
        
        # Normalize the input data
        mean = self.running_ms.mean
        std = self.running_ms.std
        x_normalized = (x - mean) / (std + 1e-8)
        
        #print(f"Output x_normalized shape: {x_normalized.shape}")
        #print(f"------------------------------------")
        
        if clip_range is not None:
            x_normalized = np.clip(x_normalized, -clip_range, clip_range)
        return x_normalized