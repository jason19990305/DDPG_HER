import numpy as np 
import torch




class ReplayBuffer:
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.max_length = args.buffer_size
        self.normalization = args.normalization
        self.size = 0       
        self.ptr = 0 
        
        # allocate numpy arrays as float32
        self.obs = np.zeros((self.max_length, args.obs_dim), dtype=np.float32)
        self.ach_goal = np.zeros((self.max_length, args.ach_goal_dim), dtype=np.float32)
        self.des_goal = np.zeros((self.max_length, args.des_goal_dim), dtype=np.float32)
        self.a = np.zeros((self.max_length, args.action_dim), dtype=np.float32)
        self.r = np.zeros((self.max_length, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.max_length, args.obs_dim), dtype=np.float32)
        self.next_ach_goal = np.zeros((self.max_length, args.ach_goal_dim), dtype=np.float32)
        self.next_des_goal = np.zeros((self.max_length, args.des_goal_dim), dtype=np.float32)
        self.done = np.zeros((self.max_length, 1), dtype=np.float32)

    def store(self, obs, ach_goal, des_goal, a, r, next_obs, next_ach_goal, next_des_goal, done):
        """Store a transition into replay buffer (goal-conditioned)"""
        self.obs[self.ptr] = obs
        self.ach_goal[self.ptr] = ach_goal
        self.des_goal[self.ptr] = des_goal
        self.a[self.ptr] = a
        self.r[self.ptr] = np.array(r)
        self.next_obs[self.ptr] = next_obs
        self.next_ach_goal[self.ptr] = next_ach_goal
        self.next_des_goal[self.ptr] = next_des_goal
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_length
        self.size = min(self.size + 1, self.max_length)

    def store_batch(self, obs, ach_goal, des_goal, a, r, next_obs, next_ach_goal, next_des_goal, done):
        """Store a batch of transitions into replay buffer"""
        batch_size = len(obs)
        
        # Ensure r and done have the correct shape (n, 1)
        if r.ndim == 1:
            r = r.reshape(-1, 1)
        if done.ndim == 1:
            done = done.reshape(-1, 1)
        
        indices = np.arange(self.ptr, self.ptr + batch_size) % self.max_length
        
        self.obs[indices] = obs
        self.ach_goal[indices] = ach_goal
        self.des_goal[indices] = des_goal
        self.a[indices] = a
        self.r[indices] = r
        self.next_obs[indices] = next_obs
        self.next_ach_goal[indices] = next_ach_goal
        self.next_des_goal[indices] = next_des_goal
        self.done[indices] = done
        
        self.ptr = (self.ptr + batch_size) % self.max_length
        self.size = min(self.size + batch_size, self.max_length)
  
  
    def sample_minibatch(self, state_normalizer, goal_normalizer):
        """Sample a minibatch and return as torch tensors"""
        index = np.random.randint(0, self.size, self.batch_size)
        
        # Apply normalization if normalizers are provided
        if self.normalization:
            obs = torch.from_numpy(state_normalizer(self.obs[index])).float()
            ach_goal = torch.from_numpy(goal_normalizer(self.ach_goal[index])).float()
            des_goal = torch.from_numpy(goal_normalizer(self.des_goal[index])).float()
            next_obs = torch.from_numpy(state_normalizer(self.next_obs[index])).float()
            next_ach_goal = torch.from_numpy(goal_normalizer(self.next_ach_goal[index])).float()
            next_des_goal = torch.from_numpy(goal_normalizer(self.next_des_goal[index])).float()
        else:
            # No normalization
            obs = torch.from_numpy(self.obs[index]).float()
            ach_goal = torch.from_numpy(self.ach_goal[index]).float()
            des_goal = torch.from_numpy(self.des_goal[index]).float()
            next_obs = torch.from_numpy(self.next_obs[index]).float()
            next_ach_goal = torch.from_numpy(self.next_ach_goal[index]).float()
            next_des_goal = torch.from_numpy(self.next_des_goal[index]).float()
        
        a = torch.from_numpy(self.a[index]).float()
        r = torch.from_numpy(self.r[index]).float()
        done = torch.from_numpy(self.done[index]).float()
        
        return obs, ach_goal, des_goal, a, r, next_obs, next_ach_goal, next_des_goal, done
